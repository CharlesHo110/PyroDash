#!/usr/bin/env python3
"""serve_small.py —— 用 transformers 起一个 OpenAI 兼容的小模型服务（替代 vLLM）

为什么需要它：
  vLLM 只支持 Linux（本机需 WSL2 + 管理员）。本脚本在 **Windows 原生** 上提供
  math_eval.py 所需的 `/v1/completions` 接口，从而无需管理员权限即可跑通评测。
  配合 torch 2.7.1+cu118 可直接用 RTX 3060（当前驱动 537.70 支持 CUDA 11.8，无需升级）。

必须兼容的 vLLM 专有语义（math_eval.py 的请求体里用到）：
  - "stop": ["<|llm_offload|>"]       命中即停止
  - "include_stop_str_in_output": true 停止串要保留在 text 里
  - "skip_special_tokens": false       特殊 token 不能被吃掉（否则看不到 <|llm_offload|>）
  响应需含：choices[0].text、choices[0].stop_reason、usage{prompt,completion,total}_tokens

用法：
  python setup/serve_small.py --model models/PyroDash-4B-GRPO-Lambda-0.05 --port 8001
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

try:  # qwen3_5 是 VL 条件生成架构，该类仅存在于较新版本
    from transformers import AutoModelForConditionalGeneration
except ImportError:  # pragma: no cover
    AutoModelForConditionalGeneration = None

try:  # VL 架构对应的 Auto 类（本模型 checkpoint 就是这种命名）
    from transformers import AutoModelForImageTextToText
except ImportError:  # pragma: no cover
    AutoModelForImageTextToText = None

MODEL = None
TOKENIZER = None
MODEL_NAME = "small-model"
LOCK = threading.Lock()


# ---------------------------------------------------------------- 模型加载
def load_model(model_path: str, dtype: str, device: str):
    global MODEL, TOKENIZER
    print(f"[load] tokenizer: {model_path}", flush=True)
    TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]
    print(f"[load] model: dtype={dtype} device={device}", flush=True)

    # qwen3_5 是 VL 条件生成架构，逐个尝试 Auto 类
    last_err = None
    candidates = []
    if AutoModelForImageTextToText is not None:
        # 本模型的 architectures 是 Qwen3_5ForConditionalGeneration，权重键名为 model.language_model.*
        candidates.append(("AutoModelForImageTextToText", AutoModelForImageTextToText))
    candidates.append(("AutoModelForCausalLM", AutoModelForCausalLM))
    if AutoModelForConditionalGeneration is not None:
        candidates.append(
            ("AutoModelForConditionalGeneration", AutoModelForConditionalGeneration))
    candidates.append(("AutoModel", AutoModel))

    for cls_name, cls in candidates:
        try:
            print(f"[load] 尝试 {cls_name} ...", flush=True)
            m = cls.from_pretrained(
                model_path, torch_dtype=torch_dtype, trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            m = m.to(device).eval()
            MODEL = m
            print(f"[load] ✅ 用 {cls_name} 加载成功", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            last_err = f"{cls_name}: {type(e).__name__}: {e}"
            print(f"[load] ❌ {last_err}", flush=True)
    if MODEL is None:
        raise RuntimeError(f"所有 Auto 类都加载失败: {last_err}")

    n = sum(p.numel() for p in MODEL.parameters())
    print(f"[load] 参数量 {n/1e9:.2f}B", flush=True)
    if device.startswith("cuda"):
        free, total = torch.cuda.mem_get_info()
        print(f"[load] 显存 {free/2**30:.1f}GB 空闲 / {total/2**30:.1f}GB 总量", flush=True)


# ------------------------------------------------------- 停止串提前终止
class StopStringsCriteria(StoppingCriteria):
    """生成过程中命中停止串即停，避免白跑满 max_tokens（4B 模型上很关键）。"""

    def __init__(self, tokenizer, stops: list[str], prompt_len: int, window: int = 32):
        self.tok = tokenizer
        self.stops = [s for s in stops if s]
        self.prompt_len = prompt_len
        self.window = window

    def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: ANN001
        if not self.stops:
            return False
        new = input_ids[0][self.prompt_len:][-self.window:]
        txt = self.tok.decode(new, skip_special_tokens=False)
        return any(s in txt for s in self.stops)


def truncate_at_stop(text: str, stops: list[str], include: bool) -> tuple[str, str | None]:
    best = None
    for s in stops:
        if not s:
            continue
        i = text.find(s)
        if i != -1 and (best is None or i < best[0]):
            best = (i, s)
    if best is None:
        return text, None
    i, s = best
    return (text[: i + len(s)] if include else text[:i]), s


# ---------------------------------------------------------------- 生成
def generate(body: dict) -> dict:
    prompt = body.get("prompt") or ""
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""
    max_tokens = int(body.get("max_tokens") or 512)
    stops = body.get("stop") or []
    if isinstance(stops, str):
        stops = [stops]
    include_stop = bool(body.get("include_stop_str_in_output", False))
    skip_special = bool(body.get("skip_special_tokens", True))
    temperature = float(body.get("temperature", 0.0) or 0.0)

    enc = TOKENIZER(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(MODEL.device)
    prompt_len = input_ids.shape[1]

    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        stopping_criteria=StoppingCriteriaList(
            [StopStringsCriteria(TOKENIZER, stops, prompt_len)]
        ),
        pad_token_id=TOKENIZER.pad_token_id or TOKENIZER.eos_token_id,
    )
    if temperature > 0:
        gen_kwargs.update(temperature=max(temperature, 1e-5), top_p=float(body.get("top_p", 1.0)))

    t0 = time.perf_counter()
    am = enc.get("attention_mask")
    with LOCK, torch.inference_mode():
        out = MODEL.generate(
            input_ids=input_ids,
            attention_mask=am.to(MODEL.device) if am is not None else None,
            **gen_kwargs,
        )
    dt = time.perf_counter() - t0

    new_ids = out[0][prompt_len:]
    n_new = int(new_ids.shape[0])
    text = TOKENIZER.decode(new_ids, skip_special_tokens=skip_special)
    text, stop_reason = truncate_at_stop(text, stops, include_stop)

    print(
        f"[gen] prompt={prompt_len}tok 生成={n_new}tok 用时={dt:.1f}s "
        f"({n_new/dt:.2f} tok/s) stop_reason={stop_reason!r}",
        flush=True,
    )
    return {
        "id": f"cmpl-{int(time.time()*1000)}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "text": text,
                "logprobs": None,
                "finish_reason": "stop" if stop_reason else "length",
                "stop_reason": stop_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_len,
            "completion_tokens": n_new,
            "total_tokens": prompt_len + n_new,
        },
    }


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # 静默默认访问日志
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._send(200, {"object": "list", "data": [
                {"id": MODEL_NAME, "object": "model", "created": 0, "owned_by": "local"}]})
        elif self.path.rstrip("/") in ("/health", "/v1/health"):
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": {"message": f"未知路径 {self.path}"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") not in ("/v1/completions", "/completions"):
            self._send(404, {"error": {"message": f"未知路径 {self.path}（仅支持 /v1/completions）"}})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            self._send(200, generate(body))
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self._send(500, {"error": {"message": f"{type(e).__name__}: {e}"}})


def main() -> None:
    global MODEL_NAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--served-model-name", default="small-model")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    MODEL_NAME = args.served_model_name
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = args.dtype
    if dtype == "auto":
        dtype = "bfloat16" if device.startswith("cuda") else "float32"

    print(f"[info] torch={torch.__version__} cuda={torch.version.cuda} "
          f"可用={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[info] GPU={torch.cuda.get_device_name(0)}", flush=True)

    load_model(args.model, dtype, device)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] http://{args.host}:{args.port}/v1  model={MODEL_NAME} device={device}",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
