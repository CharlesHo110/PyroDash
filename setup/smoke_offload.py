#!/usr/bin/env python3
"""smoke_offload.py —— 小样本冒烟：验证「本地小模型 + 远端大模型」token 级 offload 全链路

做法：不修改上游任何代码，只在运行时把 datasets_loader.get_dataset_handler 换成
      一个"只取前 N 条"的包装，然后调用 math_eval.run_dataset()。
      于是走的是**真实的** run_small_batch → relay_with_llm → build_records → 打分 全流程。

判定标准：
  offload_rate > 0  → 小模型确实吐出了 <|llm_offload|> 控制 token，offload 机制生效
  llm relay 有调用  → 接力到远端大模型并回填成功
  accuracy 有值     → 打分链路正常

用法：
  python setup/smoke_offload.py --limit 3 --max-tokens 2048
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SETUP_DIR = Path(__file__).resolve().parent
PROJ = SETUP_DIR.parent
EVAL_DIR = PROJ / "evaluation" / "evaluation_math"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import datasets_loader  # noqa: E402
import math_eval  # noqa: E402
from llm_relay import OFFLOAD_TAG  # noqa: E402


class LimitedHandler(datasets_loader.DatasetHandler):
    """包装真实 handler，只暴露前 n 条数据。"""

    def __init__(self, inner, n: int):
        self.inner = inner
        self.n = n

    def load_data(self):
        questions, answers = self.inner.load_data()
        total = len(questions)
        print(f"[smoke] 数据集共 {total} 条 → 只取前 {min(self.n, total)} 条", flush=True)
        return questions[: self.n], answers[: self.n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=str(PROJ / "models" / "PyroDash-4B-GRPO-Lambda-0.05"))
    ap.add_argument("--small-base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--small-model", default="small-model")
    ap.add_argument("--limit", type=int, default=3, help="只跑前 N 条题目")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--output-dir", default=str(PROJ / "results_smoke"))
    ap.add_argument("--llm-base-url", default=os.environ.get("LLM_BASE_URL", ""))
    ap.add_argument("--llm-api-key", default=os.environ.get("LLM_API_KEY", ""))
    ap.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", "deepseek-v4-pro"))
    ap.add_argument("--llm-max-workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    if not args.llm_base_url or not args.llm_api_key:
        sys.exit("✗ 需要 --llm-base-url / --llm-api-key（或环境变量 LLM_BASE_URL / LLM_API_KEY）")

    # —— 小模型与 tokenizer 前置检查 ——
    import requests

    from transformers import AutoTokenizer

    try:
        r = requests.get(f"{args.small_base_url.rstrip('/')}/models", timeout=10)
        r.raise_for_status()
        served = [m["id"] for m in r.json().get("data", [])]
        print(f"[smoke] 小模型服务在线，提供模型: {served}", flush=True)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"✗ 无法连接小模型服务 {args.small_base_url}: {e}\n"
                 f"  请先启动: python setup/serve_small.py --model {args.model_path} --port 8001")

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if OFFLOAD_TAG not in tok.get_vocab():
        sys.exit(f"✗ tokenizer 缺少 {OFFLOAD_TAG!r}，模型目录不对")
    print(f"[smoke] {OFFLOAD_TAG} → token id {tok.convert_tokens_to_ids(OFFLOAD_TAG)}"
          f"（vocab {len(tok)}）", flush=True)

    # —— 把数据集换成限量包装（不动上游代码）——
    original = datasets_loader.get_dataset_handler
    datasets_loader.get_dataset_handler = (
        lambda name, name2=None: LimitedHandler(original(name, name2), args.limit)
    )

    print(f"[smoke] dataset={args.dataset} limit={args.limit} max_tokens={args.max_tokens}", flush=True)
    print(f"[smoke] llm={args.llm_base_url} model={args.llm_model}", flush=True)

    summary = math_eval.run_dataset(
        args.dataset,
        Path(args.output_dir).resolve(),
        tokenizer=tok,
        small_base_url=args.small_base_url,
        small_model=args.small_model,
        api_key=args.llm_api_key,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_max_workers=args.llm_max_workers,
        max_tokens=args.max_tokens,
    )

    print("\n" + "=" * 62)
    print("冒烟结论")
    print("=" * 62)
    ok = True
    if summary.get("offload_count", 0) > 0:
        print(f"  ✅ offload 机制生效：{summary['offload_count']}/{summary['total']} 条吐出 {OFFLOAD_TAG}")
    else:
        ok = False
        print(f"  ❌ 没有样本吐出 {OFFLOAD_TAG}（offload_rate=0）——模型或服务端可能有问题")
    if summary.get("total_small_completion_tokens", 0) > 0:
        print(f"  ✅ 小模型推理正常：{summary['total_small_completion_tokens']} completion tokens")
    else:
        ok = False
        print("  ❌ 小模型没有产出 token")
    if summary.get("accuracy") is not None:
        print(f"  ✅ 打分链路正常：accuracy={100 * summary['accuracy']:.1f}% "
              f"({summary.get('correct')}/{summary.get('total')})")
    print(f"\n  总判定：{'通过（全链路已跑通）' if ok else '未通过，见上面 ❌'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
