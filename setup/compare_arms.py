#!/usr/bin/env python3
"""compare_arms.py —— 计划文档 §4「验证清单」第 3 项：三组对照实验

目的：量化 PyroDash 的「省钱 vs 准确率」取舍，画出成本-准确率点。

三个臂（都跑**同一批题目**，保证可比）：
  ① small     纯小模型下限 —— 同一 4B checkpoint，吐 <|llm_offload|> 就停、不接力
                             （即"不接大模型"时小模型单独能拿到多少分）
  ② pyrodash  PyroDash 主链路 —— 小模型 + offload 接力远端大模型（λ 由 checkpoint 决定）
  ③ llm       纯大模型上限 —— 全部题目直接调远端大模型，不经小模型

指标：
  accuracy                     准确率
  offload_rate                 小模型求助比例（核心：观察 λ 的效果）
  远端大模型 token 总消耗       成本代理（内网网关无实际计费，用 token 量衡量压力）
  总 wall 时间

用法：
  # 起小模型服务后：
  python setup/compare_arms.py --dataset gsm8k --limit 50 --max-tokens 4096

  # 只跑某几个臂：
  python setup/compare_arms.py --dataset gsm8k --limit 50 --arms small llm

  # 换 λ=0.6 checkpoint（需先重启服务指向该模型）：
  python setup/compare_arms.py --model-path models/PyroDash-4B-GRPO-Lambda-0.6 \
      --tag l06 --dataset gsm8k --limit 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

SETUP_DIR = Path(__file__).resolve().parent
PROJ = SETUP_DIR.parent
EVAL_DIR = PROJ / "evaluation" / "evaluation_math"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import datasets_loader  # noqa: E402
import math_eval  # noqa: E402
import requests  # noqa: E402
from llm_relay import OFFLOAD_TAG  # noqa: E402
from tqdm import tqdm  # noqa: E402

ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# ---------------------------------------------------------------- 数据集限量
class LimitedHandler(datasets_loader.DatasetHandler):
    def __init__(self, inner, n: int, offset: int = 0):
        self.inner, self.n, self.offset = inner, n, offset

    def load_data(self):
        questions, answers = self.inner.load_data()
        total = len(questions)
        lo = min(self.offset, total)
        hi = min(lo + self.n, total)
        print(f"[data] 数据集共 {total} 条 → 取 [{lo}, {hi}) 共 {hi - lo} 条", flush=True)
        return questions[lo:hi], answers[lo:hi]


# ---------------------------------------------------------------- 臂 ① 纯小模型
def arm_small(questions, answers, tok, args):
    """只调小模型，吐 offload token 就停，不接力。"""
    raw, _cids, usages = math_eval.run_small_batch(
        questions,
        tokenizer=tok,
        small_base_url=args.small_base_url,
        small_model=args.small_model,
        max_tokens=args.max_tokens,
    )
    # 不接力：completed 就是小模型自己产出的东西
    return math_eval.build_records(
        questions, answers, raw, list(raw), usages, [None] * len(raw)
    )


# ---------------------------------------------------------------- 臂 ② PyroDash
def arm_pyrodash(questions, answers, tok, args):
    """主链路：小模型 → 触发 offload → 接力远端大模型。"""
    raw, cids, usages = math_eval.run_small_batch(
        questions,
        tokenizer=tok,
        small_base_url=args.small_base_url,
        small_model=args.small_model,
        max_tokens=args.max_tokens,
    )
    completed = list(raw)
    llm_usages: list[dict | None] = [None] * len(raw)
    pending = [i for i, r in enumerate(raw) if OFFLOAD_TAG in r]
    if pending:
        sub_done, sub_usage = math_eval.relay_with_llm(
            [raw[i] for i in pending],
            [questions[i] for i in pending],
            api_key=args.llm_api_key,
            completion_ids=[cids[i] for i in pending],
            llm_base_url=args.llm_base_url,
            llm_model=args.llm_model,
            llm_max_workers=args.llm_max_workers,
            max_tokens=args.max_tokens,
        )
        for j, idx in enumerate(pending):
            completed[idx] = sub_done[j]
            llm_usages[idx] = sub_usage[j]
    return math_eval.build_records(questions, answers, raw, completed, usages, llm_usages)


# ---------------------------------------------------------------- 臂 ③ 纯大模型
def _one_llm(question: str, args) -> tuple[str, dict | None]:
    url = f"{args.llm_base_url.rstrip('/')}/chat/completions"
    body = {
        "model": args.llm_model,
        "messages": math_eval.build_chat_messages(question),
        "temperature": math_eval.TEMPERATURE,
        # 上限臂拿到完整预算（没有小模型先消耗）
        "max_tokens": args.max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {args.llm_api_key}",
        "Content-Type": "application/json",
    }
    r = requests.post(url, headers=headers, json=body, timeout=900.0)
    r.raise_for_status()
    d = r.json()
    return (
        str(d["choices"][0]["message"]["content"] or ""),
        math_eval.normalize_usage(d.get("usage")),
    )


def arm_llm(questions, answers, _tok, args):
    """全部直接调远端大模型，不经小模型。"""
    n = len(questions)
    completed = [""] * n
    usages: list[dict | None] = [None] * n
    with ThreadPoolExecutor(max_workers=max(1, args.llm_max_workers)) as pool:
        futs = {pool.submit(_one_llm, q, args): i for i, q in enumerate(questions)}
        for fut in tqdm(as_completed(futs), total=n, desc="[llm ] direct", unit="sample"):
            i = futs[fut]
            text, usage = fut.result()
            completed[i] = text
            usages[i] = usage
    # 小模型用量记 0（没用到）；raw_response 留空，故 has_offload=False
    return math_eval.build_records(
        questions, answers, [""] * n, completed, [dict(ZERO_USAGE)] * n, usages
    )


ARMS = {
    "small": ("纯小模型下限", arm_small),
    "pyrodash": ("PyroDash 主链路", arm_pyrodash),
    "llm": ("纯大模型上限", arm_llm),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=str(PROJ / "models" / "PyroDash-4B-GRPO-Lambda-0.05"))
    ap.add_argument("--small-base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--small-model", default="small-model")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--limit", type=int, default=50, help="题数（计划建议 50~100）")
    ap.add_argument("--offset", type=int, default=0, help="从第几条开始取（换子集用）")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--tag", default="l05", help="输出文件名标签（如 l05 / l06）")
    ap.add_argument("--output-dir", default=str(PROJ / "results_compare"))
    ap.add_argument("--llm-base-url", default=os.environ.get("LLM_BASE_URL", ""))
    ap.add_argument("--llm-api-key", default=os.environ.get("LLM_API_KEY", ""))
    ap.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", "deepseek-v4-pro"))
    ap.add_argument("--llm-max-workers", type=int, default=8)
    ap.add_argument("--small-workers", type=int, default=1,
                    help="并发小模型请求数。服务端 GPU 串行生成，>1 只会让请求排队等锁，不建议调大")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="小模型单题预算；也是 PyroDash 的 小+大 共享总预算")
    args = ap.parse_args()

    if not args.llm_base_url or not args.llm_api_key:
        sys.exit("✗ 需要 --llm-base-url / --llm-api-key（或环境变量 LLM_BASE_URL / LLM_API_KEY）")

    # 服务端是 ThreadingHTTPServer + 全局锁 → 实际串行。并发数调大会让请求排队等锁，
    # 排队时间计入单请求超时，从而在长生成（如 λ=0.6）时误报 ReadTimeout。
    math_eval.SMALL_MAX_WORKERS = max(1, args.small_workers)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if OFFLOAD_TAG not in tok.get_vocab():
        sys.exit(f"✗ tokenizer 缺少 {OFFLOAD_TAG!r}，模型目录不对：{args.model_path}")

    # 小模型服务探活（arm llm 不需要，但一并提示）
    try:
        r = requests.get(f"{args.small_base_url.rstrip('/')}/models", timeout=10)
        r.raise_for_status()
        served = [m["id"] for m in r.json().get("data", [])]
        print(f"[check] 小模型服务在线: {served}", flush=True)
    except Exception as e:  # noqa: BLE001
        if any(a != "llm" for a in args.arms):
            sys.exit(f"✗ 无法连接小模型服务 {args.small_base_url}: {e}")
        print(f"[check] 小模型服务不可用（本次不需要）: {e}", flush=True)

    # —— 一次性载入题目，三个臂共用（保证可比）——
    inner = datasets_loader.get_dataset_handler(args.dataset)
    questions, answers = LimitedHandler(inner, args.limit, args.offset).load_data()

    print(f"[config] model_path={args.model_path}")
    print(f"[config] dataset={args.dataset} limit={args.limit} offset={args.offset}")
    print(f"[config] llm={args.llm_base_url} model={args.llm_model}")
    print(f"[config] max_tokens={args.max_tokens}  arms={args.arms}\n", flush=True)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict] = {}

    for name in args.arms:
        label, fn = ARMS[name]
        print(f"\n{'=' * 62}\n>>> 臂 {name} —— {label}\n{'=' * 62}", flush=True)
        t0 = time.perf_counter()
        records = fn(questions, answers, tok, args)
        wall = time.perf_counter() - t0
        summary = math_eval.summarize_records(f"{args.dataset}:{name}", records, wall)
        summary["arm"] = name
        summary["arm_label"] = label
        summary["tag"] = args.tag
        summary["remote_tokens"] = (
            summary["total_llm_prompt_tokens"] + summary["total_llm_completion_tokens"]
        )
        summary["small_tokens"] = (
            summary["total_small_prompt_tokens"] + summary["total_small_completion_tokens"]
        )
        all_results[name] = summary
        math_eval.print_summary(summary)
        print(f"  远端大模型 token: {summary['remote_tokens']}")
        print(f"  小模型 token:     {summary['small_tokens']}")
        print(f"  wall:             {wall:.1f}s")

        p = out_dir / f"{args.tag}_{args.dataset}_{name}.json"
        with p.open("w", encoding="utf-8") as f:
            json.dump(
                {"summary": summary, "samples": [asdict(r) for r in records]},
                f, ensure_ascii=False, indent=2,
            )
        print(f"  [save] {p}", flush=True)

    # ------------------------------------------------------------ 对比表
    if len(all_results) > 1:
        # 以「纯大模型上限」为成本基准（全部直接调大模型的 token 量）：
        # 相对成本 < 1 表示比全量直调大模型更省；> 1 表示反而更贵。
        base = all_results.get("llm") or all_results.get("pyrodash") or next(iter(all_results.values()))
        base_remote = base["remote_tokens"] or 1
        print("\n" + "=" * 78)
        print(f"三组对照（{args.dataset}，{len(questions)} 题，tag={args.tag}）")
        print(f"成本基准 = 臂 {base['arm']}（{base['remote_tokens']:,} 远端 token）")
        print("=" * 78)
        print(f"{'臂':<18}{'准确率':>10}{'offload率':>11}{'远端token':>12}{'相对成本':>10}{'小模型token':>13}")
        print("-" * 78)
        for name in args.arms:
            s = all_results[name]
            rel = s["remote_tokens"] / base_remote
            print(
                f"{name + ' ' + s['arm_label']:<18}"
                f"{100 * s['accuracy']:>9.2f}%"
                f"{100 * s['offload_rate']:>10.1f}%"
                f"{s['remote_tokens']:>12,}"
                f"{rel:>9.2f}x"
                f"{s['small_tokens']:>13,}"
            )
        print("=" * 78)
        print("相对成本 = 该臂远端大模型 token / 纯大模型上限的 token；<1 表示比全量直调更省")
        print("（内网网关无实际计费，token 量即对网关的压力）")

    combined = out_dir / f"{args.tag}_{args.dataset}_compare.json"
    with combined.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "limit": args.limit,
                "offset": args.offset,
                "tag": args.tag,
                "model_path": args.model_path,
                "llm_model": args.llm_model,
                "max_tokens": args.max_tokens,
                "arms": all_results,
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"\n[save] 对比汇总 {combined}")


if __name__ == "__main__":
    main()
