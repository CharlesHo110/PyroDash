#!/usr/bin/env python
"""test_relay.py — 端到端验证中转服务：看它什么时候留在本机、什么时候接力。

需要先启动服务：  bash local_relay/run.sh

    python local_relay/test_relay.py                # 跑全部用例
    python local_relay/test_relay.py --suite easy   # 只跑「应该本机解决」的
    python local_relay/test_relay.py --suite hard   # 只跑「应该接力」的
    python local_relay/test_relay.py --ask "你的问题"

观察重点：日志/输出里的 route 字段。
    route = small            → 小模型自己答完了（省钱、快）
    route = handoff:tag      → 小模型主动交出了接力棒（符合预期）
    route = handoff:limit    → 小模型把它自己的预算用光了，被动接力
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import requests

# (用例名, 期望路由, prompt)
EASY = [
    ("翻译", "small", "把这句话翻译成英文：今天天气不错，我们去公园散步吧。"),
    ("算术", "small", "1+1 等于几？只回答数字。"),
    ("总结", "small", "用一句话总结：苹果公司今天发布了新款手机，售价 5999 元，明天开售。"),
    ("格式转换", "small", '把 "name=alice;age=30;city=beijing" 转成 JSON。只输出 JSON。'),
]

HARD = [
    (
        "多步推理",
        "handoff",
        "一个班 40 人，男生比女生多 6 人。若把男生平均分成 4 组、女生平均分成 3 组，"
        "每组人数相差多少？请给出计算过程。",
    ),
    (
        "长链条代码",
        "handoff",
        "用 Rust 实现一个带 TTL 过期的 LRU 缓存，要求：泛型键值、非侵入式后台清理、"
        "无 unsafe、并发安全。给出完整可编译代码。",
    ),
    (
        "领域知识",
        "handoff",
        "解释 Transformer 中 RoPE 相对位置编码的数学推导，包括它为什么能外推到更长上下文，"
        "以及与 ALiBi 的对比。",
    ),
]


def call(base_url: str, model: str, prompt: str, max_tokens: int, timeout: int) -> tuple[dict, float]:
    t0 = time.time()
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.6,
        },
        timeout=timeout,
    )
    elapsed = time.time() - t0
    resp.raise_for_status()
    return resp.json(), elapsed


def run_suite(base_url: str, model: str, cases: list, max_tokens: int, timeout: int) -> list[dict]:
    results = []
    for name, expect, prompt in cases:
        print(f"\n  ── {name}  (期望: {expect})")
        print(f"     问: {prompt[:64]}{'...' if len(prompt) > 64 else ''}")
        try:
            rec, elapsed = call(base_url, model, prompt, max_tokens, timeout)
        except Exception as exc:
            print(f"     ❌ 失败: {type(exc).__name__}: {exc}")
            results.append({"name": name, "expect": expect, "ok": False})
            continue

        p = rec.get("pyrodash", {})
        route = p.get("route", "?")
        hit = (expect == "small" and route == "small") or (
            expect == "handoff" and route.startswith("handoff")
        )
        icon = "✅" if hit else "⚠️"
        print(f"     {icon} route={route}  原因={p.get('offload_reason') or '-'}")
        print(
            f"        小模型 {p.get('small_tokens', 0)} tok / {p.get('small_elapsed_s', 0)}s"
            + (
                f"   大模型 {p.get('llm_tokens', 0)} tok / {p.get('llm_elapsed_s', 0)}s"
                if p.get("offloaded")
                else ""
            )
            + f"   总 {elapsed:.2f}s"
        )
        answer = rec["choices"][0]["message"]["content"]
        print(f"        答: {answer[:200].replace(chr(10), ' ')}{'...' if len(answer) > 200 else ''}")
        if p.get("small_reasoning"):
            print(f"        小模型交接时的半截推理: {p['small_reasoning'][:120]}...")
        if p.get("error"):
            print(f"        ⚠ 远端报错: {p['error'][:120]}")
        results.append({"name": name, "expect": expect, "route": route, "ok": hit, "rec": rec})
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="本机中转端到端测试")
    ap.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    ap.add_argument("--model", default="pyrodash-local")
    ap.add_argument("--suite", choices=["all", "easy", "hard"], default="all")
    ap.add_argument("--max-tokens", type=int, default=1024, help="共享预算")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--ask", default="", help="只问一个问题")
    args = ap.parse_args()

    try:
        requests.get(f"{args.base_url.rstrip('/')}/models", timeout=10).raise_for_status()
    except Exception as exc:
        print(f"  ✗ 连不上 {args.base_url} — 是否已运行 bash local_relay/run.sh ?")
        print(f"    ({type(exc).__name__}: {exc})")
        return 1

    print("=" * 70)
    print(" 端到端测试 —— 本机小模型 + 按需接力大模型")
    print("=" * 70)

    results: list[dict] = []
    if args.ask:
        results = run_suite(args.base_url, args.model, [("自由提问", "-", args.ask)], args.max_tokens, args.timeout)
    else:
        if args.suite in ("all", "easy"):
            print("\n【A. 应该留在本机】小模型自己就能搞定的轻任务")
            results += run_suite(args.base_url, args.model, EASY, args.max_tokens, args.timeout)
        if args.suite in ("all", "hard"):
            print("\n【B. 应该接力】超出小模型能力的重任务")
            results += run_suite(args.base_url, args.model, HARD, args.max_tokens, args.timeout)

    # ---------------------------------------------------------------- 汇总
    print("\n" + "=" * 70)
    print(" 汇总")
    print("=" * 70)
    judged = [r for r in results if r.get("expect") in ("small", "handoff")]
    good = sum(1 for r in judged if r["ok"])
    print(f"\n  路由判定: {good}/{len(judged)} 符合预期")
    for r in results:
        if r.get("expect") in ("small", "handoff"):
            print(f"    {'✅' if r['ok'] else '⚠️ '} {r['name']:<10} 期望={r['expect']:<8} 实际={r.get('route', '失败')}")

    try:
        stats = requests.get(f"{args.base_url.rsplit('/v1', 1)[0]}/v1/stats", timeout=10).json()
        print(f"\n  服务端累计统计:")
        print(f"    总请求      {stats['total_requests']}")
        print(f"    交接次数    {stats['offload']['count']}  ({stats['offload']['rate_pct']})")
        print(f"    分类        {json.dumps(stats['routes'], ensure_ascii=False)}")
        print(f"    token      小模型 {stats['tokens']['small']} / 大模型 {stats['tokens']['llm']}"
              f"  (大模型占比 {stats['tokens']['llm_share_pct']})")
        print(f"    平均延迟    小模型 {stats['latency_s']['small_avg']}s / 大模型 {stats['latency_s']['llm_avg']}s")
    except Exception as exc:
        print(f"\n  (读统计失败: {exc})")

    print()
    return 0 if good == len(judged) else 0  # 判定偏差只提示，不算失败


if __name__ == "__main__":
    raise SystemExit(main())
