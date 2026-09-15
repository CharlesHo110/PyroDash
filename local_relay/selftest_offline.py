#!/usr/bin/env python
"""selftest_offline.py — 离线自检：不碰 GPU、不碰 llama-server。

验证 relay 的两个纯逻辑环节：
  1) 协议注入（system 消息合并是否正确）
  2) 接力消息构造（<part_think> 是否按 PyroDash 原语义生成）

    python local_relay/selftest_offline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "evaluation" / "evaluation_math"))

from offload_protocol import OFFLOAD_TAG, inject_protocol  # noqa: E402
from llm_relay import _build_offload_messages, _llm_max_tokens  # noqa: E402

PASS, FAIL = "  ✅", "  ❌"
failures = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global failures
    print(f"{PASS if cond else FAIL} {name}")
    if detail and not cond:
        print(f"      {detail}")
    if not cond:
        failures += 1


print("=" * 68)
print(" relay 离线自检")
print("=" * 68)

# ---------------------------------------------------------------- 1. 协议注入
print("\n[1] 协议注入")

msgs_no_sys = [{"role": "user", "content": "1+1=?"}]
out = inject_protocol(msgs_no_sys)
check("无 system 时新增一条 system", out[0]["role"] == "system" and len(out) == 2)
check("协议里含交接标记", OFFLOAD_TAG in out[0]["content"])
check("原始 user 消息原样保留", out[1]["content"] == "1+1=?")

msgs_with_sys = [
    {"role": "system", "content": "你是资深 Rust 工程师。"},
    {"role": "user", "content": "写个生命周期例子"},
]
out2 = inject_protocol(msgs_with_sys)
check("有 system 时不新增消息（就地合并）", len(out2) == 2)
check("客户端原 system 内容被保留", "资深 Rust 工程师" in out2[0]["content"])
check("协议也被注入", OFFLOAD_TAG in out2[0]["content"])
check("协议在客户端设定之前", out2[0]["content"].index(OFFLOAD_TAG) < out2[0]["content"].index("资深 Rust"))
check("未污染调用方入参", msgs_with_sys[0]["content"] == "你是资深 Rust 工程师。")

# ------------------------------------------------------- 2. 接力消息构造
print("\n[2] 接力消息构造（<part_think> 协议）")

task = [{"role": "user", "content": "一个班 40 人，男生比女生多 6 人，男生几人？"}]
partial = "设男生 x 人，女生 y 人。由题意：\n x + y = 40\n x - y = 6"
handoff_msgs = _build_offload_messages(task, partial)

roles = [m["role"] for m in handoff_msgs]
check("第一条是 system（接力指令）", roles[0] == "system")
check("含一条 user（原始问题）", "user" in roles)
check("含一条 assistant（半截推理）", "assistant" in roles)

sys_text = handoff_msgs[0]["content"]
assist_text = next(m["content"] for m in handoff_msgs if m["role"] == "assistant")
user_text = next(m["content"] for m in handoff_msgs if m["role"] == "user")

check("<part_think> 包裹小模型产出", "<part_think>" in assist_text and "</part_think>" in assist_text)
check("半截推理内容完整带入", "x - y = 6" in assist_text)
check("原始问题带入 user 消息", "男生比女生多 6 人" in user_text)
check(
    "接力指令要求「从断点接着推、不要重复」（PyroDash 原文为英文）",
    "continue from where it stopped" in sys_text and "Do not repeat" in sys_text,
    f"实际 system 前 120 字: {sys_text[:120]}",
)

# 标记本身不应残留在 <part_think> 里（_offload_prefix 的职责）
handoff_with_tag = _build_offload_messages(task, partial + OFFLOAD_TAG)
assist2 = next(m["content"] for m in handoff_with_tag if m["role"] == "assistant")
check("残留的交接标记被剥掉", OFFLOAD_TAG not in assist2)

# --------------------------------------------------- 3. 共享 token 预算
print("\n[3] 共享 token 预算（规则来自 PyroDash._llm_max_tokens）")

budget = 2048
used = 300
remaining = _llm_max_tokens(budget, [[None] * used], 0)
check(f"用掉 {used} 后剩 {remaining}", remaining == budget - used, f"实际 {remaining}")
check("预算用尽时为 0（非负）", _llm_max_tokens(100, [[None] * 999], 0) == 0)

print("\n" + "=" * 68)
if failures:
    print(f"  ❌ {failures} 项未通过")
else:
    print("  ✅ 全部通过")
print("=" * 68)
raise SystemExit(1 if failures else 0)
