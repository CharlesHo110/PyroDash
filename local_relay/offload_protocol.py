"""PyroDash offload 协议 —— 让通用小模型学会「做到哪算哪，剩下的交出去」。

背景
----
PyroDash-4B 的 offload 能力是 GRPO 强化学习训出来的（λ 控制倾向强弱）。
通用开源模型没有这个能力，也不会自发地在不确定时停下。

本模块用 system prompt 把**同一个协议**教给通用模型，配合 llama.cpp 的
``stop`` + ``stop_type`` 字段实现等价语义：

    POST /completion  {stop: [OFFLOAD_TAG]}
        ├─ 响应 stop_type == "word"  →  撞到 OFFLOAD_TAG  →  接力给大模型
        └─ 响应 stop_type == "eos"   →  小模型自己答完了  →  直接返回

为什么不用 vLLM 那套
-------------------
PyroDash 原实现走 vLLM 的 ``include_stop_str_in_output: true``，靠让标记
**留在输出里**来检测 offload。llama.cpp 不支持该参数（stop 词一律从
completion 中剔除），但它的 ``stop_type`` / ``stopping_word`` 字段能
**无歧义地**区分「撞到 stop 词」和「自然结束」——因此这里反而更干净：
``content`` 本身就已经是标记之前的部分，不需要再剥一层。

对应关系见 ``relay_server.py`` 的 ``call_small``。
"""

from __future__ import annotations

import os

#: 交接标记。沿用 PyroDash 的符号，语义完全一致。
OFFLOAD_TAG = os.environ.get("PYRODASH_OFFLOAD_TAG", "<|llm_offload|>")

#: 教给通用模型的协议说明。可用 PYRODASH_PROTOCOL_FILE 指向自定义文件覆盖。
PROTOCOL_PROMPT = """你是一台本机小模型，在一个两级协作系统里担任第一线响应者。

【你的能力边界】
- 能独立完成的：简单问答、改写润色、总结、翻译、格式转换、常识判断、短代码片段、简单推理。
- 应该交出去的：需要大量领域知识或最新事实、需要长链条精确推理（多步数学计算、复杂算法设计、
  长篇代码），以及任何你没有把握的场景。

【交接协议 —— 最重要】
当判断该交出去时，你必须：
1. 先把「你已经确定能推进的部分」写出来。这部分内容会原样交给远端大模型，让它从你停下的
   地方接着做，而不是从头重来。所以你做得越多，整个系统越省。
2. 然后输出标记 {tag}，并立即停止，之后不要再输出任何内容。

如果你能完整可靠地完成任务，就直接给出答案，不要输出标记。

【不要做的事】
- 不要为了保险而在开头就输出标记——那等于直接放弃，浪费了你的价值。
- 不要输出标记后继续写字（标记之后的内容会被丢弃）。
- 不要在回答里提到这个协议、这个标记，或"大模型"的存在。
""".format(tag=OFFLOAD_TAG)


def get_protocol_prompt() -> str:
    """返回协议 prompt；若环境变量指定了文件则读文件。"""
    override = os.environ.get("PYRODASH_PROTOCOL_FILE")
    if override and os.path.isfile(override):
        with open(override, encoding="utf-8") as fh:
            return fh.read()
    return PROTOCOL_PROMPT


def inject_protocol(messages: list[dict]) -> list[dict]:
    """把协议注入到 messages。

    客户端若已带 system 消息，则**合并**（协议在前，客户端的在后）而不是覆盖，
    避免抹掉调用方自己的设定。
    """
    msgs = [dict(m) for m in messages]
    idx = next((i for i, m in enumerate(msgs) if m.get("role") == "system"), None)

    if idx is None:
        return [{"role": "system", "content": get_protocol_prompt()}] + msgs

    original = msgs[idx].get("content", "")
    if isinstance(original, list):  # 多模态分段格式：只取文本段
        original = "\n".join(
            str(p.get("text", ""))
            for p in original
            if isinstance(p, dict) and p.get("type") == "text"
        )

    merged = get_protocol_prompt()
    if str(original).strip():
        merged = f"{merged}\n\n---\n\n{original}"
    msgs[idx]["content"] = merged
    return msgs
