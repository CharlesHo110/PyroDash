#!/usr/bin/env python
"""relay_server.py — PyroDash 式本机中转服务（OpenAI 兼容）

把「本机小模型先答、搞不定就接力远端大模型」这套 PyroDash 思想，做成一个
任何 OpenAI 客户端（pi / Cursor / 脚本）都能直接用的标准服务。

    ┌──────────────┐  POST /v1/chat/completions
    │ 客户端 / pi  │ ─────────────────────────────┐
    └──────────────┘                              ▼
                                    ┌──────────────────────────┐
                                    │  relay_server (本文件)   │
                                    │  :8010                   │
                                    └───────┬──────────────────┘
                       1) 注入 offload 协议 │
                       2) /apply-template   │  渲染 chat 模板
                       3) /completion       ▼
                                    ┌──────────────────────────┐
                                    │ llama-server  :8080      │
                                    │ Qwen3-4B-Instruct-2507   │
                                    │ stop=[<|llm_offload|>]   │
                                    └───────┬──────────────────┘
                                            │
                              stop_type == "word" ?
                                   ┌────────┴────────┐
                                   否                是
                                   ▼                 ▼
                            直接返回小模型答案   构造 <part_think> 接力
                                                → DeepSeek → 合并返回

复用 PyroDash 原有组件（未改动）：
    evaluation/evaluation_math/llm_relay.py 里的
      - _build_offload_messages()   接力消息构造（<part_think> 协议）
      - _call_dashscope_chat()      远端大模型调用
      - _llm_max_tokens()           两端共享 token 预算规则
      - _offload_prefix()           取标记之前的部分

用法：
    python local_relay/relay_server.py
    python local_relay/relay_server.py --port 8010 --small-base-url http://127.0.0.1:8080

环境变量：
    RELAY_HOST / RELAY_PORT        本服务监听（默认 127.0.0.1:8010）
    SMALL_BASE_URL                 本机 llama-server（默认 http://127.0.0.1:8080）
    SMALL_MODEL                    小模型名（默认取 llama-server 的 /props）
    SMALL_TIMEOUT                  单次小模型超时秒（默认 600）
    LLM_BASE_URL                   远端大模型（默认 https://ai-api.bj.tkoffice.cn/v1）
    LLM_API_KEY                    远端密钥；缺省则读 setup/.llm_key
    LLM_MODEL                      远端模型名（默认 deepseek-v4-pro）
    LLM_ENABLE_THINKING            是否让远端开思考（默认 1）
    DEFAULT_MAX_TOKENS             共享 token 预算默认值（默认 2048）
    OFFLOAD_ON_LIMIT               小模型被截断时是否也接力（默认 1）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------- 复用 PyroDash

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "evaluation" / "evaluation_math"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_relay import (  # noqa: E402  —— 直接复用 PyroDash 未改动的接力组件
    _build_offload_messages,
    _call_dashscope_chat,
    _llm_max_tokens,
    _offload_prefix,
)
from offload_protocol import OFFLOAD_TAG, inject_protocol  # noqa: E402

# ---------------------------------------------------------------------- 配置


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on", "y")


def _resolve_llm_key() -> str:
    """密钥优先取环境变量，其次读 setup/.llm_key（gitignored）。"""
    key = os.environ.get("LLM_API_KEY", "").strip()
    if key:
        return key
    key_file = REPO_ROOT / "setup" / ".llm_key"
    if key_file.is_file():
        return key_file.read_text(encoding="utf-8").strip()
    return ""


CONFIG = {
    "relay_host": _env("RELAY_HOST", "127.0.0.1"),
    "relay_port": _env_int("RELAY_PORT", 8010),
    "small_base_url": _env("SMALL_BASE_URL", "http://127.0.0.1:8080").rstrip("/"),
    "small_model": _env("SMALL_MODEL", ""),
    "small_timeout": _env_int("SMALL_TIMEOUT", 600),
    # 本机小模型的单次生成预算。这是**不训练也能 offload** 的关键：
    # 给得小，它没在预算内答完（stop_type='limit'）就等于交接信号；
    # 在预算内答完则说明本机搞得定，省下大模型的钱。
    "small_max_tokens": _env_int("SMALL_MAX_TOKENS", 512),
    "llm_base_url": _env("LLM_BASE_URL", "https://ai-api.bj.tkoffice.cn/v1").rstrip("/"),
    "llm_api_key": _resolve_llm_key(),
    "llm_model": _env("LLM_MODEL", "deepseek-v4-flash"),
    "llm_timeout": _env_int("LLM_TIMEOUT", 600),
    "llm_enable_thinking": _env_bool("LLM_ENABLE_THINKING", False),
    "default_max_tokens": _env_int("DEFAULT_MAX_TOKENS", 2048),
    "offload_on_limit": _env_bool("OFFLOAD_ON_LIMIT", True),
    "temperature": float(_env("TEMPERATURE", "0.6")),
    # ---- 路由策略：小模型只负责「工具执行 + 日常事务」，分析类任务不交给它 ----
    # 客户端可用 body.pyrodash_task 或 header x-pyrodash-task 显式声明
    # （tool / routine / analysis），这是唯一权威的信号。没声明时按下面的规则判定。
    "unknown_task": _env("UNKNOWN_TASK", "routine"),         # 判不出时默认按事务处理
    "analysis_min_chars": _env_int("ANALYSIS_MIN_CHARS", 400),  # 长请求一律当分析
    # ---- 大模型腿的预算 ----
    # 旧行为（shared_budget=True）：大模型只能用 total_budget - 小模型已用额度，
    #   小模型花掉 512 后剩下的常常不够让开 thinking 的模型写完，被 length 截断。
    # 默认（False）：给大模型**独立**的完整预算——小模型的 token 是本机算的、
    #   不花钱，没有理由去占客户端付钱的远端预算。
    "shared_budget": _env_bool("SHARED_BUDGET", False),
    "llm_max_tokens": _env_int("LLM_MAX_TOKENS", 0),          # 0 = 用客户端给的 total_budget
}


# ---------------------------------------------------------------------- 统计


class Stats:
    """记录 offload 比例、token 消耗与延迟——用户要求可观测。"""

    def __init__(self, recent_max: int = 200) -> None:
        self._lock = threading.Lock()
        self._recent = deque(maxlen=recent_max)
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.total = 0
            self.route_small = 0
            self.route_handoff = 0
            self.reason_tag = 0
            self.reason_limit = 0
            self.reason_budget_exhausted = 0
            self.reason_analysis = 0
            self.reason_tools_fallback = 0
            self.routes = {
                "small": 0,
                "handoff:tag": 0,
                "handoff:limit": 0,
                "handoff:budget-exhausted": 0,
                "handoff:analysis": 0,
                "handoff:tools-fallback": 0,
            }
            # 任务类型分布（tool / routine / analysis）——看路由策略实际生效情况
            self.tasks: dict[str, int] = {"tool": 0, "routine": 0, "analysis": 0}
            self.tool_calls_small = 0
            self.tool_calls_llm = 0
            self.small_tokens = 0
            self.llm_tokens = 0
            self.small_seconds = 0.0
            self.llm_seconds = 0.0
            self.errors = 0
            self._recent.clear()

    def record(self, rec: dict[str, Any]) -> None:
        with self._lock:
            self.total += 1
            route = rec["pyrodash"]["route"]
            self.routes[route] = self.routes.get(route, 0) + 1
            if route == "small":
                self.route_small += 1
            else:
                self.route_handoff += 1
                if route == "handoff:tag":
                    self.reason_tag += 1
                elif route == "handoff:limit":
                    self.reason_limit += 1
                elif route == "handoff:budget-exhausted":
                    self.reason_budget_exhausted += 1
                elif route == "handoff:analysis":
                    self.reason_analysis += 1
                elif route == "handoff:tools-fallback":
                    self.reason_tools_fallback += 1
            task = rec["pyrodash"].get("task") or "?"
            self.tasks[task] = self.tasks.get(task, 0) + 1
            _src = rec["pyrodash"].get("tool_calls_from")
            if _src == "small":
                self.tool_calls_small += 1
            elif _src == "llm":
                self.tool_calls_llm += 1
            self.small_tokens += rec["pyrodash"].get("small_tokens", 0) or 0
            self.llm_tokens += rec["pyrodash"].get("llm_tokens", 0) or 0
            self.small_seconds += rec["pyrodash"].get("small_elapsed_s", 0.0) or 0.0
            self.llm_seconds += rec["pyrodash"].get("llm_elapsed_s", 0.0) or 0.0
            if rec["pyrodash"].get("error"):
                self.errors += 1
            # 只留紧凑摘要，不存完整响应体 —— 否则 20 条 recent 能让 /v1/stats
            # 涨到几百 KB，而它的用途只是「看占比」。全文请看服务端日志。
            self._recent.append(
                {
                    "route": route or "-",
                    "task": rec["pyrodash"].get("task"),
                    "llm_model": rec["pyrodash"].get("llm_model"),
                    "small_tokens": rec["pyrodash"].get("small_tokens", 0),
                    "llm_tokens": rec["pyrodash"].get("llm_tokens", 0),
                    "small_elapsed_s": rec["pyrodash"].get("small_elapsed_s", 0.0),
                    "llm_elapsed_s": rec["pyrodash"].get("llm_elapsed_s", 0.0),
                    "total_elapsed_s": rec["pyrodash"].get("total_elapsed_s", 0.0),
                    "error": rec["pyrodash"].get("error"),
                }
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            total = self.total
            handoff = self.route_handoff
            n = max(total, 1)
            return {
                "total_requests": total,
                "routes": dict(self.routes),
                "tasks": dict(self.tasks),
                "offload": {
                    "count": handoff,
                    "rate": round(handoff / n, 4),
                    "rate_pct": round(100 * handoff / n, 1),  # 数值型百分比，便于程序化消费
                    "rate_pct_str": f"{100 * handoff / n:.1f}%",  # 给人看的
                    "by_reason": {
                        "tag": self.reason_tag,
                        "limit": self.reason_limit,
                        "budget-exhausted": self.reason_budget_exhausted,
                        "analysis": self.reason_analysis,
                        "tools-fallback": self.reason_tools_fallback,
                    },
                },
                "tool_calls": {
                    "total": self.tool_calls_small + self.tool_calls_llm,
                    "by_small": self.tool_calls_small,  # 小模型自己完成的工具调用
                    "by_llm": self.tool_calls_llm,      # 小模型没做成、回退给大模型的
                },
                "tokens": {
                    "small": self.small_tokens,
                    "llm": self.llm_tokens,
                    "total": self.small_tokens + self.llm_tokens,
                    "llm_share": round(
                        self.llm_tokens / max(self.small_tokens + self.llm_tokens, 1), 4
                    ),
                    "llm_share_pct": round(
                        100 * self.llm_tokens / max(self.small_tokens + self.llm_tokens, 1), 1
                    ),
                },
                "latency_s": {
                    "small_total": round(self.small_seconds, 2),
                    "llm_total": round(self.llm_seconds, 2),
                    "small_avg": round(self.small_seconds / n, 3),
                    "llm_avg": round(self.llm_seconds / max(handoff, 1), 3),
                },
                "errors": self.errors,
                "recent": list(self._recent)[-20:],
            }


STATS = Stats()

# --------------------------------------------------------------- 小模型调用


_SMALL_TEMPLATE_KWARGS: dict[str, Any] = {}
if (raw_kwargs := _env("SMALL_TEMPLATE_KWARGS", "").strip()):
    try:
        _SMALL_TEMPLATE_KWARGS = json.loads(raw_kwargs)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"✗ SMALL_TEMPLATE_KWARGS 不是合法 JSON: {exc}") from exc


def apply_template(messages: list[dict], *, timeout: float = 60.0) -> str:
    """用 llama-server 的 /apply-template 渲染 chat 模板，拿到裸 prompt 字符串。

    ``SMALL_TEMPLATE_KWARGS``（JSON，如 ``{"enable_thinking": false}``）会作为
    ``chat_template_kwargs`` 透传给模板。MiniCPM5-2B / Qwen3 这类「混合思考」模型的
    模板默认开思考：实测单题几千 token 的 reasoning 会把 ``SMALL_MAX_TOKENS`` 一口气
    烧光、``stop_type`` 恒为 ``limit``，于是每道题都被判成「小模型搞不定」交给云端——
    relay 退化成纯代理、本地零节省。Qwen3-4B-Instruct-2507 的模板没有
    ``enable_thinking``，传了无副作用。
    """
    payload: dict[str, Any] = {"messages": messages}
    if _SMALL_TEMPLATE_KWARGS:
        payload["chat_template_kwargs"] = dict(_SMALL_TEMPLATE_KWARGS)
    resp = requests.post(
        f"{CONFIG['small_base_url']}/apply-template",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return str(resp.json()["prompt"])


def call_small(
    prompt: str,
    *,
    n_predict: int,
    temperature: float,
) -> dict[str, Any]:
    """调 llama-server 的原生 /completion，把交接标记设为 stop 词。

    为什么不走 /v1/chat/completions：原生端点的响应里有 ``stop_type`` /
    ``stopping_word``，能**无歧义**区分「撞到交接标记」和「自然答完」——
    OpenAI 兼容端点只给 ``finish_reason``，两种情况都是 "stop"。
    """
    body = {
        "prompt": prompt,
        "stop": [OFFLOAD_TAG],
        "n_predict": int(n_predict),
        "temperature": temperature,
        "cache_prompt": True,
        "stream": False,
    }
    t0 = time.time()
    resp = requests.post(
        f"{CONFIG['small_base_url']}/completion",
        json=body,
        timeout=CONFIG["small_timeout"],
    )
    elapsed = time.time() - t0
    resp.raise_for_status()
    data = resp.json()

    timings = data.get("timings") or {}
    return {
        "content": str(data.get("content") or ""),
        "stop_type": str(data.get("stop_type") or ""),
        "stopping_word": str(data.get("stopping_word") or ""),
        "tokens_predicted": int(timings.get("predicted_n") or 0),
        "elapsed_s": round(elapsed, 3),
    }


def call_small_chat(
    messages: list[dict],
    *,
    tools: list[dict],
    tool_choice: Any = None,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """让小模型自己执行工具调用——走 llama-server 的 /v1/chat/completions。

    为什么这次换端点：``--jinja`` 下的 chat 端点会按模型自带的工具调用模板
    渲染 prompt，并把模型吐出的 tool_call 文本解析成结构化 ``tool_calls``。
    自己在 /completion 上拼工具模板既容易错、还得跟着模型版本维护。

    返回 dict；``tool_calls`` 为空表示小模型没给出可用的工具调用，调用方
    （chat_completion）据此回退给大模型——**不能静默失败**。
    """
    payload: dict[str, Any] = {
        "model": CONFIG["small_model"] or "local",
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": temperature,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

    t0 = time.time()
    resp = requests.post(
        f"{CONFIG['small_base_url']}/v1/chat/completions",
        json=payload,
        timeout=CONFIG["small_timeout"],
    )
    elapsed = time.time() - t0
    if resp.status_code >= 400:
        return {
            "content": "",
            "think": "",
            "tool_calls": [],
            "finish_reason": "",
            "tokens": 0,
            "elapsed_s": round(elapsed, 3),
            "error": f"HTTP {resp.status_code} {resp.text[:200]}",
        }
    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = data.get("usage") or {}
    return {
        "content": str(msg.get("content") or ""),
        "think": str(msg.get("reasoning_content") or ""),
        "tool_calls": list(msg.get("tool_calls") or []),
        "finish_reason": str(choice.get("finish_reason") or ""),
        "tokens": int(usage.get("completion_tokens") or 0),
        "elapsed_s": round(elapsed, 3),
        "error": "",
    }


# ------------------------------------------------------------- 远端接力调用


def _call_llm_with_tools(
    messages: list[dict],
    *,
    max_tokens: int,
    tools: list[dict],
    tool_choice: Any = None,
) -> tuple[str, str, list[dict], dict[str, Any] | None, str]:
    """带工具透传的远端调用，返回 (content, think, tool_calls, usage)。

    PyroDash 原有的 ``_call_dashscope_chat`` 不带 tools，所以这里单独实现；
    不带 tools 的普通接力仍然复用原函数（见 handoff）。
    """
    payload: dict[str, Any] = {
        "model": CONFIG["llm_model"],
        "messages": messages,
        "max_tokens": int(max_tokens),
        "tools": tools,
    }
    if tool_choice:
        payload["tool_choice"] = tool_choice
    if not CONFIG["llm_enable_thinking"]:
        # 同 _call_dashscope_chat：内网网关只认 thinking={"type":"disabled"}，
        # 不传就默认开思考，2048 预算下会被 length 截断。
        payload["thinking"] = {"type": "disabled"}

    resp = requests.post(
        f"{CONFIG['llm_base_url']}/chat/completions",
        headers={
            "Authorization": f"Bearer {CONFIG['llm_api_key']}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=float(CONFIG["llm_timeout"]),
    )
    if resp.status_code >= 400:
        return f"[Error: HTTP {resp.status_code} {resp.text[:200]}]", "", [], None, ""
    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = str(msg.get("content") or "")
    think = str(msg.get("reasoning_content") or "")
    return (
        content,
        think,
        list(msg.get("tool_calls") or []),
        data.get("usage"),
        str(choice.get("finish_reason") or ""),
    )


def handoff(
    original_messages: list[dict],
    partial: str,
    *,
    max_tokens: int,
    tools: list[dict] | None = None,
    tool_choice: Any = None,
) -> tuple[str, str, dict[str, Any] | None, float, str, list[dict], str]:
    """把小模型的半截推理接力给远端大模型。

    返回 (答案, 思考, usage, 耗时, 错误, tool_calls)。

    直接复用 PyroDash 的 ``_build_offload_messages``：它会把 partial 包成
    ``<part_think>...</part_think>``，并配上「从断点接着推、不要重复」的
    system prompt。
    """
    if max_tokens <= 0:
        return "", "", None, 0.0, "", [], ""


    messages = _build_offload_messages(original_messages, partial)
    t0 = time.time()
    tool_calls: list[dict] = []
    llm_finish = ""
    if tools:
        content, think, tool_calls, usage, llm_finish = _call_llm_with_tools(
            messages, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice
        )
    else:
        content, think, usage = _call_dashscope_chat(
            messages,
            api_key=CONFIG["llm_api_key"],
            base_url=CONFIG["llm_base_url"],
            model=CONFIG["llm_model"],
            max_tokens=max_tokens,
            timeout=float(CONFIG["llm_timeout"]),
            enable_thinking=bool(CONFIG["llm_enable_thinking"]),
        )
        # _call_dashscope_chat 是 PyroDash 原样复用的函数，不返回 finish_reason。
        # 截断时必然把预算用满，所以用「completion 顶到上限」作为截断信号。
        _ct = int((usage or {}).get("completion_tokens") or 0)
        llm_finish = "length" if max_tokens > 0 and _ct >= max_tokens else "stop"
    elapsed = time.time() - t0
    error = ""
    if content.startswith("[Error:"):
        error = content
        content = ""
    return content, think, usage, round(elapsed, 3), error, tool_calls, llm_finish


# ------------------------------------------------------------------ 主流程


# ------------------------------------------------------------ 任务类型判定
# 路由策略：小模型只做**工具执行**和**日常事务性任务**；
# **分析类任务不交给小模型**，直接转给大模型。
#
# 这里不挂第二个分类模型（为省一次推理不值得），而用三条信号，优先级从高到低：
#   1) 客户端显式声明：body 的 `pyrodash_task`（HTTP 层会把 header
#      `x-pyrodash-task` 合并进来）—— 只有调用方知道自己要什么，这是唯一权威信号；
#   2) 请求带 tools → 工具执行，那是小模型的活；
#   3) 关键词 / 长度启发式 → analysis 或 routine。
#
# 启发式一定会判错，所以它的作用被刻意限制为「决定要不要让小模型先试一下」：
# 判成 analysis 就走直连交接（小模型完全不参与，不浪费本机算力）；判成 routine
# 则小模型先答，答不完仍由预算耗尽兜底。判错的最坏结果是多/少一次本机尝试，
# 不会产出错误答案。

ANALYSIS_HINTS = (
    "分析", "推理", "推导", "证明", "论证", "为什么", "原理", "权衡",
    "对比", "评估", "设计一个", "设计方案", "架构", "优化", "重构",
    "调试", "定位问题", "根因", "复杂度",
    "analyze", "analyse", "reason", "prove", "derive", "explain why",
    "why does", "why is", "trade-off", "tradeoff", "compare", "evaluate",
    "design", "architect", "optimize", "refactor", "debug", "root cause",
)
ROUTINE_HINTS = (
    "翻译", "格式化", "格式转换", "转换成", "提取", "重命名", "总结",
    "摘要", "改写", "列出", "列举", "统计", "计数", "排序", "去重",
    "补全", "加注释", "重写为",
    "translate", "format", "convert", "extract", "rename", "summarize",
    "summarise", "rewrite", "list", "count", "sort", "dedupe", "comment",
)


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return " ".join(
                str(p.get("text") or "")
                for p in c
                if isinstance(p, dict)
            )
    return ""


def classify_task(
    body: dict[str, Any], messages: list[dict], tools: list[dict] | None
) -> tuple[str, str]:
    """判定任务类型 → (task, source)；task ∈ {tool, routine, analysis}。"""
    hint = str(body.get("pyrodash_task") or "").strip().lower()
    if hint in ("tool", "routine", "analysis"):
        return hint, "declared"
    if tools:
        return "tool", "tools"
    text = _last_user_text(messages)
    if len(text) >= int(CONFIG["analysis_min_chars"]):
        return "analysis", "long"
    low = text.lower()
    for k in ANALYSIS_HINTS:
        if k in low:
            return "analysis", "keyword"
    for k in ROUTINE_HINTS:
        if k in low:
            return "routine", "keyword"
    return str(CONFIG["unknown_task"]), "default"


def _finish_reason(
    tool_calls: list[dict],
    route: str,
    small_stop_type: str,
    small_finish: str,
    llm_finish: str,
) -> str:
    """OpenAI 兼容的 finish_reason。

    这里以前恒返回 "stop"（tool_calls 除外），造成**截断被静默掩盖**：
    小模型腿用满预算被切断、或远端大模型腿被 length 截断，调用方都会当作
    「正常答完」。agent 客户端（pi / mini-swe-agent）信任这个字段就会提前
    结束回合。现在把各种截断情形如实上报为 "length"。
    """
    if tool_calls:
        return "tool_calls"
    if route == "handoff:budget-exhausted":
        return "length"        # 只回填了小模型的半截，必然不完整
    if route == "small" and (small_stop_type == "limit" or small_finish == "length"):
        return "length"        # 本机答到上限被切断
    if llm_finish == "length":
        return "length"        # 远端大模型被 length 截断
    return "stop"


def chat_completion(body: dict[str, Any]) -> dict[str, Any]:
    """一次完整的「小模型先答 → 按需接力」流程。"""
    messages = body.get("messages") or []
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages 不能为空")

    total_budget = int(body.get("max_tokens") or CONFIG["default_max_tokens"])
    # 注意：不能用 `body.get("temperature") or CONFIG[...]`，因为 0.0 是 falsy——
    # 传 temperature=0（评测/确定性场景最常用的值）会被静默换成默认的 0.6。
    temperature = (
        float(body["temperature"])
        if body.get("temperature") is not None
        else float(CONFIG["temperature"])
    )
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    t_start = time.time()

    tools = body.get("tools") or None
    tool_choice = body.get("tool_choice")
    task, task_source = classify_task(body, messages, tools)

    small_text = ""
    small_tokens = 0
    small_elapsed = 0.0
    small_stop_type = ""
    small_finish = ""          # 小模型腿的 finish_reason（工具腿来自 chat 端点）
    small_tool_calls: list[dict] = []
    route = "small"
    offload_reason = ""
    error = ""                # 小模型腿的错误（工具腿可能产生）

    if task == "tool":
        # 工具执行是小模型的活：让它自己产出 tool_calls。
        # 给不出可用调用就回退给大模型——**不静默失败**。
        sc = call_small_chat(
            messages,
            tools=tools or [],
            tool_choice=tool_choice,
            max_tokens=min(CONFIG["small_max_tokens"], total_budget),
            temperature=temperature,
        )
        small_tokens = sc["tokens"]
        small_elapsed = sc["elapsed_s"]
        small_finish = sc["finish_reason"]
        small_text = sc["content"].strip()
        error = sc["error"]
        if sc["tool_calls"]:
            small_tool_calls = sc["tool_calls"]
            route = "small"
        else:
            route, offload_reason = "handoff:tools-fallback", "tools-fallback"
    elif task == "analysis":
        # 分析类任务不交给小模型：直接转大模型，本机一 token 都不烧，
        # 也就不会出现「小模型先写 512 token 半成品再被丢掉」的浪费。
        route, offload_reason = "handoff:analysis", "analysis"
    else:
        # 日常事务性任务：小模型先答，预算内答完=本机搞定，否则交接。
        # 1) 注入 offload 协议
        small_messages = inject_protocol(messages)
        # 2) 渲染 chat 模板  + 3) 小模型先答（stop 在交接标记上）
        prompt = apply_template(small_messages)
        # 小模型用**自己的**小预算：答得完=本机搞定，答不完=交接。
        # 若直接用 total_budget，它会把预算全吃光才会触发 limit。
        small_budget = min(CONFIG["small_max_tokens"], total_budget)
        small = call_small(prompt, n_predict=small_budget, temperature=temperature)
        small_tokens = small["tokens_predicted"]
        small_elapsed = small["elapsed_s"]
        small_stop_type = small["stop_type"]
        small_text = small["content"].strip()

        if small_stop_type == "word" and small["stopping_word"] == OFFLOAD_TAG:
            route, offload_reason = "handoff:tag", "tag"
        elif small_stop_type == "limit" and CONFIG["offload_on_limit"]:
            route, offload_reason = "handoff:limit", "limit"

    llm_text = ""
    llm_think = ""
    llm_tool_calls: list[dict] = []
    llm_usage: dict[str, Any] | None = None
    llm_elapsed = 0.0
    llm_finish = ""            # 远端大模型腿是否被截断（见 _finish_reason）
    remaining = total_budget   # 大模型腿额额度，见下面分支

    if route == "small":
        content = small_text
    else:
        # 小模型可能无视 stop 词仍把标记吐出来，_offload_prefix 再剥一层兜底
        partial = _offload_prefix(small_text)
        # 大模型腿的预算：
        #   旧行为（SHARED_BUDGET=1）沿用 PyroDash 的 _llm_max_tokens，
        #   「大模型只能用总预算里剩下的额度」——小模型花掉 512 后只剩 1536，
        #   开 thinking 的模型写不完推理+代码就被 length 截断（实测
        #   HumanEval/130：1536 失败 → 2157 通过）。小模型的 token 是本机
        #   算的、不花钱，没有理由去占客户端付钱的远端预算。
        #   默认改为独立预算，用客户端给的完整 total_budget。
        if CONFIG["shared_budget"]:
            remaining = _llm_max_tokens(total_budget, [[None] * small_tokens], 0)
        else:
            remaining = int(CONFIG["llm_max_tokens"]) or total_budget
        if remaining <= 0:
            route = "handoff:budget-exhausted"
            content = partial
        else:
            (
                llm_text,
                llm_think,
                llm_usage,
                llm_elapsed,
                error,
                llm_tool_calls,
                llm_finish,
            ) = handoff(
                messages,
                partial,
                max_tokens=remaining,
                tools=tools,
                tool_choice=tool_choice,
            )
            content = llm_text or ("" if llm_tool_calls else partial)

    llm_tokens = int((llm_usage or {}).get("completion_tokens") or 0)
    prompt_tokens = int((llm_usage or {}).get("prompt_tokens") or 0)
    # 工具调用的来源可能是小模型（task=tool 且它给了调用）或大模型
    tool_calls_out = llm_tool_calls or small_tool_calls

    usage = {
        "prompt_tokens": prompt_tokens or small_tokens,
        "completion_tokens": small_tokens + llm_tokens,
        "total_tokens": (prompt_tokens or 0) + small_tokens + llm_tokens,
    }

    response = {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model") or "pyrodash-local",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    **({"tool_calls": tool_calls_out} if tool_calls_out else {}),
                },
                "finish_reason": _finish_reason(
                    tool_calls_out, route, small_stop_type, small_finish, llm_finish
                ),
            },
        ],
        "usage": usage,
        # 非标准扩展字段：给调用方/日志看的可观测信息
        "pyrodash": {
            "route": route,
            "offload_reason": offload_reason,
            "offloaded": route != "small",
            "task": task,
            "task_source": task_source,
            "tools_passthrough": bool(tools) and route != "small",
            "tool_calls": len(tool_calls_out),
            "tool_calls_from": (
                "small" if small_tool_calls else ("llm" if llm_tool_calls else None)
            ),
            "small_model": CONFIG["small_model"] or "local",
            "small_stop_type": small_stop_type or None,
            "small_finish_reason": small_finish or None,
            "small_tokens": small_tokens,
            "llm_model": CONFIG["llm_model"] if route != "small" else None,
            "llm_tokens": llm_tokens,
            "total_budget": total_budget,
            "llm_budget": None if route == "small" else remaining,
            "budget_used": small_tokens + llm_tokens,
            "small_elapsed_s": small_elapsed,
            "llm_elapsed_s": llm_elapsed,
            "total_elapsed_s": round(time.time() - t_start, 3),
            "small_reasoning": small_text if route != "small" else None,
            "llm_reasoning": llm_think or None,
            "error": error or None,
        },
    }
    STATS.record(response)
    _log_request(response)
    return response


def _log_request(rec: dict[str, Any]) -> None:
    p = rec["pyrodash"]
    mark = "→LLM" if p["offloaded"] else " 本机"
    reason = f" [{p['offload_reason']}]" if p["offload_reason"] else ""
    err = f"  ⚠ {p['error'][:60]}" if p["error"] else ""
    print(
        f"  {mark}{reason:<8} "
        f"small={p['small_tokens']:>4}tok {p['small_elapsed_s']:>6.2f}s"
        + (
            f"  llm={p['llm_tokens']:>4}tok {p['llm_elapsed_s']:>6.2f}s"
            if p["offloaded"]
            else ""
        )
        + f"  合计 {p['total_elapsed_s']:>6.2f}s{err}",
        flush=True,
    )


# ----------------------------------------------------------------- HTTP 层


class Handler(BaseHTTPRequestHandler):
    server_version = "pyrodash-relay/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # 静音默认访问日志
        return

    # -------------------------------------------------- 工具方法
    def _send(self, code: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _sse(self, rec: dict[str, Any]) -> None:
        """把最终结果按 SSE 分块吐出（兼容请求 stream=true 的客户端）。"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        base = {
            "id": rec["id"],
            "object": "chat.completion.chunk",
            "created": rec["created"],
            "model": rec["model"],
        }

        def emit(delta: dict[str, Any], finish: str | None = None) -> None:
            chunk = {
                **base,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()

        emit({"role": "assistant", "content": ""})
        message = rec["choices"][0]["message"]
        text = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []
        step = 96  # 按固定宽度切块，纯为兼容流式客户端
        for i in range(0, len(text), step):
            emit({"content": text[i : i + step]})
        finish = rec["choices"][0].get("finish_reason") or "stop"
        if tool_calls:
            # OpenAI 流式协议里 tool_calls 走 delta.tool_calls，一次给完
            # （TCP 已经帮我们分帧，没必要再切）。**漏了这一步会让 pi 的工具
            # 调用在流式下静默失效** —— pi 默认就是流式 + 工具。
            for tc in tool_calls:
                emit({"tool_calls": [tc]})
        # finish_reason 必须透传真实值（"length" = 被截断）。以前这里硬编码
        # "stop"，相当于把刚在 _finish_reason 里修好的截断上报又在流式路径丢掉了。
        emit({}, finish)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    # -------------------------------------------------- GET
    def do_GET(self) -> None:  # noqa: N802
        path = self.path.rstrip("/")
        if path in ("/health", "/v1/health"):
            self._send(200, {"status": "ok", "config": _public_config()})
        elif path in ("/v1/models", "/models"):
            self._send(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "pyrodash-local",
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "pyrodash",
                        }
                    ],
                },
            )
        elif path in ("/v1/stats", "/stats"):
            self._send(200, STATS.snapshot())
        else:
            self._send(404, {"error": {"message": f"未知路径 {self.path}"}})

    # -------------------------------------------------- POST
    def do_POST(self) -> None:  # noqa: N802
        path = self.path.rstrip("/")
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw.decode("utf-8") or "{}")
        except Exception as exc:
            self._send(400, {"error": {"message": f"请求体解析失败: {exc}"}})
            return

        # 把 header 里显式声明的任务类型合并进 body（body 里的优先）。
        # 这是路由策略唯一权威的信号，见 classify_task：调用方说是什么任务，
        # 就按什么任务路由，不靠猜。
        hdr_task = (self.headers.get("x-pyrodash-task") or "").strip()
        if hdr_task and not body.get("pyrodash_task"):
            body["pyrodash_task"] = hdr_task

        if path in ("/v1/stats/reset", "/stats/reset"):
            STATS.reset()
            self._send(200, {"ok": True, "message": "统计已重置"})
            return

        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._send(404, {"error": {"message": f"未知路径 {self.path}"}})
            return

        try:
            rec = chat_completion(body)
        except Exception as exc:
            with STATS._lock:
                STATS.errors += 1
            self._send(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
            return

        if body.get("stream"):
            try:
                self._sse(rec)
            except Exception:
                pass
        else:
            self._send(200, rec)


def _public_config() -> dict[str, Any]:
    c = dict(CONFIG)
    c["llm_api_key"] = "***" if c["llm_api_key"] else "(未配置)"
    return c


# -------------------------------------------------------------------- 入口


def _detect_small_model() -> None:
    """向 llama-server 问一次模型名，便于日志显示。"""
    if CONFIG["small_model"]:
        return
    try:
        props = requests.get(f"{CONFIG['small_base_url']}/props", timeout=10).json()
        name = props.get("model_path") or props.get("model_alias") or ""
        CONFIG["small_model"] = Path(str(name)).name or "local-small"
    except Exception:
        CONFIG["small_model"] = "local-small(未探测到)"


def main() -> int:
    ap = argparse.ArgumentParser(description="PyroDash 式本机中转服务")
    ap.add_argument("--host", default=CONFIG["relay_host"])
    ap.add_argument("--port", type=int, default=CONFIG["relay_port"])
    ap.add_argument("--small-base-url", default=CONFIG["small_base_url"])
    ap.add_argument("--llm-model", default=CONFIG["llm_model"])
    args = ap.parse_args()

    CONFIG["relay_host"] = args.host
    CONFIG["relay_port"] = args.port
    CONFIG["small_base_url"] = args.small_base_url.rstrip("/")
    CONFIG["llm_model"] = args.llm_model

    _detect_small_model()

    print("=" * 68)
    print(" PyroDash 式本机中转服务")
    print("=" * 68)
    print(f"   监听        : http://{CONFIG['relay_host']}:{CONFIG['relay_port']}")
    print(f"   本机小模型  : {CONFIG['small_base_url']}  ({CONFIG['small_model']})")
    print(f"   交接标记    : {OFFLOAD_TAG}")
    print(f"   远端大模型  : {CONFIG['llm_base_url']}  ({CONFIG['llm_model']})")
    print(f"   远端密钥    : {'已配置' if CONFIG['llm_api_key'] else '✗ 缺失（接力会失败）'}")
    print(f"   共享预算    : {CONFIG['default_max_tokens']} tokens（可用 max_tokens 覆盖）")
    print(f"   截断也接力  : {CONFIG['offload_on_limit']}")
    print("=" * 68)
    if not CONFIG["llm_api_key"]:
        print("  ⚠ 未找到远端密钥。请设置 LLM_API_KEY 或写入 setup/.llm_key")
    print("  端点: POST /v1/chat/completions | GET /v1/models | GET /v1/stats")
    print("=" * 68, flush=True)

    httpd = ThreadingHTTPServer((CONFIG["relay_host"], CONFIG["relay_port"]), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
