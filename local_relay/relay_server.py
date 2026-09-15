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
    "llm_model": _env("LLM_MODEL", "deepseek-v4-pro"),
    "llm_timeout": _env_int("LLM_TIMEOUT", 600),
    "llm_enable_thinking": _env_bool("LLM_ENABLE_THINKING", True),
    "default_max_tokens": _env_int("DEFAULT_MAX_TOKENS", 2048),
    "offload_on_limit": _env_bool("OFFLOAD_ON_LIMIT", True),
    "temperature": float(_env("TEMPERATURE", "0.6")),
}


# ---------------------------------------------------------------------- 统计


class Stats:
    """记录 offload 比例、token 消耗与延迟——用户要求可观测。"""

    def __init__(self, recent_max: int = 200) -> None:
        self._lock = threading.Lock()
        self._recent = deque(maxlen=recent_max)
        self.reset()

    def reset(self) -> None:
        with getattr(self, "_lock", threading.Lock()):
            self.total = 0
            self.route_small = 0
            self.route_handoff = 0
            self.reason_tag = 0
            self.reason_limit = 0
            self.reason_budget_exhausted = 0
            self.reason_tools = 0
            self.routes = {
                "small": 0,
                "handoff:tag": 0,
                "handoff:limit": 0,
                "handoff:budget-exhausted": 0,
                "handoff:tools": 0,
            }
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
                elif route == "handoff:tools":
                    self.reason_tools += 1
            self.small_tokens += rec["pyrodash"].get("small_tokens", 0) or 0
            self.llm_tokens += rec["pyrodash"].get("llm_tokens", 0) or 0
            self.small_seconds += rec["pyrodash"].get("small_elapsed_s", 0.0) or 0.0
            self.llm_seconds += rec["pyrodash"].get("llm_elapsed_s", 0.0) or 0.0
            if rec["pyrodash"].get("error"):
                self.errors += 1
            self._recent.append(rec)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            total = self.total
            handoff = self.route_handoff
            n = max(total, 1)
            return {
                "total_requests": total,
                "routes": dict(self.routes),
                "offload": {
                    "count": handoff,
                    "rate": round(handoff / n, 4),
                    "rate_pct": round(100 * handoff / n, 1),  # 数值型百分比，便于程序化消费
                    "rate_pct_str": f"{100 * handoff / n:.1f}%",  # 给人看的
                    "by_reason": {
                        "tag": self.reason_tag,
                        "limit": self.reason_limit,
                        "budget-exhausted": self.reason_budget_exhausted,
                        "tools": self.reason_tools,
                    },
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


def apply_template(messages: list[dict], *, timeout: float = 60.0) -> str:
    """用 llama-server 的 /apply-template 渲染 chat 模板，拿到裸 prompt 字符串。"""
    resp = requests.post(
        f"{CONFIG['small_base_url']}/apply-template",
        json={"messages": messages},
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


# ------------------------------------------------------------- 远端接力调用


def _call_llm_with_tools(
    messages: list[dict],
    *,
    max_tokens: int,
    tools: list[dict],
    tool_choice: Any = None,
) -> tuple[str, str, list[dict], dict[str, Any] | None]:
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
        return f"[Error: HTTP {resp.status_code} {resp.text[:200]}]", "", [], None
    data = resp.json()
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    content = str(msg.get("content") or "")
    think = str(msg.get("reasoning_content") or "")
    return content, think, list(msg.get("tool_calls") or []), data.get("usage")


def handoff(
    original_messages: list[dict],
    partial: str,
    *,
    max_tokens: int,
    tools: list[dict] | None = None,
    tool_choice: Any = None,
) -> tuple[str, str, dict[str, Any] | None, float, str, list[dict]]:
    """把小模型的半截推理接力给远端大模型。

    返回 (答案, 思考, usage, 耗时, 错误, tool_calls)。

    直接复用 PyroDash 的 ``_build_offload_messages``：它会把 partial 包成
    ``<part_think>...</part_think>``，并配上「从断点接着推、不要重复」的
    system prompt。
    """
    if max_tokens <= 0:
        return "", "", None, 0.0, "", []


    messages = _build_offload_messages(original_messages, partial)
    t0 = time.time()
    tool_calls: list[dict] = []
    if tools:
        content, think, tool_calls, usage = _call_llm_with_tools(
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
    elapsed = time.time() - t0
    error = ""
    if content.startswith("[Error:"):
        error = content
        content = ""
    return content, think, usage, round(elapsed, 3), error, tool_calls


# ------------------------------------------------------------------ 主流程


def chat_completion(body: dict[str, Any]) -> dict[str, Any]:
    """一次完整的「小模型先答 → 按需接力」流程。"""
    messages = body.get("messages") or []
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages 不能为空")

    total_budget = int(body.get("max_tokens") or CONFIG["default_max_tokens"])
    temperature = float(body.get("temperature") or CONFIG["temperature"])
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    t_start = time.time()

    tools = body.get("tools") or None
    tool_choice = body.get("tool_choice")

    small_text = ""
    small_tokens = 0
    small_elapsed = 0.0
    small_stop_type = ""
    route = "small"
    offload_reason = ""

    if tools:
        # 带 tools 的请求（pi / agent 客户端）跳过本机小模型：
        # 4B 模型做不了可靠的工具编排，硬让它试只会得到跑不通的 agent 行为。
        # 直接交给大模型，工具调用的语义与质量都由它保证。
        route, offload_reason = "handoff:tools", "tools"
    else:
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
    error = ""

    if route == "small":
        content = small_text
    else:
        # 小模型可能无视 stop 词仍把标记吐出来，_offload_prefix 再剥一层兼底
        partial = _offload_prefix(small_text)
        # 共享预算：大模型只能用剩下的额度（规则来自 PyroDash 的 _llm_max_tokens）
        remaining = _llm_max_tokens(total_budget, [[None] * small_tokens], 0)
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
                        **({"tool_calls": llm_tool_calls} if llm_tool_calls else {}),
                    },
                    "finish_reason": "tool_calls" if llm_tool_calls else "stop",
                },
        ],
        "usage": usage,
        # 非标准扩展字段：给调用方/日志看的可观测信息
        "pyrodash": {
            "route": route,
            "offload_reason": offload_reason,
            "offloaded": route != "small",
            "tools_passthrough": bool(tools),
            "tool_calls": len(llm_tool_calls),
            "small_model": CONFIG["small_model"] or "local",
            "small_stop_type": small_stop_type or None,
            "small_tokens": small_tokens,
            "llm_model": CONFIG["llm_model"] if route != "small" else None,
            "llm_tokens": llm_tokens,
            "total_budget": total_budget,
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
        if tool_calls:
            # OpenAI 流式协议里 tool_calls 走 delta.tool_calls，一次给完
            # （TCP 已经帮我们分帧，没必要再切）。**漏了这一步会让 pi 的工具
            # 调用在流式下静默失效** —— pi 默认就是流式 + 工具。
            for tc in tool_calls:
                emit({"tool_calls": [tc]})
            emit({}, "tool_calls")
        else:
            emit({}, "stop")
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
