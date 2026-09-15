#!/usr/bin/env bash
# run.sh — 一条命令拉起「本机小模型 + PyroDash 式中转服务」
#
#   bash local_relay/run.sh              # 启动（默认 8080 小模型 / 8010 中转）
#   bash local_relay/run.sh --stop       # 停止
#   bash local_relay/run.sh --model models/llama/其他.gguf --ctx 32768
#
# 启动后，任何 OpenAI 客户端指向 http://127.0.0.1:8010/v1 即可。
# 看 offload 统计： curl -s http://127.0.0.1:8010/v1/stats
#
# 为什么小模型走 llama.cpp 而不是 PyroDash 原来的 transformers/vLLM 路径：
#   1) 通用模型（Qwen3-4B）比 PyroDash-4B 的数学专用模型更实用；
#   2) llama.cpp 原生 /completion 的 stop_type 字段能无歧义判定「是否触发交接」，
#      比 vLLM 的 include_stop_str_in_output 更干净；
#   3) 纯推理零 Python 依赖，Q4 量化 2.5GB 完全放得进 12GB 显存。
set -uo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="$PROJ/setup/llama.cpp"
RUN_DIR="$PROJ/local_relay/.run"
VENV_PY="$PROJ/setup/.venv-win/Scripts/python.exe"
PY="${PYTHON:-$VENV_PY}"

MODEL="$PROJ/models/llama/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
SMALL_PORT=8080
RELAY_PORT=8010
CTX=16384
NGL=99
STOP_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --stop)       STOP_ONLY=1; shift ;;
    --model)      MODEL="$2"; shift 2 ;;
    --ctx)        CTX="$2"; shift 2 ;;
    --ngl)        NGL="$2"; shift 2 ;;
    --port)       SMALL_PORT="$2"; shift 2 ;;
    --relay-port) RELAY_PORT="$2"; shift 2 ;;
    -h|--help)    sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \?/  /'; exit 0 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

klog() { echo "  $*"; }
kill_pidfile() {
  local f="$1" name="$2"
  [ -f "$f" ] || return 0
  local pid; pid=$(cat "$f" 2>/dev/null)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    klog "停止 $name (pid $pid)"
    kill "$pid" 2>/dev/null
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -9 "$pid" 2>/dev/null
  fi
  rm -f "$f"
}

# ------------------------------------------------------------------ 停止
if [ "$STOP_ONLY" = "1" ]; then
  echo "== 停止本机中转 =="
  kill_pidfile "$RUN_DIR/relay.pid" "relay"
  kill_pidfile "$RUN_DIR/llama.pid" "llama-server"
  echo "  完成"
  exit 0
fi

mkdir -p "$RUN_DIR"

# ------------------------------------------------------------------ 前置检查
echo "======================================================================"
echo " PyroDash 式本机中转 —— 启动"
echo "======================================================================"

LLAMA_BIN="$LLAMA_DIR/llama-server.exe"
if [ ! -x "$LLAMA_BIN" ]; then
  echo "  ✗ 未找到 $LLAMA_BIN"
  echo "    请先运行: bash setup/07-install-llama.sh"
  exit 1
fi
[ -f "$MODEL" ] || { echo "  ✗ 模型不存在: $MODEL"; echo "    请先运行: python setup/08-download-gguf.py"; exit 1; }
if [ ! -x "$PY" ]; then
  echo "  ⚠ 未找到 venv python ($PY)，回退到系统 python"
  PY="python"
fi

# 清掉可能残留的旧进程
kill_pidfile "$RUN_DIR/relay.pid" "relay"
kill_pidfile "$RUN_DIR/llama.pid" "llama-server"

# ------------------------------------------------------------- 1. llama-server
echo
klog "启动 llama-server (:${SMALL_PORT})"
klog "  模型: $(basename "$MODEL") ($(du -h "$MODEL" 2>/dev/null | cut -f1))"
klog "  上下文 ${CTX} / GPU 层 ${NGL}"

cd "$LLAMA_DIR" || exit 1
nohup ./llama-server.exe \
  --model "$MODEL" \
  --port "$SMALL_PORT" \
  --ctx-size "$CTX" \
  --n-gpu-layers "$NGL" \
  --jinja \
  --host 127.0.0.1 \
  > "$RUN_DIR/llama.log" 2>&1 &
echo $! > "$RUN_DIR/llama.pid"
klog "  pid $(cat "$RUN_DIR/llama.pid")  日志: local_relay/.run/llama.log"

# 等就绪（最多 180s）
klog "  等待模型加载..."
READY=0
for i in $(seq 1 180); do
  if curl -s -m 2 -o /dev/null "http://127.0.0.1:${SMALL_PORT}/health" 2>/dev/null; then READY=1; break; fi
  if ! kill -0 "$(cat "$RUN_DIR/llama.pid")" 2>/dev/null; then
    echo "  ✗ llama-server 已退出，日志尾部："
    tail -25 "$RUN_DIR/llama.log" | sed 's/^/    /'
    exit 1
  fi
  [ $((i % 15)) -eq 0 ] && klog "  ... 已等 ${i}s"
  sleep 1
done
if [ "$READY" != "1" ]; then
  echo "  ✗ 180s 内未就绪，日志尾部："
  tail -25 "$RUN_DIR/llama.log" | sed 's/^/    /'
  kill_pidfile "$RUN_DIR/llama.pid" "llama-server"
  exit 1
fi
klog "  ✅ 就绪"

# 冒烟：确认 /apply-template 可用（--jinja 生效）
TMPL=$(curl -s -m 20 -X POST "http://127.0.0.1:${SMALL_PORT}/apply-template" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}]}' 2>/dev/null)
if echo "$TMPL" | grep -q '"prompt"'; then
  klog "  ✅ /apply-template 可用（chat 模板正常）"
else
  klog "  ⚠ /apply-template 未返回 prompt，接力将无法渲染对话"
  echo "$TMPL" | head -3 | sed 's/^/    /'
fi

# ------------------------------------------------------------------ 2. relay
echo
klog "启动 relay (:${RELAY_PORT}, OpenAI 兼容)"
cd "$PROJ" || exit 1
nohup "$PY" "$PROJ/local_relay/relay_server.py" --port "$RELAY_PORT" \
  --small-base-url "http://127.0.0.1:${SMALL_PORT}" \
  > "$RUN_DIR/relay.log" 2>&1 &
echo $! > "$RUN_DIR/relay.pid"

for i in $(seq 1 30); do
  if curl -s -m 2 -o /dev/null "http://127.0.0.1:${RELAY_PORT}/health" 2>/dev/null; then break; fi
  if ! kill -0 "$(cat "$RUN_DIR/relay.pid")" 2>/dev/null; then
    echo "  ✗ relay 启动失败，日志："
    tail -25 "$RUN_DIR/relay.log" | sed 's/^/    /'
    kill_pidfile "$RUN_DIR/llama.pid" "llama-server"
    exit 1
  fi
  sleep 1
done

echo
sed 's/^/  /' "$RUN_DIR/relay.log"
echo
echo "======================================================================"
echo " ✅ 全部就绪 —— 把 OpenAI 客户端指向下面这个地址即可"
echo
echo "      Base URL : http://127.0.0.1:${RELAY_PORT}/v1"
echo "      API Key  : 任意非空字符串"
echo "      Model    : pyrodash-local"
echo
echo "      看 offload 统计 : curl -s http://127.0.0.1:${RELAY_PORT}/v1/stats"
echo "      重置统计        : curl -s -X POST http://127.0.0.1:${RELAY_PORT}/v1/stats/reset"
echo "      观察实时日志    : tail -f local_relay/.run/relay.log"
echo "      停止全部        : bash local_relay/run.sh --stop"
echo "======================================================================"
echo
echo "  提示：日志里『→LLM』= 小模型交接给了大模型，『本机』= 小模型自己答完了。"
