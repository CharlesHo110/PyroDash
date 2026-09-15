#!/usr/bin/env bash
# 04-run-math-eval.sh —— 在 WSL Ubuntu 中执行
# 作用：本地 PyroDash-4B（vLLM）+ 内网 DeepSeek 大模型，跑数学评测
# 用法：bash /mnt/d/hecan/PyroDash/setup/04-run-math-eval.sh [数据集...]
#       默认 gsm8k（冒烟）；全量：bash .../04-run-math-eval.sh gsm8k minerva olympiad aime2024 aime2025

set -euo pipefail

PROJ=/mnt/d/hecan/PyroDash
VENV="${VENV:-$HOME/pyrodash-venv}"
MODEL="${MODEL:-$PROJ/models/PyroDash-4B-GRPO-Lambda-0.05}"

# ---- 远端大模型：tkoffice 内网 DeepSeek（取自 aiConfig/claude/enableClaudeChina.py）----
LLM_BASE_URL="${LLM_BASE_URL:-https://ai-api.bj.tkoffice.cn/v1}"
LLM_API_KEY="${LLM_API_KEY:-$(cat "$PROJ/setup/.llm_key" 2>/dev/null)}"
LLM_MODEL="${LLM_MODEL:-deepseek-v4-pro}"          # 省钱可换 deepseek-v4-flash

# ---- 本地 vLLM：RTX 3060 12GB 的参数（40960 上下文在 12GB 上会 OOM，压到 16384）----
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.90}"
PORT="${PORT:-8001}"
OUT_DIR="${OUT_DIR:-$PROJ/results}"

DATASETS=("$@")
[ ${#DATASETS[@]} -eq 0 ] && DATASETS=(gsm8k)

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export LD_LIBRARY_PATH="$VENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# 评测数据集（gsm8k / amc / minerva / olympiad / aime2024 / aime2025 / math）
# 优先用 setup/hf_home 预置离线缓存（实测 6 个数据集完全离线可用，规避本机间歇性断网）
export HF_HOME="${HF_HOME:-$PROJ/setup/hf_home}"
if [ -d "$HF_HOME/datasets" ]; then
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
  echo "[datasets] 使用预置离线缓存: $HF_HOME"
else
  echo "[datasets] 未找到预置缓存，回退联网拉取" >&2
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
fi

if [ ! -d "$MODEL" ]; then
  echo "!! 模型目录不存在: $MODEL" >&2
  echo "   请先执行: bash $PROJ/setup/03-download-model.sh" >&2
  exit 1
fi
mkdir -p "$OUT_DIR"

echo "===== 1) 启动本地 vLLM（小模型）====="
echo "模型: $MODEL"
echo "上下文: $MAX_MODEL_LEN  显存占用比: $GPU_UTIL  端口: $PORT"
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --served-model-name small-model \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code \
  >"$PROJ/vllm_serve.log" 2>&1 &
VLLM_PID=$!
trap 'echo "[vllm] 停止 pid=$VLLM_PID"; kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true' EXIT

echo "[vllm] 启动中...（日志: $PROJ/vllm_serve.log）"
for i in $(seq 1 150); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "[vllm] 就绪（等待 $((i*2))s）"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "!! vLLM 进程已退出，最后 40 行日志：" >&2
    tail -40 "$PROJ/vllm_serve.log" >&2
    exit 1
  fi
  sleep 2
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "!! vLLM 启动超时" >&2; tail -40 "$PROJ/vllm_serve.log" >&2; exit 1; }

echo
echo "===== 2) 跑评测（数据集: ${DATASETS[*]}）====="
python "$PROJ/evaluation/evaluation_math/math_eval.py" \
  --model-path "$MODEL" \
  --small-base-url "http://127.0.0.1:$PORT/v1" \
  --small-model small-model \
  --output-dir "$OUT_DIR" \
  --datasets "${DATASETS[@]}" \
  --llm-base-url "$LLM_BASE_URL" \
  --llm-api-key "$LLM_API_KEY" \
  --llm-model "$LLM_MODEL" \
  --max-tokens 8192

echo
echo "===== 完成，结果目录: $OUT_DIR ====="
