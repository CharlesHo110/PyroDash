#!/usr/bin/env bash
# 06-run-windows-native.sh —— 【推荐路径】Windows 原生跑通「本地小模型 + 内网 DeepSeek」评测
#
# 为什么有这条路径：
#   vLLM 只支持 Linux，官方脚本要 WSL2；而装 WSL2 需要管理员权限 + 升级 NVIDIA 驱动
#   （torch 2.11 是 CUDA 13 构建，要求驱动 >= 580）。
#   本脚本改用 「transformers + 自建 OpenAI 兼容服务（serve_small.py）」，
#   配 torch 2.7.1+cu118 —— 当前驱动 537.70（CUDA 12.2）即可直接用 GPU，
#   **全程不需要管理员权限，也不需要升级驱动**。
#
# 前置（已在本机完成，见 README）：
#   python -m venv setup/.venv-win
#   setup/.venv-win/Scripts/python.exe -m pip install torch==2.7.1+cu118 \
#       --index-url https://download.pytorch.org/whl/cu118 \
#       --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
#   setup/.venv-win/Scripts/python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
#       transformers==5.5.3 datasets==5.0.1 accelerate math-verify mathruler pylatexenc \
#       requests tqdm pandas openai httpx huggingface-hub
#
# 用法：
#   bash setup/06-run-windows-native.sh                 # 冒烟：gsm8k 前 5 题
#   bash setup/06-run-windows-native.sh 20              # 冒烟：gsm8k 前 20 题
#   bash setup/06-run-windows-native.sh --full          # 全量（跑全部数据集，耗时长）
#   bash setup/06-run-windows-native.sh --dataset aime2024 --limit 2 --max-tokens 2000
#   bash setup/06-run-windows-native.sh --stop          # 只停掉小模型服务
#   bash setup/06-run-windows-native.sh --compare 50      # 三组对照实验（计划 §4 验证清单第 3 项）
#   PYRODASH_MODEL=models/PyroDash-4B-GRPO-Lambda-0.6 bash setup/06-run-windows-native.sh --compare 50 --tag l06
set -uo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$PROJ/setup/.venv-win/Scripts/python.exe"
# 可切换 checkpoint（对比 λ 用）：
#   PYRODASH_MODEL=models/PyroDash-4B-GRPO-Lambda-0.6 bash setup/06-run-windows-native.sh 50
_PM="${PYRODASH_MODEL:-models/PyroDash-4B-GRPO-Lambda-0.05}"
case "$_PM" in
  /*|[A-Za-z]:*) MODEL_DIR="$_PM" ;;
  *) MODEL_DIR="$PROJ/$_PM" ;;
esac
PORT=8001
SMALL_URL="http://127.0.0.1:$PORT/v1"

# ---- 内网大模型（tkoffice 网关，仅公司网络/VPN 可达）----
# ⚠️ 切勿把密钥硬编码进仓库（本仓库曾这么干过，已清除）。按以下优先级提供：
#   1) export LLM_API_KEY=sk-xxx
#   2) 写入 setup/.llm_key（已在 .gitignore 忽略，只放一行密钥）
#   3) 从 aiConfig/claude/enableClaudeChina.py 拷
export LLM_BASE_URL="${LLM_BASE_URL:-https://ai-api.bj.tkoffice.cn/v1}"
if [ -z "${LLM_API_KEY:-}" ] && [ -f "$PROJ/setup/.llm_key" ]; then
  LLM_API_KEY="$(tr -d '[:space:]' < "$PROJ/setup/.llm_key")"
fi
export LLM_API_KEY="${LLM_API_KEY:-}"
export LLM_MODEL="${LLM_MODEL:-deepseek-v4-pro}"

# ---- 小模型单请求超时（秒）----
# 服务端 serve_small.py 是 ThreadingHTTPServer + 全局锁，GPU 实际串行生成：
# 并发提交时后面的请求要排队，排队时间也算进这个超时。长生成（λ 大、小模型自己推理多）
# 或高并发下必须调大，否则会 requests.ReadTimeout。
export SMALL_TIMEOUT="${SMALL_TIMEOUT:-3600}"

# ---- 数据集离线缓存（setup/05 产出；缺失则回退到在线镜像）----
if [ -d "$PROJ/setup/hf_home/datasets" ]; then
  export HF_HOME="$PROJ/setup/hf_home"
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
  echo "[env] 使用离线数据集缓存 $HF_HOME（HF_HUB_OFFLINE=1）"
else
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  echo "[env] 未找到离线缓存，改为在线拉取 HF_ENDPOINT=$HF_ENDPOINT（可先跑 setup/05-prepare-datasets.sh）"
fi

server_up() { curl -s -m 5 -o /dev/null "http://127.0.0.1:$PORT/v1/models"; }

stop_server() {
  echo "[stop] 停止小模型服务 ..."
  powershell -NoProfile -Command \
    "Get-Process python -ErrorAction SilentlyContinue | Where-Object { \$_.Path -like '*venv-win*' } | Stop-Process -Force" \
    2>/dev/null || true
  sleep 2
  echo "[stop] 已停止"
}

if [ "${1:-}" = "--stop" ]; then stop_server; exit 0; fi

# ---------------------------------------------------------------- 前置检查
[ -x "$VENV_PY" ] || { echo "✗ 未找到 venv: $VENV_PY"; echo "  请先按脚本头部注释创建（约需 3GB 磁盘）"; exit 1; }
[ -d "$MODEL_DIR" ] || { echo "✗ 未找到模型目录: $MODEL_DIR"; echo "  请先跑 setup/03-download-model.sh 或 setup/download_model.py"; exit 1; }
[ -n "$LLM_API_KEY" ] || { echo "✗ 缺少 LLM_API_KEY（勿写入仓库）。"
  echo "  请 export LLM_API_KEY=...，或写入 $PROJ/setup/.llm_key"
  echo "  密钥来源：aiConfig/claude/enableClaudeChina.py"; exit 1; }

# ---------------------------------------------------------------- 启动小模型服务
if server_up; then
  echo "[serve] 服务已在运行：$SMALL_URL"
else
  echo "[serve] 启动小模型服务（加载 8.45GB 权重，约需 40-60 秒）..."
  LOG="$PROJ/setup/serve_small.log"; ERR="$PROJ/setup/serve_small.err"
  rm -f "$LOG" "$ERR"
  HF_HUB_OFFLINE=1 powershell -NoProfile -Command \
    "Start-Process -FilePath '$(cygpath -w "$VENV_PY")' \
     -ArgumentList '$(cygpath -w "$PROJ/setup/serve_small.py")','--model','$(cygpath -w "$MODEL_DIR")','--port','$PORT','--served-model-name','small-model' \
     -RedirectStandardOutput '$(cygpath -w "$LOG")' -RedirectStandardError '$(cygpath -w "$ERR")' \
     -WorkingDirectory '$(cygpath -w "$PROJ")' -WindowStyle Hidden" >/dev/null 2>&1

  for i in $(seq 1 40); do
    if server_up; then echo "[serve] ✅ 就绪（等待 $((i*5)) 秒）"; break; fi
    if [ "$i" = 40 ]; then
      echo "✗ 服务启动超时，日志尾部："; tail -20 "$LOG" 2>/dev/null; tail -10 "$ERR" 2>/dev/null
      exit 1
    fi
    sleep 5
  done
fi

# ---------------------------------------------------------------- 执行评测
ARGS=("$@")
if [ "${1:-}" = "--full" ]; then
  ARGS=(--datasets math gsm8k minerva olympiad aime2024 aime2025 --max-tokens 8192)
fi

COMPARE=0
if [ "${1:-}" = "--compare" ]; then
  COMPARE=1; shift
  ARGS=("$@")
  if [ ${#ARGS[@]} -eq 0 ]; then
    ARGS=(--dataset gsm8k --limit 50 --max-tokens 4096)
  elif [[ "${ARGS[0]}" =~ ^[0-9]+$ ]]; then
    ARGS=(--dataset gsm8k --limit "${ARGS[0]}" --max-tokens 4096 "${ARGS[@]:1}")
  fi
fi

if [ ${#ARGS[@]} -eq 0 ]; then
  ARGS=(--dataset gsm8k --limit 5 --max-tokens 2500)
elif [[ "${ARGS[0]}" =~ ^[0-9]+$ ]]; then
  # 首个参数是数字 -> 视为 gsm8k 的样本数，并保留其余选项
  ARGS=(--dataset gsm8k --limit "${ARGS[0]}" --max-tokens 2500 "${ARGS[@]:1}")
fi

if [ "$COMPARE" = "1" ]; then
  echo "[run] compare_arms.py ${ARGS[*]}"
  echo "[run] 模型 $MODEL_DIR"
  echo "[run] LLM=$LLM_BASE_URL model=$LLM_MODEL"
  echo "---------------------------------------------------------------"
  "$VENV_PY" "$PROJ/setup/compare_arms.py" \
    --model-path "$MODEL_DIR" \
    --small-base-url "$SMALL_URL" --small-model small-model \
    --llm-base-url "$LLM_BASE_URL" --llm-api-key "$LLM_API_KEY" --llm-model "$LLM_MODEL" \
    "${ARGS[@]}"
  rc=$?
  echo "---------------------------------------------------------------"
  echo "[run] 退出码 $rc"
  echo "[run] 停服务请执行：bash setup/06-run-windows-native.sh --stop"
  exit $rc
fi

echo "[run] smoke_offload.py ${ARGS[*]}"
echo "[run] LLM=$LLM_BASE_URL model=$LLM_MODEL"
echo "---------------------------------------------------------------"

if [ "${1:-}" = "--full" ]; then
  set -x
  "$VENV_PY" "$PROJ/evaluation/evaluation_math/math_eval.py" \
    --model-path "$MODEL_DIR" \
    --small-base-url "$SMALL_URL" --small-model small-model \
    --llm-base-url "$LLM_BASE_URL" --llm-api-key "$LLM_API_KEY" --llm-model "$LLM_MODEL" \
    --output-dir "$PROJ/results" \
    "${ARGS[@]}"
  rc=$?
else
  "$VENV_PY" "$PROJ/setup/smoke_offload.py" \
    --model-path "$MODEL_DIR" \
    --small-base-url "$SMALL_URL" --small-model small-model \
    --llm-base-url "$LLM_BASE_URL" --llm-api-key "$LLM_API_KEY" --llm-model "$LLM_MODEL" \
    "${ARGS[@]}"
  rc=$?
fi

echo "---------------------------------------------------------------"
echo "[run] 退出码 $rc（0 = 冒烟判定通过）"
echo "[run] 停服务请执行：bash setup/06-run-windows-native.sh --stop"
exit $rc
