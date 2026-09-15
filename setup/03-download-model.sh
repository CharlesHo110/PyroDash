#!/usr/bin/env bash
# 03-download-model.sh —— 可在 WSL 或 Windows Git Bash 中执行
# 作用：下载 PyroDash-4B 权重到项目 models/ 目录，并校验 <|llm_offload|> 特殊 token
#
# 网络说明：本机网络对 huggingface.co / hf-mirror.com / github.com 有阻断，
#          但 modelscope.cn（魔搭）可达且镜像了同一模型（文件大小逐字节一致），
#          因此默认走 ModelScope，HF 仅作备源。
#
# 用法:
#   bash setup/03-download-model.sh                       # 默认 Lambda-0.05，魔搭主源
#   bash setup/03-download-model.sh pyromind/PyroDash-4B-GRPO-Lambda-0.6
#   SOURCE=hf bash setup/03-download-model.sh             # 强制走 HF
#   DEST=/path/to/dir bash setup/03-download-model.sh     # 自定义目标目录

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$HERE")"

REPO="${1:-pyromind/PyroDash-4B-GRPO-Lambda-0.05}"
SOURCE="${SOURCE:-modelscope}"
VENV="${VENV:-$HOME/pyrodash-venv}"

# 选 Python：优先虚拟环境，其次系统 python3/python
PY=""
if [ -x "$VENV/bin/python" ]; then
  PY="$VENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
else
  echo "!! 未找到 python，请先执行 02-setup-wsl.sh" >&2
  exit 1
fi
echo "使用 Python: $PY ($($PY --version 2>&1))"

# 依赖：requests（WSL 里装过 vllm 就一定有）
$PY -c "import requests" 2>/dev/null || $PY -m pip install -q requests

ARGS=(--source "$SOURCE" --repo "$REPO")
if [ -n "${DEST:-}" ]; then
  ARGS+=(--dest "$DEST")
fi

exec $PY "$HERE/download_model.py" "${ARGS[@]}"
