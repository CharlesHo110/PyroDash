#!/usr/bin/env bash
# 02-setup-wsl.sh —— 在 WSL Ubuntu 中执行（非 Windows）
# 作用：驱动硬校验 → apt 依赖 → Python venv → 安装 vLLM/torch（CUDA 13）→ 版本校验
# 用法：bash /mnt/d/hecan/PyroDash/setup/02-setup-wsl.sh
#
# 🔴 前置条件：Windows 侧 NVIDIA 驱动必须 >= 580（CUDA 13）。
#    torch 2.11.0 是 CUDA 13 构建，驱动 537.70（CUDA 12.2）会在 CUDA 初始化时报
#    "CUDA driver version is insufficient for CUDA runtime version"。

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$HERE")"
VENV="${VENV:-$HOME/pyrodash-venv}"
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
REQ="$HERE/requirements-math.txt"

echo "===== 0) 系统信息 ====="
grep -E '^(NAME|VERSION)=' /etc/os-release
python3 --version

echo
echo "===== 1) apt 依赖 ====="
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip git curl

echo
echo "===== 2) GPU / 驱动硬校验（关键门禁）====="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  cat >&2 <<'EOF'
!! WSL 内没有 nvidia-smi —— Windows 侧 NVIDIA 驱动过旧或不支持 WSL CUDA。

   请在 Windows 上升级 NVIDIA 驱动到 [最新版 / >= 580]，然后：
     1) Windows 执行 wsl --shutdown
     2) 重新进入 WSL 再跑本脚本

   参考：CUDA 13.0 GA 官方要求 Linux 驱动 >= 580.65.06；
         WSL2 使用 Windows 驱动，故 Windows 驱动需 >= 580。
   当前驱动为 537.70（CUDA 12.2），不满足要求。
EOF
  ls /usr/lib/wsl/lib 2>/dev/null | head >&2 || echo "(/usr/lib/wsl/lib 不存在)" >&2
  exit 1
fi

nvidia-smi | head -4
CUDA_VER="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p' | head -1)"
CUDA_MAJOR="${CUDA_VER%%.*}"
echo "检测到驱动支持的最高 CUDA 版本: ${CUDA_VER:-未知}"

if [ -z "${CUDA_MAJOR:-}" ] || [ "${CUDA_MAJOR:-0}" -lt 13 ]; then
  cat >&2 <<EOF
!! 驱动不满足要求：CUDA ${CUDA_VER:-未知} < 13.0

   torch 2.11.0 / vLLM 0.25.1 是 CUDA 13 构建，必须驱动 >= 580（CUDA 13.0 GA 官方要求 >= 580.65.06）。
   请升级 Windows 侧 NVIDIA 驱动到最新版，然后 wsl --shutdown 并重进 WSL。
   （继续安装也能装完，但运行 vLLM 时必然 CUDA 初始化失败。）
EOF
  echo
  read -r -p "仍要继续安装依赖? [y/N] " ans </dev/tty || ans="n"
  case "$ans" in [yY]*) echo "继续..." ;; *) exit 1 ;; esac
fi

echo
echo "===== 3) 建立虚拟环境 ====="
[ -d "$VENV" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip -i "$PIP_MIRROR"
pip config set global.index-url "$PIP_MIRROR" >/dev/null
pip config set global.trusted-host "$(echo "$PIP_MIRROR" | sed -E 's#https?://([^/]+).*#\1#')" >/dev/null

echo
echo "===== 4) 安装依赖（vLLM 0.25.1 + torch 2.11.0，约 8-10GB，耗时较长）====="
echo "清单: $REQ"
pip install -r "$REQ"

echo
echo "===== 5) 版本校验 ====="
python - <<'PY'
import torch, transformers
print("torch        :", torch.__version__)
print("torch cuda   :", torch.version.cuda)
print("cuda 可用    :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("显卡         :", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f"显存         : {total/1073741824:.1f} GB 总量 / {free/1073741824:.1f} GB 空闲")
print("transformers :", transformers.__version__)
try:
    import vllm
    print("vllm         :", vllm.__version__)
except Exception as e:
    print("vllm 导入失败 :", e)
PY

echo
echo "===== 完成 ====="
echo "虚拟环境: $VENV"
echo "下一步: bash $PROJ/setup/04-run-math-eval.sh gsm8k"
echo "（权重已提前下载并通过校验，03 步可跳过）"
