#!/usr/bin/env bash
# 05-prepare-datasets.sh —— 在 WSL Ubuntu 中执行
# 作用：把 Windows 侧已生成的**预置离线 HF 缓存**（setup/hf_home，9.8MB）安装到 WSL 的
#       ~/.cache/huggingface，使评测时 `load_dataset("openai/gsm8k", ...)` 完全离线可用。
#
# 背景与结论（已实测）：
#   本机对 HF 的连通性是**间歇性**的；评测中途断网会导致数据集加载失败。
#   实测发现：**只铺 hub 缓存（原始 parquet + refs）不够**——datasets 库还需要 API 元数据，
#   HF_HUB_OFFLINE=1 时会报 ConnectionError。
#   可行的做法是让 datasets 在线跑一次，产出完整的 $HF_HOME（含 datasets/ arrow 缓存 + hub/ 原始缓存），
#   之后该目录在离线模式下可直接复用（已实测 6 个数据集全部通过）。
#   setup/hf_home 就是这样一个缓存，由 datasets 5.0.1 生成（故 requirements 里 pin 了该版本）。
#
# 用法：bash /mnt/d/hecan/PyroDash/setup/05-prepare-datasets.sh [--check]
#   --check  只校验离线加载是否正常，不重新安装

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/hf_home"
DST="${HF_HOME:-$HOME/.cache/huggingface}"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

echo "===== 安装预置离线数据集缓存 ====="
echo "源  : $SRC"
echo "目标: $DST"

if [ ! -d "$SRC" ]; then
  echo "!! 未找到预置缓存 $SRC" >&2
  echo "   请在 Windows 侧执行: python setup/prefetch_datasets.py" >&2
  echo "   然后在线跑一次 datasets 生成缓存（见 setup/README.md「数据集离线化」一节）" >&2
  exit 1
fi

if [ "$CHECK_ONLY" -eq 0 ]; then
  mkdir -p "$DST"
  # 用 -a 保留结构；已存在文件覆盖。缓存的目录名带 -- / ___ 分隔，不会与其它内容冲突。
  for sub in hub datasets; do
    if [ -d "$SRC/$sub" ]; then
      echo "  安装 $sub ..."
      cp -a "$SRC/$sub/." "$DST/$sub/"
    fi
  done
  echo "  已安装: $(du -sh "$DST" | cut -f1)"
fi

echo
echo "===== 离线加载自检（HF_HUB_OFFLINE=1）====="
if ! python3 -c "import datasets" 2>/dev/null; then
  echo "  (datasets 未安装，跳过；装完依赖后执行 bash $0 --check 复验)"
  exit 0
fi

HF_HOME="$DST" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python3 - <<'PY'
import sys
from datasets import load_dataset
specs = [
    ("openai/gsm8k", "main", "test"),
    ("zwhe99/amc23", None, "test"),
    ("zwhe99/simplerl-minerva-math", None, "test"),
    ("zwhe99/simplerl-OlympiadBench", None, "test"),
    ("HuggingFaceH4/aime_2024", None, "train"),
    ("yentinglin/aime_2025", "default", "train"),
]
ok = 0
for name, cfg, split in specs:
    try:
        ds = load_dataset(name, cfg, split=split) if cfg else load_dataset(name, split=split)
        print(f"  ✅ {name:34} {len(ds):5} 条  {ds.column_names}")
        ok += 1
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ {name:34} {type(e).__name__}: {str(e)[:100]}")
print(f"\n离线就绪: {ok}/{len(specs)}")
sys.exit(0 if ok == len(specs) else 1)
PY

echo
echo "===== 完成 ====="
echo "评测脚本已默认使用该缓存（HF_HOME=$DST，HF_HUB_OFFLINE=1）"
