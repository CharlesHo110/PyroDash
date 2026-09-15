#!/usr/bin/env bash
# 07-install-llama.sh — 安装 llama.cpp（Windows CUDA 预编译包，解压即用，无需编译）
#
# 用途：给本机 PyroDash 式 relay 提供 OpenAI 兼容的推理服务
#       （llama-server 自带 /v1/chat/completions + stop 截断 + --jinja 工具调用）
#
# 用法：
#   bash setup/07-install-llama.sh              # 装默认版本（脚本内 LLAMA_TAG）
#   LLAMA_TAG=b10982 bash setup/07-install-llama.sh
#   bash setup/07-install-llama.sh --latest     # 自动解析 GitHub 最新 release
#   CUDA_TAG=13.3 bash setup/07-install-llama.sh # 换 CUDA 版本（需对应驱动）
#
# 说明：
#   - CUDA 12.4 运行时包在 CUDA 12.x 驱动上可用（次版本兼容），
#     本机驱动 537.70（CUDA 12.2）满足条件。
#   - 产物落在 setup/llama.cpp/，已被 setup/.gitignore 忽略。
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/llama.cpp"
DL="$DEST/.download"
CUDA_TAG="${CUDA_TAG:-12.4}"
LLAMA_TAG="${LLAMA_TAG:-b10982}"
GH="https://github.com/ggml-org/llama.cpp/releases/download"

log() { printf '  %s\n' "$*"; }
die() { printf '  ✗ %s\n' "$*" >&2; exit 1; }

# ---------- 0. 可选：解析最新 tag ----------
if [ "${1:-}" = "--latest" ]; then
  log "解析 GitHub 最新 release ..."
  tmp="$(mktemp)"
  curl -s -m 40 -H 'User-Agent: Mozilla/5.0' \
    "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20" -o "$tmp" \
    || die "GitHub API 不可达"
  LLAMA_TAG="$(python - "$tmp" "$CUDA_TAG" <<'PY'
import json, sys, pathlib
try:
    rels = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
except Exception:
    sys.exit(0)
want = f"win-cuda-{sys.argv[2]}-x64"
for r in rels:
    for a in r.get("assets", []):
        n = a.get("name", "")
        if n.startswith("llama-") and want in n:
            print(r["tag_name"]); sys.exit(0)
PY
)"
  rm -f "$tmp"
  [ -n "$LLAMA_TAG" ] || die "未能解析出含 CUDA $CUDA_TAG 的最新 tag"
  log "最新 tag = $LLAMA_TAG"
fi

BIN_ZIP="llama-${LLAMA_TAG}-bin-win-cuda-${CUDA_TAG}-x64.zip"
RT_ZIP="cudart-llama-bin-win-cuda-${CUDA_TAG}-x64.zip"

echo "=========================================================="
echo " 安装 llama.cpp"
echo "   tag        : $LLAMA_TAG"
echo "   CUDA 运行时: $CUDA_TAG"
echo "   目标目录   : $DEST"
echo "=========================================================="

mkdir -p "$DL"

fetch() {
  local name="$1" dest="$DL/$1"
  local url="$GH/$LLAMA_TAG/$name"
  # 已完整下载过就跳过（用大小粗判：解压出的 ok 标记优先）
  if [ -f "$dest" ] && [ -f "$dest.ok" ]; then
    log "跳过（已下载）: $name  ($(du -m "$dest" | cut -f1) MB)"
    return 0
  fi
  log "下载: $name"
  for attempt in 1 2 3; do
    if curl -fL --retry 3 --retry-delay 2 -C - \
        -A 'Mozilla/5.0' --connect-timeout 30 -m 3600 \
        -o "$dest" "$url"; then
      touch "$dest.ok"
      log "  ✓ $(du -m "$dest" | cut -f1) MB"
      return 0
    fi
    log "  第 $attempt 次失败，重试 ..."
    sleep 3
  done
  die "下载失败: $url"
}

fetch "$BIN_ZIP"
fetch "$RT_ZIP"

echo
log "解压 ..."
python - "$DL/$BIN_ZIP" "$DL/$RT_ZIP" "$DEST" <<'PY'
import sys, zipfile, pathlib
bin_zip, rt_zip, dest = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
for z in (bin_zip, rt_zip):
    with zipfile.ZipFile(z) as zf:
        zf.extractall(dest)
    print(f"  ✓ {pathlib.Path(z).name}")
PY

echo
echo "=== 产物校验 ==="
for exe in llama-server.exe llama-cli.exe llama-bench.exe; do
  if [ -f "$DEST/$exe" ]; then
    log "✓ $exe"
  else
    log "· $exe 不存在（可能已改名）"
  fi
done
echo
log "CUDA 后端 DLL:"
ls "$DEST"/*.dll 2>/dev/null | while read -r f; do
  log "   $(basename "$f")  ($(du -m "$f" | cut -f1) MB)"
done

echo
echo "=== 版本自检 ==="
if [ -f "$DEST/llama-server.exe" ]; then
  "$DEST/llama-server.exe" --version 2>&1 | head -5 | sed 's/^/  /' || true
fi

echo
echo "完成。启动示例："
echo "  $DEST/llama-server.exe -m models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \\"
echo "      --host 127.0.0.1 --port 8080 -ngl 99 -c 16384 --jinja"
