#!/usr/bin/env python
"""08-download-gguf.py — 从 modelscope 下载 GGUF 权重（支持断点续传）

本机 HF 被墙，统一走 modelscope。

用法：
    # 列出该仓库所有量化档及体积
    python setup/08-download-gguf.py --list

    # 下载默认（Qwen3-4B-Instruct-2507 Q4_K_M）
    python setup/08-download-gguf.py

    # 指定仓库 / 量化文件
    python setup/08-download-gguf.py \
        --repo unsloth/Qwen3-4B-Instruct-2507-GGUF \
        --file Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
        --dest models/llama

    # 按关键词挑（如 Q4_K_M / UD-Q4_K_XL）
    python setup/08-download-gguf.py --match UD-Q4_K_XL
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0"}
DEFAULT_REPO = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
DEFAULT_FILE = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
DEFAULT_DEST = "models/llama"


def _open(url: str, headers: dict[str, str] | None = None, method: str = "GET"):
    h = dict(UA)
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h, method=method)
    return urllib.request.urlopen(req, timeout=60)


def list_files(repo: str) -> list[dict]:
    url = (
        f"https://modelscope.cn/api/v1/models/{repo}"
        f"/repo/files?Revision=master&Recursive=true"
    )
    data = _open(url).read()
    import json

    payload = json.loads(data)
    return (payload.get("Data") or {}).get("Files") or []


def remote_size(repo: str, path: str) -> int:
    """用 Range 探测总长度（modelscope 的 HEAD 不给 Content-Length）。"""
    url = file_url(repo, path)
    resp = _open(url, {"Range": "bytes=0-0"})
    cr = resp.headers.get("Content-Range", "")
    if "/" in cr:
        return int(cr.rsplit("/", 1)[1])
    return int(resp.headers.get("Content-Length") or 0)


def file_url(repo: str, path: str) -> str:
    return (
        f"https://modelscope.cn/api/v1/models/{repo}/repo"
        f"?Revision=master&FilePath={urllib.parse.quote(path)}"
    )


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def download(repo: str, path: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(path).name
    url = file_url(repo, path)

    total = remote_size(repo, path)
    have = dest.stat().st_size if dest.exists() else 0

    if total and have == total:
        print(f"  ✓ 已完整存在，跳过：{dest}  ({human(total)})")
        return dest
    if total and have > total:
        print(f"  ! 本地文件比远端大，重新下载：{dest}")
        dest.unlink()
        have = 0

    print(f"  远端长度 : {human(total)}" if total else "  远端长度 : 未知")
    print(f"  本地已有 : {human(have)}")
    print(f"  目标     : {dest}")
    if have:
        print("  → 断点续传")
    print()

    headers = {"Range": f"bytes={have}-"} if have else {}
    t0 = time.time()
    base = have
    last = 0.0
    resp = _open(url, headers)
    mode = "ab" if have and resp.status == 206 else "wb"
    if mode == "wb":
        have = 0
        base = 0

    with open(dest, mode) as fh:
        while True:
            chunk = resp.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            fh.write(chunk)
            have += len(chunk)
            now = time.time()
            if now - last >= 3:
                last = now
                el = max(now - t0, 1e-6)
                speed = (have - base) / el
                if total:
                    pct = 100 * have / total
                    eta = (total - have) / speed if speed > 0 else 0
                    print(
                        f"    {pct:5.1f}%  {human(have)}/{human(total)}"
                        f"  {human(speed)}/s  ETA {eta:5.0f}s",
                        flush=True,
                    )
                else:
                    print(f"    {human(have)}  {human(speed)}/s", flush=True)

    got = dest.stat().st_size
    print()
    if total and got != total:
        print(f"  ✗ 大小不符：期望 {total}，实际 {got}", file=sys.stderr)
        return dest
    el = time.time() - t0
    print(f"  ✓ 完成 {human(got)}  用时 {el:.0f}s")
    print(f"    {dest}")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description="从 modelscope 下载 GGUF")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--file", default=None)
    ap.add_argument("--match", default=None, help="按关键词挑文件，如 Q4_K_M")
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--list", action="store_true", help="只列出可用文件")
    args = ap.parse_args()

    print("=" * 60)
    print(" GGUF 下载器（modelscope）")
    print(f"   仓库: {args.repo}")
    print("=" * 60)

    files = [f for f in list_files(args.repo) if str(f.get("Path", "")).endswith(".gguf")]
    if not files:
        print("  ✗ 仓库中没有 .gguf 文件", file=sys.stderr)
        return 1

    if args.list:
        for f in sorted(files, key=lambda x: x.get("Size", 0) or 0):
            print(f"  {human(f.get('Size', 0) or 0):>10}  {f.get('Path')}")
        return 0

    if args.file:
        want, target = args.file, None
        for f in files:
            if f.get("Path") == want or Path(str(f.get("Path"))).name == want:
                target = f
                break
        if target is None:
            print(f"  ✗ 仓库中找不到 {want}", file=sys.stderr)
            return 1
    elif args.match:
        hits = [f for f in files if args.match in str(f.get("Path"))]
        if not hits:
            print(f"  ✗ 没有匹配 {args.match!r} 的文件", file=sys.stderr)
            return 1
        target = sorted(hits, key=lambda x: x.get("Size", 0) or 0)[0]
    else:
        target = next(
            (f for f in files if Path(str(f.get("Path"))).name == DEFAULT_FILE), None
        )
        if target is None:
            print(f"  ✗ 默认文件 {DEFAULT_FILE} 不在仓库中", file=sys.stderr)
            return 1

    print(f"  选中: {target.get('Path')}  ({human(target.get('Size', 0) or 0)})")
    print()
    download(args.repo, str(target["Path"]), Path(args.dest).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
