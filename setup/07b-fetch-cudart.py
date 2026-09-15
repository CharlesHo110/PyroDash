#!/usr/bin/env python
"""07b-fetch-cudart.py — 从清华 PyPI 镜像取 CUDA 12.4 运行时 DLL

为什么不直接下 GitHub 上的 ``cudart-llama-bin-win-cuda-12.4-x64.zip``：
那个包 391MB，GitHub 在国内实测只有 ~165 KB/s（要 40 分钟）；同样的 DLL
打成 PyPI wheel 放在清华镜像上，实测 1 分钟内拉完，内容完全等价。

取两个包：
    nvidia-cuda-runtime-cu12==12.4.127  → cudart64_12.dll
    nvidia-cublas-cu12==12.4.5.8        → cublas64_12.dll, cublasLt64_12.dll

版本特意选 12.4.x，与 llama.cpp 的 `bin-win-cuda-12.4-x64` 构建对齐。
（用户驱动 537.70 = CUDA 12.2，CUDA 12.x 的 minor version compatibility
保证 12.4 运行时可用。）

    python setup/07b-fetch-cudart.py
    python setup/07b-fetch-cudart.py --dest setup/llama.cpp --mirror https://pypi.tuna.tsinghua.edu.cn
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent

PACKAGES = [
    ("nvidia-cuda-runtime-cu12", "12.4.127"),
    ("nvidia-cublas-cu12", "12.4.5.8"),
]

# 只抽这些 DLL，其余（头文件、静态库等）不要
NEEDED_PREFIXES = ("cudart64_", "cublas64_", "cublasLt64_", "nvrtc64_", "nvJitLink")


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def wheel_url(pkg: str, version: str, mirror: str) -> tuple[str, int]:
    """查 PyPI JSON 拿到指定版本的 win_amd64 wheel 下载地址与大小。"""
    api = f"{mirror.rstrip('/')}/pypi/{pkg}/json"
    with urllib.request.urlopen(api, timeout=40) as fh:
        data = json.load(fh)
    for entry in data["releases"].get(version, []):
        name = entry["filename"]
        if "win_amd64" in name and name.endswith(".whl"):
            # PyPI JSON 里给的 url 指向官方 CDN files.pythonhosted.org，
            # **不是镜像本体**。直接用它等于绕过了镜像、照样走国际链路（实测
            # 只有 ~106 KB/s）。必须把 host 换成镜像，实测 5.07 MB/s，快 48 倍。
            url = entry["url"].replace(
                "https://files.pythonhosted.org", mirror.rstrip("/")
            )
            return url, int(entry.get("size") or 0)
    raise SystemExit(f"  ✗ {pkg}=={version} 没有 win_amd64 wheel")


def download(url: str, dest: Path, size: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=120) as resp, open(dest, "wb") as out:
        total = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
            if size and (total % (32 << 20) < (1 << 20) or total == size):
                pct = 100 * total / size
                rate = total / max(time.time() - t0, 1e-6)
                print(f"      {pct:5.1f}%  {human(total)}/{human(size)}  {human(rate)}/s", flush=True)
    print(f"      完成 {human(total)}，用时 {time.time() - t0:.1f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description="取 CUDA 12.4 运行时 DLL")
    ap.add_argument("--dest", default=str(PROJ / "setup" / "llama.cpp"),
                    help="释放 DLL 的目录（llama-server.exe 所在处）")
    ap.add_argument("--mirror", default="https://pypi.tuna.tsinghua.edu.cn",
                    help="PyPI 镜像（用清华源，别用官方源）")
    args = ap.parse_args()

    dest = Path(args.dest)
    if not dest.is_dir():
        raise SystemExit(f"  ✗ 目标目录不存在: {dest}\n    请先运行 setup/07-install-llama.sh")

    print("=" * 68)
    print(" 取 CUDA 12.4 运行时 DLL（清华 PyPI 镜像）")
    print("=" * 68)
    print(f"   镜像    : {args.mirror}")
    print(f"   释放到  : {dest}")

    tmp = Path(tempfile.mkdtemp(prefix="cudart-"))
    extracted: list[str] = []
    try:
        for pkg, version in PACKAGES:
            print(f"\n  [{pkg}=={version}]")
            url, size = wheel_url(pkg, version, args.mirror)
            whl = tmp / f"{pkg}-{version}.whl"
            print(f"     来源 {human(size) if size else '?'}")
            download(url, whl, size)

            with zipfile.ZipFile(whl) as zf:
                for member in zf.namelist():
                    base = Path(member).name
                    if not base.lower().endswith(".dll"):
                        continue
                    if not base.startswith(NEEDED_PREFIXES):
                        continue
                    target = dest / base
                    with zf.open(member) as src, open(target, "wb") as out:
                        shutil.copyfileobj(src, out)
                    extracted.append(base)
                    print(f"     ✅ {base}  ({human(target.stat().st_size)})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 68)
    if extracted:
        print(f" ✅ 释放 {len(extracted)} 个 DLL 到 {dest}")
        for name in ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"):
            mark = "✅" if (dest / name).exists() else "⚠️ 缺"
            print(f"    {mark} {name}")
    else:
        print(" ⚠ 没有释放任何 DLL —— 检查包的内部结构是否变化")
    print("=" * 68)
    return 0 if extracted else 1


if __name__ == "__main__":
    raise SystemExit(main())
