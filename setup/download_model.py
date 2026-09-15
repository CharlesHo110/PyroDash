#!/usr/bin/env python3
"""PyroDash 权重下载器（支持 ModelScope 主源 / HuggingFace 备源 + 断点续传）。

背景：本机所在网络对 github.com / huggingface.co / hf-mirror.com 存在阻断，
      但 modelscope.cn 可达且已镜像同一模型（文件大小逐字节一致）。

用法:
    python download_model.py [--source auto|modelscope|hf] [--repo <id>] [--dest <dir>]
默认:
    source = auto（先 ModelScope，失败再试 HF）
    repo   = pyromind/PyroDash-4B-GRPO-Lambda-0.05
    dest   = <项目根>/models/<repo 名>

中断后直接重跑即可续传（已完成的文件按大小校验后跳过）。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import requests

HERE = pathlib.Path(__file__).resolve().parent
PROJ = HERE.parent

MS_API = "https://www.modelscope.cn/api/v1/models/{repo}/repo/files"
MS_FILE = "https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={path}"
HF_FILE = "https://huggingface.co/{repo}/resolve/main/{path}"

SKIP_SUFFIX = (".png", ".jpg", ".jpeg", ".gif", ".mp4")
CHUNK = 1 << 20  # 1 MiB


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def list_files_modelscope(repo: str) -> list[dict]:
    r = requests.get(
        MS_API.format(repo=repo),
        params={"Revision": "master", "Recursive": "true"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json().get("Data", {})
    files = data.get("Files") or data.get("files") or []
    out = []
    for f in files:
        p, sz = f.get("Path"), f.get("Size") or 0
        if p and not p.endswith(SKIP_SUFFIX):
            out.append({"path": p, "size": sz})
    return out


def list_files_hf(repo: str) -> list[dict]:
    r = requests.get(
        f"https://huggingface.co/api/models/{repo}", params={"blobs": "true"}, timeout=30
    )
    r.raise_for_status()
    out = []
    for s in r.json().get("siblings", []):
        p, sz = s["rfilename"], s.get("size") or 0
        if not p.endswith(SKIP_SUFFIX) and not p.startswith("original/"):
            out.append({"path": p, "size": sz})
    return out


def download_one(url: str, dest: pathlib.Path, size: int, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (size == 0 or dest.stat().st_size == size):
        log(f"跳过（已完整） {label}")
        return

    part = dest.with_suffix(dest.suffix + ".part")
    attempt = 0
    while True:
        attempt += 1
        have = part.stat().st_size if part.exists() else 0
        if size and have >= size:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(20, 120)) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                total = size or (have + int(r.headers.get("Content-Length") or 0))
                mode = "ab" if (have and r.status_code == 206) else "wb"
                if mode == "wb":
                    have = 0
                with open(part, mode) as fh:
                    last = time.time()
                    done = have
                    for chunk in r.iter_content(CHUNK):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 15:
                            pct = f"{100 * done / total:.1f}%" if total else "?"
                            log(f"  {label}: {human(done)}/{human(total)} ({pct})")
                            last = time.time()
            if size and part.stat().st_size != size:
                raise RuntimeError(
                    f"大小不符 got={part.stat().st_size} want={size}，将重试"
                )
            break
        except Exception as e:  # noqa: BLE001
            if attempt >= 8:
                raise
            wait = min(30, 2 ** attempt)
            log(f"  !! {label} 第 {attempt} 次失败：{e}；{wait}s 后重试（已存 {human(part.stat().st_size if part.exists() else 0)}）")
            time.sleep(wait)

    part.replace(dest)
    log(f"完成 {label}  {human(dest.stat().st_size)}")


def verify(dest: pathlib.Path) -> None:
    tok_path = dest / "tokenizer.json"
    cfg_path = dest / "config.json"
    if not tok_path.exists():
        raise SystemExit("!! 缺少 tokenizer.json")
    tok = json.loads(tok_path.read_text(encoding="utf-8"))
    added = {a["content"]: a["id"] for a in tok.get("added_tokens", [])}
    tid = added.get("<|llm_offload|>")
    if tid is None:
        raise SystemExit("!! tokenizer 中未找到 <|llm_offload|>，模型不完整或选错仓库")
    weights = sum(f.stat().st_size for f in dest.glob("*.safetensors"))
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    log("================ 校验 ================")
    log(f"<|llm_offload|> -> token id {tid}")
    log(f"架构: {cfg.get('architectures')} | model_type: {cfg.get('model_type')}")
    log(f"权重体积: {human(weights)}")
    if weights < 8_000_000_000:
        raise SystemExit("!! 权重体积异常，下载不完整")
    log("校验通过 ✅")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="auto", choices=["auto", "modelscope", "hf"])
    ap.add_argument("--repo", default="pyromind/PyroDash-4B-GRPO-Lambda-0.05")
    ap.add_argument("--dest", default=None)
    a = ap.parse_args()

    dest = pathlib.Path(a.dest) if a.dest else PROJ / "models" / a.repo.split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)
    log(f"仓库={a.repo}")
    log(f"目标={dest}")

    sources = ["modelscope", "hf"] if a.source == "auto" else [a.source]
    files: list[dict] = []
    use = None
    for s in sources:
        try:
            files = list_files_modelscope(a.repo) if s == "modelscope" else list_files_hf(a.repo)
            use = s
            break
        except Exception as e:  # noqa: BLE001
            log(f"源 {s} 不可用：{e}")
    if not use:
        raise SystemExit("!! 所有下载源均不可用")

    total = sum(f["size"] for f in files)
    log(f"下载源={use}  文件数={len(files)}  总计={human(total)}")
    for f in files:
        url = (
            MS_FILE.format(repo=a.repo, path=f["path"])
            if use == "modelscope"
            else HF_FILE.format(repo=a.repo, path=f["path"])
        )
        download_one(url, dest / f["path"], f["size"], f["path"])

    verify(dest)
    log("下一步: bash setup/04-run-math-eval.sh gsm8k")
    return 0


if __name__ == "__main__":
    sys.exit(main())
