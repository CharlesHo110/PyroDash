#!/usr/bin/env python3
"""prefetch_datasets.py —— 预取 PyroDash 数学评测所需数据集（Windows 侧运行）

背景：本机对 GitHub/HF 的连通性是**间歇性**的。评测时 datasets_loader.py 会用
      `load_dataset("openai/gsm8k", ...)` 等从 HF Hub 拉数据，一旦网络不可用就失败。
      本脚本把原始文件一次性抓到 setup/datasets/，并记录 sha，供 WSL 侧离线铺设缓存。

数据集来源（取自 evaluation/evaluation_math/datasets_loader.py）：
  math      -> https://openaipublic.blob.core.windows.net/simple-evals/math_500_test.csv
  gsm8k     -> HF openai/gsm8k                (config main,      split test)
  amc       -> HF zwhe99/amc23                (                  split test)
  minerva   -> HF zwhe99/simplerl-minerva-math(                  split test)
  olympiad  -> HF zwhe99/simplerl-OlympiadBench(                 split test)
  aime2024  -> HF HuggingFaceH4/aime_2024     (                  split train)
  aime2025  -> HF yentinglin/aime_2025        (config default,    split train)

用法：
  python setup/prefetch_datasets.py            # 全部
  python setup/prefetch_datasets.py gsm8k      # 指定数据集
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "datasets"

# 直连优先，失败再走镜像（两者实测都能用，但会间歇性失败，故双源重试）
ENDPOINTS = ["https://huggingface.co", "https://hf-mirror.com"]

HF_DATASETS = {
    "gsm8k": ("openai/gsm8k", ["main/test-00000-of-00001.parquet"]),
    "amc": ("zwhe99/amc23", None),  # None = 自动发现全部数据文件
    "minerva": ("zwhe99/simplerl-minerva-math", None),
    "olympiad": ("zwhe99/simplerl-OlympiadBench", None),
    "aime2024": ("HuggingFaceH4/aime_2024", None),
    "aime2025": ("yentinglin/aime_2025", None),
}

MATH_CSV = "https://openaipublic.blob.core.windows.net/simple-evals/math_500_test.csv"

DATA_EXT = (".parquet", ".csv", ".json", ".jsonl", ".arrow")


def fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "pyrodash-prefetch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_with_fallback(path: str, timeout: int = 60) -> tuple[bytes, str]:
    """依次尝试各 endpoint，返回 (内容, 实际使用的 endpoint)。"""
    last = None
    for ep in ENDPOINTS:
        try:
            return fetch(ep + path, timeout), ep
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"    {ep} 失败: {type(e).__name__}: {e}")
    raise RuntimeError(f"所有源均失败: {path} ({last})")


def api_json(path: str) -> dict:
    body, _ = fetch_with_fallback(path)
    return json.loads(body)


def sha256_of(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def prefetch_hf(key: str, repo: str, files: list[str] | None) -> dict:
    print(f"\n=== {key}: {repo} ===")
    info = api_json(f"/api/datasets/{repo}")
    sha = info.get("sha") or ""
    print(f"  revision sha: {sha[:12]}")

    if files is None:
        sib = [s.get("rfilename") for s in info.get("siblings", [])]
        files = [f for f in sib if f.endswith(DATA_EXT)]
        if not files:
            raise RuntimeError(f"{repo} 未发现数据文件: {sib}")
    print(f"  数据文件 {len(files)} 个: {files}")

    d = OUT / repo.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    records = []
    for f in files:
        body, ep = fetch_with_fallback(f"/datasets/{repo}/resolve/main/{f}", timeout=180)
        p = d / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
        rec = {"path": f, "bytes": len(body), "sha256": sha256_of(body), "via": ep}
        records.append(rec)
        print(f"    ✅ {f}  {len(body):,} B  via {ep}")
    return {"kind": "hf", "repo": repo, "revision_sha": sha, "files": records}


def prefetch_math() -> dict:
    print("\n=== math: simple-evals CSV ===")
    body = fetch(MATH_CSV, timeout=120)
    d = OUT / "simple-evals"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "math_500_test.csv"
    p.write_bytes(body)
    print(f"    ✅ math_500_test.csv  {len(body):,} B")
    return {
        "kind": "url",
        "url": MATH_CSV,
        "files": [{"path": "math_500_test.csv", "bytes": len(body), "sha256": sha256_of(body)}],
    }


def main() -> int:
    wanted = sys.argv[1:] or ["math", *HF_DATASETS]
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}

    for key in wanted:
        try:
            if key == "math":
                manifest["math"] = prefetch_math()
            elif key in HF_DATASETS:
                repo, files = HF_DATASETS[key]
                manifest[key] = prefetch_hf(key, repo, files)
            else:
                print(f"!! 未知数据集: {key}")
                continue
        except Exception as e:  # noqa: BLE001
            print(f"!! {key} 预取失败: {type(e).__name__}: {e}")
            manifest.setdefault(key, {})["error"] = str(e)

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8")

    print("\n========== 汇总 ==========")
    for k, v in manifest.items():
        if v.get("error"):
            print(f"  {k:10} ❌ 失败")
            continue
        n = len(v.get("files", []))
        tot = sum(f["bytes"] for f in v.get("files", []))
        print(f"  {k:10} ✅ {n} 文件 {tot:,} B")
    print(f"\n清单: {manifest_path}")
    print(f"目录: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
