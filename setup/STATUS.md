# 部署进度台账（STATUS）

> 目标：在 `D:\hecan\PyroDash` 部署 PyroDash-4B（本地小模型）+ 内网 DeepSeek（远端大模型），跑通数学评测
> 最后更新：2026-09-15

## ✅ 结论：已在 Windows 原生环境**跑通全链路**（无需管理员权限）

```
Dataset:     gsm8k
Accuracy:    100.00% (5/5)
Offload:     5/5 (100.0%)
小模型 tokens: prompt 614 + completion 431
大模型 tokens: prompt 2097 + completion 1844
总耗时:      55.6s
```

**架构**：`transformers` 本地推理（RTX 3060，含 `<|llm_offload|>` 停止串）+ 自建 OpenAI 兼容服务 → 触发 offload → 接力内网 `deepseek-v4-pro` → `math_verify` 打分。

## 两条路径的对比

| | 路径 A：WSL2 + vLLM（官方文档路径） | **路径 B：Windows 原生（已跑通）** |
|---|---|---|
| 需要管理员权限 | ❌ **需要**（启用 WSL2 功能） | ✅ **不需要** |
| 需要升级 NVIDIA 驱动 | ❌ **需要 ≥580**（torch 2.11 = CUDA 13 构建） | ✅ **不需要**（torch 2.7.1+cu118，驱动 537.70 够用） |
| 状态 | 🔴 阻塞（Pi 无法弹 UAC） | ✅ **已验证可用** |
| 推理速度 | 快（vLLM 优化） | 8.9~10.2 tok/s（线性注意力走了 torch 回退实现） |
| 入口脚本 | `01→02→05→04` | **`06-run-windows-native.sh`** |

> 路径 A 未废弃：若之后拿到管理员权限，仍可按 `01/02/03/04` 走 WSL2+vLLM 获取更高吞吐。

## 进度总览

| # | 步骤 | 状态 | 证据 |
|---|---|---|---|
| 1 | 仓库落地 `D:\hecan\PyroDash` | ✅ | `git clone` 成功 |
| 2 | 官方模型存在性核实 | ✅ | HF 3 个 + 魔搭同名仓库 API 均 200 |
| 3 | 核心机制 `<|llm_offload|>` | ✅ | **token id 248077**，vocab 248078，加/解码往返一致 |
| 4 | 内网 DeepSeek 可用 | ✅ | 36 个模型；`deepseek-v4-pro` 正常返回并带 usage |
| 5 | 绕过公网阻断 | ✅ | 权重走魔搭 4.77MB/s |
| 6 | 下载 8.47GB 权重 | ✅ | 14/14 文件大小与魔搭清单一致，9,097,469,720 字节 |
| 7 | 依赖链可行性验证 | ✅ | 全部 pin 在清华镜像存在 |
| 8 | 数据集预取 + 离线化 | ✅ | gsm8k sha256 == HF lfs.oid；6 个数据集离线加载全通过 |
| 9 | **Windows 原生 venv 环境** | ✅ | torch 2.7.1+cu118，`cuda.is_available()=True` |
| 10 | **模型 GPU 加载** | ✅ | `AutoModelForImageTextToText`，4.54B，显存占用 9.8GB |
| 11 | **自建 OpenAI 兼容服务** | ✅ | `/v1/completions` 支持 vLLM 专有参数，`stop_reason` 正确 |
| 12 | **触发 offload** | ✅ | AIME 难题 39 tok 后吐出 `<|llm_offload|>` |
| 13 | **接力内网 DeepSeek** | ✅ | `[llm] relay 5/5 via deepseek-v4-pro` |
| 14 | **发现并修复打分 bug** | ✅ | `math_verify` 恒判 False → 修 `boxed_socre.py` |
| 15 | **端到端冒烟通过** | ✅ | gsm8k 100%（5/5），offload 100% |
| 16 | 一键脚本 | ✅ | `06-run-windows-native.sh` 实测退出码 0 |

## 🔴 修复的上游 bug：Windows 上准确率恒为 0

**现象**：接力后的输出里明明有正确的 `\boxed{18}`，却判 0 分；gsm8k **0/3**。

**定位过程**：

```
extract_pred_boxed(resp)        → '18'      ✅
extract_ground_truth_boxed('18')→ '18'      ✅
_parse_boxed_content('18')      → [18,'18'] ✅
verify_boxed('18','18')         → False     ❌
math_verify.verify([18,'18'],[18,'18']) → False  ❌  ← 连相同整数都判错
```

**根因**：`math_verify.verify()` 默认 `timeout_seconds=5`，其超时保护依赖
`multiprocessing`/`signal` 子进程；在 Windows 上该子进程 spawn 失败
（`OSError: [WinError 6] 句柄无效`），于是 **`verify()` 无条件返回 False**。

**验证**：

| 调用 | 结果 |
|---|---|
| `verify([18,'18'],[18,'18'])` 默认 | ❌ False |
| `verify([18,'18'],[18,'18'], timeout_seconds=None)` | ✅ True |
| `verify([18,'18'],[19,'19'], timeout_seconds=None)` | ✅ False（错误答案正确拒绝） |
| `verify(1/2, 0.5, timeout_seconds=None)` | ✅ True（等价分数识别正常） |

**修复**（`evaluation/evaluation_math/boxed_socre.py`）：`verify_boxed()` 补传
`timeout_seconds=None`（并保留 `TypeError` 回退以兼容旧版）。
原作者只对 `math_verify.parse` 传了 `parsing_timeout=None`，**漏了 `verify`**。

**效果**：同一批已保存结果复评 → gsm8k **0/3 → 3/3**。

## 环境与版本（Windows 原生 venv）

```
setup/.venv-win/
  torch        2.7.1+cu118   ← CUDA 11.8 构建；驱动 537.70 直接可用，无需升级
  transformers 5.5.3         ← 提供 Qwen3_5 架构支持
  datasets     5.0.1         ← 与 setup/hf_home 缓存指纹匹配
  accelerate   1.15.0
  math-verify / mathruler / pylatexenc / requests / tqdm / pandas / openai / httpx
```

**关键决策依据**：`transformers 5.5.3` 只要求 `torch>=2.4`，因此**不需要**上游 pin 的
torch 2.11.0（那是 CUDA 13 构建，才要求驱动 ≥580.65.06）。降级到能配 537.70 驱动的
cu118 构建，是绕开管理员权限的关键。

## 模型架构要点

| 项 | 值 |
|---|---|
| architectures | `Qwen3_5ForConditionalGeneration`（VL，含 vision tower） |
| model_type | `qwen3_5` |
| 参数 | 4.54B（文本 + 视觉） |
| 文本侧 | 32 层 / hidden 2560 / 16 heads / 4 KV heads / head_dim 256 |
| 注意力 | 混合结构，含 `linear_attn`（A_log / conv1d） |
| 权重键名 | `model.language_model.*` + `model.visual.*` |
| 加载方式 | `AutoModelForImageTextToText`（`AutoModelForCausalLM` 键名不匹配） |
| 显存 | 12GB 卡上占用约 9.8GB，余 2.2GB |

⚠️ `The fast path is not available ... flash-linear-attention / causal-conv1d` ——
线性注意力层走了 torch 回退实现，这是速度只有 ~9 tok/s 的原因。
装了这两个库会快很多，但都需要 CUDA 编译工具链，Windows 上成本高。

## 已知限制

1. **推理速度 ~9-10 tok/s**（线性注意力未走 fast path）。全量评测（如 gsm8k 1319 题 × 最多 8192 tok）在本机不现实，建议：
   - 小样本冒烟用 `06-run-windows-native.sh <N>`；
   - 需全量请走 WSL2 + vLLM（路径 A）。
2. **共享 token 预算很关键**：`max_tokens` 是小模型 + 大模型的总预算。
   AIME 用 700 会不够（接力方算到一半被截断，写不出 `\boxed{}` → 判错）。
   建议 AIME/Olympiad 类难题用 ≥4096。
3. **`math_verify` 的 spawn 报错日志**：即使修复后仍会打印
   `OSError: [WinError 6]`，来自其他未关超时的调用路径，**不影响结果**（已被捕获）。
4. **内网限制**：`ai-api.bj.tkoffice.cn` 仅公司网络/VPN 可达。
5. **模型未改用新控制 token**：仓库已支持 `<|llm_offload|>N<|/llm_offload|>`，
   但 `evaluation_math` 的 relay 仍用旧的单 token 形式（保持了训练时行为）。

## 产出文件

```
D:\hecan\PyroDash\
├── evaluation/evaluation_math/boxed_socre.py   # ★ 修复 math_verify 超时 bug
├── models/PyroDash-4B-GRPO-Lambda-0.05/        # 8.47GB，14/14 已校验
└── setup/
    ├── README.md                     # 执行手册
    ├── STATUS.md                     # 本文件
    ├── 06-run-windows-native.sh      # ★ 一键：起服务 + 跑评测（推荐入口）
    ├── serve_small.py                # ★ OpenAI 兼容服务（替代 vLLM）
    ├── smoke_offload.py              # ★ 小样本冒烟（复用真实 relay 全链路）
    ├── requirements-math.txt         # WSL/vLLM 路径的精确依赖 pin
    ├── 01-enable-wsl.ps1             # 路径 A：启用 WSL2（需管理员）
    ├── 01b-move-wsl-to-d.ps1         # 路径 A：发行版迁 D 盘（需管理员）
    ├── 02-setup-wsl.sh               # 路径 A：装 vLLM（含驱动硬校验门禁）
    ├── 03-download-model.sh          # 模型下载+校验（已完成）
    ├── 04-run-math-eval.sh           # 路径 A：vLLM + 内网 DeepSeek 评测
    ├── 05-prepare-datasets.sh        # 离线数据集缓存 + 自检
    ├── download_model.py             # 双源断点续传下载器
    ├── prefetch_datasets.py          # 数据集预取
    ├── hf_home/                      # 离线 HF 缓存（9.8MB，实测离线可用）
    ├── datasets/                     # 预取原始数据集 + manifest.json
    ├── driver/                       # 预下载 616.92 驱动（路径 A 用）
    └── .venv-win/                    # Windows 原生 venv（约 3GB）
```

## 复现步骤（Windows 原生，从零）

```bash
# 1. 建 venv 并装依赖（约 3GB 下载）
python -m venv setup/.venv-win
setup/.venv-win/Scripts/python.exe -m pip install torch==2.7.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118 \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
setup/.venv-win/Scripts/python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    transformers==5.5.3 datasets==5.0.1 accelerate math-verify mathruler \
    pylatexenc requests tqdm pandas openai httpx huggingface-hub

# 2. 准备离线数据集（可跳过，会自动回退到镜像）
bash setup/05-prepare-datasets.sh

# 3. 一键跑评测
bash setup/06-run-windows-native.sh 5          # gsm8k 前 5 题
bash setup/06-run-windows-native.sh --stop     # 用完停服务
```

## 本地校验记录

| 检查项 | 结果 |
|---|---|
| `bash -n` 全部 shell 脚本 | ✅ 通过 |
| PowerShell Parser 校验 ps1 | ✅ 加 UTF-8 BOM 后通过 |
| `py_compile` 全部 python | ✅ 通过 |
| 模型 14/14 文件大小 | ✅ 与魔搭清单一致 |
| gsm8k sha256 vs HF lfs.oid | ✅ 逐字节一致 |
| 驱动文件 vs 官方 | ✅ 990,853,104 字节，差异 0.00 |
| 6 个数据集离线加载 | ✅ 全通过 |
| venv `torch.cuda.is_available()` | ✅ True（RTX 3060，驱动 537.70） |
| 模型加载 | ✅ `AutoModelForImageTextToText`，4.54B |
| offload token 触发 | ✅ AIME 难题 39 tok 后触发 |
| 端到端评测 | ✅ gsm8k 100%（5/5），offload 100%，退出码 0 |
| 系统 Python torch 已还原 | ✅ 2.7.1+cu118 |
