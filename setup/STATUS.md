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
| 17 | **下载 λ=0.6 对照模型** | ✅ | 14 文件 / 9,097,469,695 字节，下载器自校验通过 |
| 18 | **修复长生成请求超时** | ✅ | 服务端串行 → `SMALL_TIMEOUT` 可调；修复后 0 超时跑完 |
| 19 | **§4 验证清单③：三组对照** | ✅ | GSM8K 50 题 × 3 臂，见下方「对照实验结果」 |
| 20 | **§4 验证清单④：λ=0.6 对比** | ✅ | offload 96% → 0%，远端成本 41,916 → 0 |

## 🧪 对照实验结果（计划 §4 验证清单 ③④）

规模：GSM8K 前 50 题（3 臂同一批题），远端 `deepseek-v4-pro`，`--max-tokens 4096`。
命令：`bash setup/06-run-windows-native.sh --compare 50 --tag l05`（λ=0.6 加 `PYRODASH_MODEL=...`）

| λ | 臂 | 准确率 | offload 率 | 远端 token | 相对成本 |
|---|---|---|---|---|---|
| λ=0.05 | 纯小模型下限 | 4.00% (2/50) | 96.0% | 0 | 0.00x |
| λ=0.05 | **PyroDash 主链路** | **98.00% (49/50)** | 96.0% | 41,916 | **1.26x** |
| λ=0.05 | 纯大模型上限 | 96.00% (48/50) | — | 33,366 | 1.00x |
| λ=0.6 | 纯小模型下限 | 90.00% (45/50) | 2.0% | 0 | 0.00x |
| λ=0.6 | **PyroDash 主链路** | **90.00% (45/50)** | **0.0%** | **0** | **0.00x** |
| λ=0.6 | 纯大模型上限 | 96.00% (48/50) | — | 33,063 | 1.00x |

> 相对成本 = 该臂远端 token / 纯大模型上限的 token（内网网关无实际计费，token 量即对网关的压力）

### 结论

1. **λ 的效果得到确认（§4 第 4 项）**：λ=0.6 把 offload 率从 96% 压到 0~2%，
   远端 token 从 41,916 降到 **0**。远端成本确实随 λ 增大而下降。

2. **λ=0.05 在 GSM8K 上反而比全量直调大模型更贵（1.26x）**。
   原因：offload 率高达 96%，而每次接力都要把已产出的推理链作为 prompt 前缀
   一起发给大模型（`<part_think>` 机制），prompt token 大量重复消耗。
   也就是说——**便宜与否不只看 offload 率，还看「接力时带过去多少上下文」**。

3. **λ=0.6 在 GSM8K 上性价比最高**：准确率 90%（与纯小模型相同，因为根本没求助），
   远端成本为 **0**，即完全用本地算力；代价是比纯大模型上限低 6 个百分点。

4. **小模型的部分推理链对难题有帮助**：λ=0.05 的 PyroDash 拿到 98%，
   反而**高于**纯大模型上限的 96%。说明 offload 不只是省钱，还可能提升上限
   （把本地已经算对的部分喂给大模型，减少它走偏的机会）。

5. 纯小模型下限在两个 λ 下差异巨大（4% vs 90%）：λ=0.05 的模型几乎不肯自己算，
   λ=0.6 的模型基本自己算完。**这直接影响部署选型**：
   - 想省钱 → λ=0.6（本地扁住，极少调远端）
   - 想冲准确率 → λ=0.05（几乎每题都问大模型，但注意 token 反而更多）

### 复现命令

```bash
# λ=0.05
bash setup/06-run-windows-native.sh --compare 50 --tag l05

# λ=0.6（需先停服务，脚本会用新 checkpoint 重启）
bash setup/06-run-windows-native.sh --stop
PYRODASH_MODEL=models/PyroDash-4B-GRPO-Lambda-0.6 \
  bash setup/06-run-windows-native.sh --compare 50 --tag l06
```

### ⚠️ 实测到的坑：小模型请求超时

`serve_small.py` 是 `ThreadingHTTPServer` + 全局锁，**GPU 实际串行生成**。
`math_eval.call_small_completion` 原本硬编码 `timeout=600.0`，于是：
λ=0.6（小模型自己推理多 → 单题生成长）× 50 题并发提交 → 后面的请求排队，
排队时间计入超时 → `requests.exceptions.ReadTimeout`，**实测跑到 27/50 挂掉**。

修法（两处）：
- `math_eval.py`：超时改为 `SMALL_TIMEOUT` 环境变量可覆盖（默认仍 600s）
- `compare_arms.py`：新增 `--small-workers`（默认 **1**，既然服务端串行就不必并发）
- `06-run-windows-native.sh`：默认导出 `SMALL_TIMEOUT=3600`

修复后 λ=0.6 全程 **0 次超时**跑完。

## 🔴 修复的上游 bug：Windows 上准确率恒为 0
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
| 一键脚本热启动跑评测 | ✅ gsm8k 100%（2/2），参数透传正常 |
| **一键脚本冷启动跑评测** | ✅ 自动拉起服务（等待 25s）→ gsm8k 100%（3/3），全程 1m30s，退出码 0 |

## 冷启动耗时实测

```
[stop]  停止小模型服务（模拟冷启动）
[serve] 启动小模型服务（加载 8.45GB 权重）...
[serve] ✅ 就绪（等待 25 秒）
[run]   smoke_offload.py --dataset gsm8k --limit 3 --max-tokens 2500
  Dataset:  gsm8k
  Accuracy: 100.00% (3/3)
  Offload:  3/3 (100.0%)
总耗时: 1m29.7s   退出码: 0
```

## 提交记录（本地，未能推送）

上游 `github.com/PyroMind-Dynamics/PyroDash` 是**第三方仓库**，推送返回 403：

```
remote: Permission to PyroMind-Dynamics/PyroDash.git denied to CharlesHo110.
fatal: unable to access ...: The requested URL returned error: 403
```

因此以下 3 个提交保留在本地 `main`：

| commit | 内容 |
|---|---|
| `42c97c2` | 新增 Windows 原生部署路径（serve_small.py / smoke_offload.py / 06 脚本）+ 修复 math_verify 打分 bug |
| `98d98d6` | 新增 .gitattributes 固定行尾（防 shell 脚本被检出成 CRLF） |
| `08e14dd` | 修复 06 脚本：数字参数会吞掉后续选项 |

> 如需备份，可另加自己的 remote：`git remote add mine <你的仓库>` 后 `git push mine main`。
