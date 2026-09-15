# PyroDash 本地部署 —— 执行手册

> 方案全文见 `D:\hecan\business2\pyrodash-local-deploy-plan.md`
> 目标：本地 RTX 3060 12GB 跑 PyroDash-4B（vLLM）+ 远端 tkoffice 内网 DeepSeek 协同推理
> 当前进度见 `STATUS.md`

## ✅ 推荐路径：Windows 原生（已验证跑通，无需管理员）

**先看这里。** 官方文档走 WSL2 + vLLM，但那条路需要管理员权限 + 升级 NVIDIA 驱动。
已实测出一条**不需要管理员、也不需要升级驱动**的等效路径，并已跑通全链路：

```
Dataset:  gsm8k
Accuracy: 100.00% (5/5)
Offload:  5/5 (100.0%)
```

原理：用 `transformers` + 自建 OpenAI 兼容服务（`serve_small.py`）替代 vLLM，
配 `torch 2.7.1+cu118` —— 当前驱动 537.70（CUDA 12.2）即可直接用 GPU 跑 4B 模型。

```bash
bash setup/06-run-windows-native.sh 5             # gsm8k 前 5 题冒烟
bash setup/06-run-windows-native.sh --compare 50  # 三组对照实验（§4 验证清单核心）
bash setup/06-run-windows-native.sh --stop         # 用完停服务
bash setup/06-run-windows-native.sh --full         # 全量（本机很慢，见下）

# 换 λ 对照 checkpoint
PYRODASH_MODEL=models/PyroDash-4B-GRPO-Lambda-0.6 \
  bash setup/06-run-windows-native.sh --compare 50 --tag l06
```

### 实测结论（GSM8K 前 50 题，同一批题，`deepseek-v4-pro`）

| λ | 臂 | 准确率 | offload 率 | 远端 token | 相对成本 |
|---|---|---|---|---|---|
| λ=0.05 | 纯小模型下限 | 4.00% | 96.0% | 0 | 0.00x |
| λ=0.05 | **PyroDash** | **98.00%** | 96.0% | 41,916 | **1.26x** |
| λ=0.05 | 纯大模型上限 | 96.00% | — | 33,366 | 1.00x |
| λ=0.6 | 纯小模型下限 | 90.00% | 2.0% | 0 | 0.00x |
| λ=0.6 | **PyroDash** | **90.00%** | **0.0%** | **0** | **0.00x** |
| λ=0.6 | 纯大模型上限 | 96.00% | — | 33,063 | 1.00x |

**要点**：

- λ 确实控制 offload 率：**96% → 0%**，远端成本跟着从 41,916 → **0**。
- λ=0.05 在 GSM8K 上反而比全量直调大模型**更贵**（1.26x）：offload 率太高，
  且每次接力都要把已产出的推理链当 prompt 前缀再发一遍，prompt token 重复消耗。
- λ=0.6 在 GSM8K 上性价比最高：远端成本 0，准确率 90%（比纯大模型低 6 个点）。
- λ=0.05 的 PyroDash 拿到 **98%**，反超纯大模型上限的 96%——本地已算对的部分
  喂给大模型，反而减少它走偏。**offload 不只是省钱，也可能提升上限**。

详细分析与复现命令见 `STATUS.md` 的「对照实验结果」。


| | 路径 A：WSL2 + vLLM | **路径 B：Windows 原生** |
|---|---|---|
| 需要管理员 | ❌ 需要 | ✅ **不需要** |
| 需要升级驱动 | ❌ 需 ≥580 | ✅ **不需要** |
| 推理速度 | 快（vLLM 优化） | ~9-10 tok/s（线性注意力走 torch 回退） |
| 适用场景 | 全量评测 | 冒烟 / 小样本验证 |

> ⚠️ **上线前务必先看 `STATUS.md` 记录的上游 `math_verify` bug**：
> 若不修 `evaluation/evaluation_math/boxed_socre.py`，Windows 上所有准确率会
> **恒为 0**（本仓库已修）。原因：`math_verify.verify()` 默认的 5s 超时保护依赖
> 子进程，在 Windows 上必然失败，导致 `verify()` 无条件返回 False。

> ❗ Windows 原生路径下**共享 token 预算**（`--max-tokens`）很关键：它是小模型 +
> 大模型的**总**预算。GSM8K 用 2500 够，AIME/Olympiad 建议 ≥4096，否则接力方
> 算到一半被截断、写不出 `\boxed{}`，会被判为 `no_pred_boxed`。

以下章节是**路径 A（WSL2 + vLLM）**的完整说明——只有在你能拿到管理员权限、
且需要全量高吞吐评测时才需要走它。
## 环境事实（已实测）

| 项目 | 实测值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3060，12288 MiB |
| 驱动 | **537.70 → CUDA 12.2**（路径 A/WSL 必须升到 ≥580；路径 B/Windows 原生**无需升级**） |
| WSL2 | 未安装，需管理员提权安装 |
| 磁盘 | C: 可用 241GB，D: 可用 1368GB |
| Windows Python | 3.12.6；git 2.47.0 |

## 🔴 风险 1 已定量核实：驱动必须升到 580+

从 PyPI 官方元数据查到 `torch 2.11.0` 的依赖是 **CUDA 13** 构建：

```
nvidia-cudnn-cu13==9.19.0.56
nvidia-cusparselt-cu13==0.8.0
nvidia-nccl-cu13==2.28.9
nvidia-nvshmem-cu13==3.4.5
```

CUDA Toolkit 13.0 官方 Release Notes（Table 3）：**Linux x86_64 驱动 >= 580.65.06**。
WSL2 使用 Windows 驱动 → **Windows 驱动必须 >= 580**。当前 537.70 不满足。

已定位并预下载适配 RTX 3060 的驱动（NVIDIA 官方查询接口）：

| 项 | 值 |
|---|---|
| 版本 | **616.92**（GeForce Game Ready，WHQL） |
| 发布 | 2026-09-09 |
| 大小 | 990,853,104 字节（=官方 990.85 MB，差异 0.00） |
| 文件 | `setup\driver\nvidia-616.92-win10-win11-64bit-dch-whql.exe` |
| 校验 | PE 头有效；URL 为 `desktop-...-international-dch` 通用桌面包，含 Ampere(RTX 3060) |

## 网络实测结论（重要：连通性是**间歇性**的）

| 目标 | 结果 |
|---|---|
| `www.modelscope.cn` | ✅ 可达，4.77 MB/s，支持 Range 续传 |
| `pypi.tuna.tsinghua.edu.cn` | ✅ 可达（pip 用，版本齐全） |
| `pypi.org` | ✅ 可达 |
| `openaipublic.blob.core.windows.net` | ✅ 可达（math 数据集 CSV） |
| `www.nvidia.cn` / `us.download.nvidia.com` | ✅ 可达（驱动可下） |
| `ai-api.bj.tkoffice.cn` | ✅ 可达（内网大模型，无需公网） |
| `huggingface.co` | ⚠️ **间歇可达**（有时 200，有时超时） |
| `hf-mirror.com` | ⚠️ **间歇可达**（注意需 `-L`，无 `-L` 返回 308 空体） |
| `github.com` / `gitee.com` / `ghproxy.net` | ❌ 阻断 |

**策略**：模型权重走魔搭（已验证与 HF 逐字节一致）；数据集已**预取 + 离线化**（见下）；pip 走清华源。

## 依赖可行性（已核实，无需等装完才发现问题）

| 包 | 上游 requirement | 镜像实际可用 | 结论 |
|---|---|---|---|
| vllm | `>=0.25.1` | 0.25.1 存在；最新 0.29.0 | ✅ |
| torch | `==2.11.0` | 存在（cp312 manylinux_2_28） | ✅ |
| torchaudio / torchvision | `==2.11.0` / `==0.26.0` | 存在 | ✅ |
| transformers | `>=5.5.3` | 5.5.3 存在；最新 5.17.0 | ✅ |

**关键**：`vllm 0.25.1` 声明的依赖是 `torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 transformers>=5.5.3`
—— 与上游 `requirements.txt` 的 pin **完全一致**。
因此 `requirements-math.txt` 把 vllm **固定为 0.25.1**：若写 `>=0.25.1`，pip 会选 0.29.0，而它要求 `torch==2.13.0`，与 pin 冲突。

## 数据集离线化（已实测通过）

`datasets_loader.py` 的数据来源：

| 数据集 | 来源 |
|---|---|
| math | `openaipublic.blob.core.windows.net/simple-evals/math_500_test.csv` |
| gsm8k | HF `openai/gsm8k`（config `main`，test） |
| amc | HF `zwhe99/amc23`（test） |
| minerva | HF `zwhe99/simplerl-minerva-math`（test） |
| olympiad | HF `zwhe99/simplerl-OlympiadBench`（test） |
| aime2024 | HF `HuggingFaceH4/aime_2024`（train） |
| aime2025 | HF `yentinglin/aime_2025`（config `default`，train） |

**踩过的坑**：只把原始 parquet + `refs/main` 铺进 hub 缓存**不够**——`datasets` 还需要 hub 的 API 元数据，
`HF_HUB_OFFLINE=1` 时会报 `ConnectionError: Couldn't reach ... (OfflineModeIsEnabled)`。
可行做法是让 `datasets` **在线跑一次**产出完整 `$HF_HOME`，之后即可完全离线复用。

`setup/hf_home/`（9.8MB）就是这样一份缓存，已实测 **6 个数据集全部离线加载成功**：

```
openai/gsm8k                   1319 条  ['question','answer']
zwhe99/amc23                     40 条
zwhe99/simplerl-minerva-math    272 条
zwhe99/simplerl-OlympiadBench   675 条
HuggingFaceH4/aime_2024          30 条
yentinglin/aime_2025             30 条
```

gsm8k 预取文件经 **sha256 与 HF lfs.oid 逐字节比对一致**（`ee7b8da9…4f59`）。
该缓存由 `datasets==5.0.1` 生成，故 `requirements-math.txt` 固定了该版本以保证指纹匹配。

## 执行顺序

```bash
# ① Windows 管理员：安装驱动 616.92（不装后面必然失败）
D:\hecan\PyroDash\setup\driver\nvidia-616.92-win10-win11-64bit-dch-whql.exe

# ② Windows 管理员 PowerShell：启用 WSL2 + 装 Ubuntu-24.04
powershell -ExecutionPolicy Bypass -File D:\hecan\PyroDash\setup\01-enable-wsl.ps1

# ③ 重启 Windows（驱动与 WSL 都需要）

# ④ 进入 Ubuntu（首次需创建 Linux 用户名/密码）
wsl -d Ubuntu-24.04

# ⑤ WSL：装依赖 + 建虚拟环境（约 8-10GB，耗时较长；内含驱动门禁校验）
bash /mnt/d/hecan/PyroDash/setup/02-setup-wsl.sh

# ⑥ WSL：安装预置离线数据集缓存并自检
bash /mnt/d/hecan/PyroDash/setup/05-prepare-datasets.sh

# ⑦ 冒烟评测（模型已提前下好校验完，03 步可跳过）
bash /mnt/d/hecan/PyroDash/setup/04-run-math-eval.sh gsm8k

# ⑧ 全量数学评测
bash /mnt/d/hecan/PyroDash/setup/04-run-math-eval.sh gsm8k math amc minerva olympiad aime2024 aime2025
```

## 脚本说明

### 路径 B（Windows 原生，推荐）—— 无需管理员

| 脚本 | 运行位置 | 作用 |
|---|---|---|
| **`06-run-windows-native.sh`** | Windows / Git Bash | ★ 一键：自检 + 起小模型服务 + 跑评测（`--stop` 停服务，`--full` 跑全量，`--compare N` 跑三组对照） |
| **`serve_small.py`** | Windows | ★ OpenAI 兼容服务，替代 vLLM。支持 `include_stop_str_in_output` / `skip_special_tokens` / `stop` 等 vLLM 专有语义 |
| **`smoke_offload.py`** | Windows | ★ 小样本冒烟：只裁数据集条数，其余复用上游真实 relay 全链路 |
| **`compare_arms.py`** | Windows | ★ 三组对照实验（纯小模型下限 / PyroDash / 纯大模型上限），输出 accuracy + offload 率 + 远端 token |
| `download_model.py` | 任意 | 跨平台下载器（ModelScope/HF 双源、断点续传、大小+token 校验） |
| `prefetch_datasets.py` | Windows | 预取 7 个数据集原始文件 + 记录 sha |

环境变量（都可覆盖）：`PYRODASH_MODEL`（切 checkpoint）、`SMALL_TIMEOUT`（小模型单请求超时秒数，默认 3600）、`LLM_BASE_URL` / `LLM_MODEL`。

### ⚠️ API 密钥怎么给（不要写进仓库）

脚本**不再内置密钥默认值**。按优先级：

1. `export LLM_API_KEY=sk-xxx`（推荐，一次性）
2. 写入 `setup/.llm_key`（已在 `.gitignore` 忽略，只放一行密钥）——最省事，脚本自动读
3. 从 `aiConfig/claude/enableClaudeChina.py` 复制

密钥缺失时脚本会直接报错退出并提示，不会静默拿空 key 去请求。

```bash
# 方式 2 的最快建法
printf '%s\n' 'sk-你的密钥' > setup/.llm_key
```

### 路径 A（WSL2 + vLLM）—— 需管理员

| 脚本 | 运行位置 | 作用 |
|---|---|---|
| `01-enable-wsl.ps1` | Windows **管理员** | 启用 VirtualMachinePlatform + WSL，装 Ubuntu-24.04 |
| `01b-move-wsl-to-d.ps1` | Windows **管理员** | 可选：把发行版迁到 D 盘 |
| `02-setup-wsl.sh` | WSL | **驱动硬校验门禁** + apt 依赖 + venv + vLLM/torch/数学依赖 |
| `03-download-model.sh` | WSL 或 Git Bash | 调用 `download_model.py`（模型已下好，可跳过） |
| `04-run-math-eval.sh` | WSL | 起本地 vLLM(16K) + 内网 DeepSeek，跑数学评测 |
| `05-prepare-datasets.sh` | WSL | 安装离线数据集缓存 + 离线加载自检 |
| `requirements-math.txt` | — | 路径 A 的精确依赖清单（vllm==0.25.1 + torch==2.11.0，已剔除 coding 依赖） |

> 路径 B 的依赖装法见 `STATUS.md` 的「复现步骤」（`torch==2.7.1+cu118` + `transformers==5.5.3`）。

> ⚠️ PowerShell 脚本已写为 **UTF-8 with BOM**。若手动编辑后丢失 BOM，
> Windows PowerShell 5.1 会按 GBK 解码中文注释导致语法错误（已踩过此坑）。

## 12GB 显存的关键调参

RTX 3060 只有 12288 MiB，官方脚本的 `--max-model-len 40960` 会 OOM。04 脚本已改为：

- `--max-model-len 16384`（权重 8.47GB + KV ≈ 2GB，余量安全）
- `--gpu-memory-utilization 0.90`

想更省显存可用环境变量覆盖：`MAX_MODEL_LEN=8192 GPU_UTIL=0.85 bash .../04-run-math-eval.sh`

## 覆盖环境变量

`04-run-math-eval.sh` 支持：

```bash
MODEL=/path/to/model        # 换 SFT / Lambda-0.6 模型做对比
LLM_MODEL=deepseek-v4-flash # 换更省的大模型
LLM_BASE_URL=... LLM_API_KEY=...      # 换网关
MAX_MODEL_LEN=8192 GPU_UTIL=0.85 PORT=8001 OUT_DIR=...
HF_HOME=/path/to/hf_home              # 换数据集缓存位置
```

`03-download-model.sh` / `download_model.py` 支持：

```bash
SOURCE=modelscope           # 默认；SOURCE=hf 强制走 HuggingFace
DEST=/other/dir             # 自定义目标目录
```

## 已知风险

1. **🔴 驱动过旧（最可能卡住）**：当前 537.70 仅支持 CUDA 12.2，而 torch 2.11.0 是 CUDA 13 构建，
   必须 >= 580（CUDA 13.0 GA 官方要求 >= 580.65.06）。已预下载 616.92。
   02 脚本会先做驱动门禁校验并明确报错，不会静默失败。
2. **公网间歇性阻断**：HF/GitHub 时通时断。模型走魔搭；数据集已离线化；pip 走清华源。
   若后续要跑 SWE-Bench coding 评测，docker 镜像与部分依赖可能拉不动。
3. **上游依赖很新**：`vllm 0.25.1` / `transformers>=5.5.3`（qwen3_5 新架构必须）。
   只装 math 链路子集；SWE-Bench 依赖（swebench / mini-swe-agent / docker）未装。
4. **内网 key 在脚本里**：`04-run-math-eval.sh` 含 tkoffice 内网 key，仅限私有环境。
5. **评测中途断网**：数据集已离线；但远端大模型调用需要内网可达（公司网络/VPN）。
6. 评测结束脚本会自动 kill vLLM；异常退出时可 `pkill -f "vllm serve"`。
