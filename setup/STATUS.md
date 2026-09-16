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
| 21 | **本机可用版（`local_relay/`）** | ✅ | 路由 7/7 正确；轻任务本机 0.4–2.5s、大模型 0 token |
| 22 | **llama.cpp 落地（免编译）** | ✅ | `llama-server.exe` build 10982，`/apply-template` 实测可用 |
| 23 | **Qwen3-4B GGUF（2.33 GB）** | ✅ | 走魔搭 5.45 MB/s，437 秒下完 |
| 24 | **交接机制不依赖训练** | ✅ | 实测通用模型不主动交接，改用「预算内没答完」触发 |
| 25 | **带 tools 请求透传** | ✅ | `route=handoff:tools`，3.4s，`tool_calls` 正确 |
| 26 | **接入 pi** | ✅ | `~/.pi/agent/models.json` 新增 `pyrodash-local`，原 4 个 provider 逐字节未变 |

## 🚀 新增：本机可用版（`local_relay/`）

上面第 6–20 项做的是**数学评测**（验证论文的机制与 λ 对交接率的影响）。
本节做的是另一件事：**把同一套机制做成日常真的能用的东西** —— 一个 OpenAI 兼容端点，
接进 pi 或任何 OpenAI 客户端，本机小模型先接活、搞不定就交给远端大模型接着写。

```
pi / 任何 OpenAI 客户端
        │  http://127.0.0.1:8010/v1
        ▼
  relay_server.py   ── 注入交接协议 / 判定路由 / 共享预算
        ├── route=small          → llama-server :8080（Qwen3-4B Q4_K_M）
        └── route=handoff:*      → deepseek-v4-pro（收到半截推理，接着写）
```

### 实测结果（RTX 3060，Qwen3-4B Q4_K_M @ llama.cpp CUDA）

| 用例 | 期望 | 实际路由 | 小模型 | 大模型 | 总耗时 |
|---|---|---|---|---|---|
| 翻译 | small | ✅ `small` | 17 tok / 0.55 s | 0 | 0.59 s |
| 算术（1+1） | small | ✅ `small` | 2 tok / 0.04 s | 0 | 0.08 s |
| 总结 | small | ✅ `small` | 19 tok / 0.24 s | 0 | 0.28 s |
| 格式转换 | small | ✅ `small` | 21 tok / 0.26 s | 0 | 0.30 s |
| 多步推理 | handoff | ✅ `handoff:limit` | 512 tok / 5.6 s | 1536 tok / 28.7 s | 34.4 s |
| 长链条代码 | handoff | ✅ `handoff:limit` | 512 tok / 6.0 s | 1536 tok / 28.1 s | 34.0 s |
| 领域知识 | handoff | ✅ `handoff:limit` | 512 tok / 5.9 s | 1536 tok / 28.5 s | 34.5 s |
| 带 `tools` 的请求 | handoff | ✅ `handoff:tools` | 0（跳过） | 77 tok | 3.4 s |

**路由判定 7/7 全部符合预期**；大模型 token 占比 74.3%（总 6203 里 4608 给大模型）。

小模型侧原生吞吐：生成 **92 tok/s**、prompt 处理 **912 tok/s**。

#### CPU 回退 vs GPU（同一批用例）

| | CPU 回退（缺 cublas） | GPU（补齐后） | 提升 |
|---|---|---|---|
| 小模型原生生成 | 10.2 tok/s | **92 tok/s** | 9× |
| 轻任务（本机回答） | 0.4 – 2.5 s | **0.04 – 0.55 s** | 约 9× |
| 小模型平均延迟 | 47.98 s | **2.66 s** | 18× |
| 困难任务的小模型段 | 53 s | **5.6 s** | 9.5× |

> GPU 不工作时整套仍然能跑通，只是困难任务里「小模型先写 512 token 半截推理」这一段要等
> 53 秒，体验就崩了。补齐 CUDA 运行时 DLL 后降到 5.6 秒，整个设计才真正可用。

### ⚠️ 重要发现：通用模型不会自发交接

给 Qwen3-4B-Instruct-2507 灌完整套「遇到不会的就输出 `<|llm_offload|>` 然后停止」的协议
提示词，实测它**一次都不肯交**（`handoff:tag` 恒为 0）——它会硬着头皮往下写，直到被 token
上限截断。**这正是 PyroDash 要花力气做 GRPO 训练的原因：「知道自己不会」需要训练。**

因此本 relay 换了一条不依赖模型自知之明的路：

1. `stop_type == "word"` —— 模型主动吐交接标记（训练过的模型才常见，属优化项）；
2. `stop_type == "limit"` —— **本机小模型没在 `SMALL_MAX_TOKENS`（默认 512）内答完**。
   这条不需要任何训练，天然就是「本机干不完」的信号，而且它已经写出的半截推理正好
   就是大模型的起手式。

于是 `SMALL_MAX_TOKENS` 成了唯一的、也是最有效的旋钮：调小→交接更积极更省钱，
调大→本机承担更多、延迟更低。

### 为什么小模型侧走 `llama.cpp 原生 /completion`

因为**只有原生端点能无歧义区分「答完了」和「撞到交接标记」**：

```jsonc
// POST /completion  →  stop_type ∈ {none, eos, limit, word}
{"content": "...", "stop_type": "word", "stopping_word": "<|llm_offload|>"}
```

OpenAI 兼容端点两种情况都只给 `finish_reason: "stop"`；vLLM 那套
`include_stop_str_in_output` 在 llama.cpp 里**不存在**（实测 `/completion` 与
`/v1/chat/completions` 都无此参数），但 `stop_type`/`stopping_word` 比它更干净 ——
标记本来就不会混进 `content`。

### 新增文件

| 文件 | 作用 |
|---|---|
| `local_relay/relay_server.py` | 中转服务本体（纯标准库 HTTP，无需 fastapi/uvicorn） |
| `local_relay/offload_protocol.py` | 交接协议提示词与 `inject_protocol()` |
| `local_relay/run.sh` | 一键启停（llama-server + relay，含就绪等待与自测） |
| `local_relay/test_relay.py` | 端到端测试（简单/困难两组 + `--ask` 自由提问） |
| `local_relay/selftest_offline.py` | 纯离线自检，不联网不占显存 |
| `local_relay/README.md` | 使用说明、调参、排错 |
| `setup/07-install-llama.sh` | 下载 llama.cpp Windows CUDA 预编译包 |
| `setup/07b-fetch-cudart.py` | 走清华 PyPI 取 CUDA 运行时 DLL |
| `setup/08-download-gguf.py` | 从 modelscope 下载 GGUF |

### 复现

```bash
bash   setup/07-install-llama.sh      # llama.cpp（242 MB）
python setup/07b-fetch-cudart.py      # CUDA 运行时 DLL
python setup/08-download-gguf.py      # Qwen3-4B Q4_K_M（2.33 GB）

bash local_relay/run.sh               # 起服务，打印 Base URL / API Key / Model
python local_relay/test_relay.py --suite all
python local_relay/selftest_offline.py
bash local_relay/run.sh --stop        # 停
```

### ⚠️ 踩到的两个坑（已解决）

**1. GitHub 是唯一慢源，但 PyPI 镜像上有同一批 DLL。**
`cudart-llama-bin-win-cuda-12.4-x64.zip` 在 GitHub 上 391 MB，实测只有约 165 KB/s
（已试 6 个加速镜像，全部更慢或不可用）。同一批 DLL 在 PyPI 上打包成
`nvidia-cublas-cu12` / `nvidia-cuda-runtime-cu12` 的 win_amd64 wheel，改走清华镜像
速度高一个数量级。`07b-fetch-cudart.py` 就是干这个的。
（注意：PyPI JSON 里返回的 `url` 字段指向官方 CDN `files.pythonhosted.org`，
不是镜像本体，用之前要把 host 换掉，否则等于没走镜像。）

**2. 这个 build 用上 GPU 也**不**打印 CUDA 日志 —— 别用日志判断跑在哪里。**

只补了 `cudart64_12.dll`（缺 cublas）时，`llama-server.exe` 能正常启动、模型正常加载、
`/health` 返回 ok，但实际跑在 CPU 上（约 10 tok/s）；补齐 `cublas64_12.dll` +
`cublasLt64_12.dll` 后**用上了 GPU，可日志内容一字不改**（verbosity 3 下两边都是
一个 `CUDA` 字样都没有）。

正确的判断方法：

```bash
cd setup/llama.cpp && ./llama-server.exe --list-devices
# 输出 CUDA0: NVIDIA GeForce RTX 3060 (12287 MiB, 11147 MiB free) 才算真的能用
```

或者直接看吞吐：本机实测 CPU 约 **10 tok/s**，GPU 约 **92 tok/s**（9 倍）。

**3. 缺 cublas 是静默回退，不是报错。**
CUDA 后端的加载失败在 verbosity 3 下不产生任何警告行，程序照常启动、照常服务，
只是全部算在 CPU 上。所以「能跑」不等于「跑对了」，得看 `--list-devices` 或吞吐。

---

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

## 🔐 安全问题：曾被硬编码到仓库的 API 密钥

**本仓库曾把内网 DeepSeek 的 `LLM_API_KEY` 明文写在脚本里**（`42c97c2` 引入，
`setup/04-run-math-eval.sh` 与 `setup/06-run-windows-native.sh`），并随提交推到了公开 fork。

### 已做的处理

1. 两个脚本**已删除硬编码密钥**，改为只从环境变量 / `setup/.llm_key`（已 gitignore）读取；
   缺失时脚本直接报错退出，不会静默拿空 key 去请求。
2. 实测验证：`env -u LLM_API_KEY` 跑通（gsm8k 2/2、100%、退出码 0）。

### ❗ 仅清工作区是不够的 —— 已重写 git 历史

删掉文件里的密钥只能保护「以后的提交」，历史里那个还会被 `git log -S` 挖出来。
所以密钥已从**全部 39 个提交**中清除，并强制推送覆盖了远端：

```
sk-8KaSEw...FZyVAm   ->   $(cat "$PROJ/setup/.llm_key" 2>/dev/null)
```

即历史里的脚本也改成读本地密钥文件，而不是硬编码。

**验证结果**：

- 全部 39 个提交无明文密钥（逐提交 `git grep`）
- 本地对象库层面无任何含密钥的对象（`git cat-file --batch-all-objects`）
- 重写后 HEAD 内容与重写前**逐字节一致**（`git diff` 为空）——只换了密钥那两行
- **上游的 31 个提交 SHA 与 GPG 签名原样保留**，fork 仍与上游共享历史

> 重写时踩到的坑：先用 `git filter-repo` 做，结果**上游提交的 SHA 全变了**。
> 原因：上游有 5 个提交带 GPG 签名，而 filter-repo 默认会**剥离签名**，
> 导致从这些提交派生的所有 SHA 变化（根提交就带签名，所以全部 39 个都变了）。
> 改用 `git filter-branch` 且**只重写自己那 8 个提交**（`4330d20..HEAD`）后，
> 范围外的上游提交根本不被触碰，SHA 与签名自然保持。

### ⚠️ 仍需你做：轮换密钥

**历史清干净 ≠ 密钥安全**。这个密钥曾经在公开仓库里待过（GitHub 公开仓库基本会被
爬虫扫到），应当视为已泄露。重写历史拦不住已经抓取过它的人，**去把密钥重置掉**才是根治。
轮换后旧的自动失效。

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

---

## 后续：本机中转（`local_relay/`）—— 从「能跑评测」到「日常能用」

上面这套是在**跑论文评测**。之后的目标变成「让 PyroDash 的思路在本人机器上真的能用」，
也就是「本机小模型 + 远端大模型」当作一个 OpenAI 兼容服务挂给 pi 用。这部分落在
`local_relay/`，详细文档见 `local_relay/README.md`，这里只记结论和踩过的坑。

### 1. GPU 启用（10.2 → 92 tok/s）

| 项 | 结果 |
|---|---|
| 设备 | RTX 3060 12GB，compute capability 8.6（Ampere），驱动 537.70 |
| CPU 推理 | 9.6 – 10.2 tok/s |
| **GPU 推理** | **87.5 – 92 tok/s（约 9 倍）** |
| prompt 吞吐 | 800 – 949 tok/s |
| 带宽上限 | ≈ 360 GB/s ÷ 2.33 GB ≈ **154 tok/s**（已用到理论值的 ~57%） |

坑：**这个 llama.cpp 构建即使 GPU 生效也不会在日志里打 CUDA 行。** 判设备位置只能靠
`llama-server.exe --list-devices`（→ `CUDA0: NVIDIA GeForce RTX 3060`）或吞吐量。
另外缺 DLL 时它会**静默回退 CPU**：服务照常启动、`/health` 照常 200、照常返回结果，
只是慢 9 倍。

CUDA DLL 取自清华 PyPI 的 wheel（5.07 MB/s），没用 GitHub 上那个 391 MB 的 zip（106 KB/s）。
注意 `07b-fetch-cudart.py` 有个坑：PyPI JSON 里的 `url` 指向官方 CDN，必须替换**主机名**，
不能拼接路径。

### 2. 路由策略：小模型做「工具 + 日常事务」，分析类直接上大模型

见 `local_relay/README.md` §1.1。三层判定（显式声明 > `tools` 存在 > 关键词/长度），
9 个用例全部符合预期；工具调用现在由本机 4B 自己产出（实测 `get_weather({"city":"北京"})`）。

### 3. 远端模型：`deepseek-v4-flash` + 关掉思考

内网网关（`ai-api.bj.tkoffice.cn`，36 个模型）只认 **`thinking: {"type": "disabled"}`**；
relay 原本继承自 `_call_dashscope_chat` 的 `chat_template_kwargs.enable_thinking`
被网关**静默忽略**，所以 `LLM_ENABLE_THINKING` 一直没起作用。

两个模型开着思考时都会把 2048 预算全烧在 reasoning 上、正文 0 字被截断；
关掉后 flash 8.0s / pro 25.3s 正常答完，因此默认 `LLM_ENABLE_THINKING=0`。
同题同预算 flash 比 pro 快 2–3 倍。

### 4. 修掉的三个 bug（都会静默出错）

| # | 位置 | 症状 | 修法 |
|---|---|---|---|
| 1 | `relay_server.py` 取 temperature | `float(body.get("temperature") or ...)` 把 `temperature=0.0` 当成 falsy 而换成 0.6，**确定性测试全是假的** | 改 `is not None` 判断 |
| 2 | `_finish_reason` | 恒返回 `stop`（`tool_calls` 除外），**所有截断被静默掩盖**；流式分支又硬编码一遍 `stop` | 按 route / `small_stop_type` / `llm_finish` 如实上报 `length`；流式路径透传真实值 |
| 3 | 大模型腿预算 | 共享预算下 `llm_tokens=0` 却报 `stop` | 独立预算（`SHARED_BUDGET=0`）+ 预算耗尽显式标 `handoff:budget-exhausted` |

顺带把 `_sse()` 补上 `delta.tool_calls` + `finish_reason: "tool_calls"`。
**漏了这一步会让 pi 的工具调用在流式下静默失效**（pi 默认就是流式 + 工具）。

### 5. 官方 WSL2 路径仍未做

需要管理员权限 + UAC（装驱动 616.92 后重启），命令：

```powershell
powershell -ExecutionPolicy Bypass -File D:\hecan\PyroDash\setup\01-enable-wsl.ps1
```

之后 `02 → 05 → 04`。**目前的工作路径不依赖它**；只有要跑全量 vLLM 评测才需要。
另外：**不建议为了这个项目升级驱动/CUDA**（537.70 已满足 CUDA 12.x 的 minor 兼容，
RTX 3060 的 sm_86 有原生 cubin，升级只带来百分之几）。

