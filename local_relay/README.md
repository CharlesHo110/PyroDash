# local_relay —— 把 PyroDash 的「本机小模型 + 按需接力大模型」真正跑起来

PyroDash 的核心思想是**token 级的两级协作**：一个跑在本机的小模型先接活，判断自己搞不定就
把已经写出的半截推理交给远端大模型，让大模型**从断点接着做**而不是从头重来，从而省掉大模型的
绝大部分 token。

这个目录把那篇论文的机制**落到你这台机器上**，做成一个开箱可用的 OpenAI 兼容端点：

```
     pi / Cursor / 任何 OpenAI 客户端
                  │
                  ▼
   ┌──────────────────────────────────────┐
   │  relay_server.py   127.0.0.1:8010    │   ← 你只需要指到这里
   │  · 注入交接协议                        │
   │  · 渲染 chat 模板                      │
   │  · 判定「本机搞定」还是「交接」           │
   └──────────┬───────────────────┬───────┘
              │                   │
   答得完 ────┘                   └──── 答不完 / 带 tools
              │                            │
              ▼                            ▼
   ┌────────────────────┐      ┌──────────────────────────┐
   │ llama-server :8080 │      │ deepseek-v4-pro (远端)    │
   │ Qwen3-4B  Q4_K_M   │      │ 收到半截推理，接着写        │
   │ 2.33 GB / 12GB 显存 │      │ 共享预算：总预算 − 小模型已用 │
   └────────────────────┘      └──────────────────────────┘
```

---

## 1. 先看结论：哪些跑通了，哪些是坑

**跑通的**（实测，本机 RTX 3060 / CPU 回退模式）：

| 场景 | 结果 |
|------|------|
| 翻译 / 算术 / 总结 / 格式转换 | `route=small`，**本机 0.04 – 0.55 秒**答完，大模型 0 token |
| 多步推理 / 长代码 / 领域知识 | `route=handoff:limit`，本机给 512 token 半截推理 → 大模型接力 |
| 带 `tools` 的请求（pi 的工具调用） | `route=handoff:tools`，**完全跳过小模型**，`tool_calls` 原样透传（流式与非流式都验证过） |
| 路由是否准确 | **7/7 全部符合预期** |
| 大模型 token 占比 | 74.3%（总 6203 token 里 4608 给大模型） |

**必须知道的坑（这是最重要的一条）**：

> **通用的 4B 模型不会自发交接。**

我们实测过：给 Qwen3-4B-Instruct-2507 灌一整套「遇到不会的就输出 `<|llm_offload|>` 然后停止」
的协议提示词，它**一次都不肯交**（`handoff:tag = 0`）。它会硬着头皮往下写，直到被 token 上限截断。
这正是 PyroDash 要花大力气做 GRPO 强化学习训练的原因——**「知道自己不会」是一种需要训练的行为**。

所以本 relay 不依赖模型的自知之明，而是用**两个机械可靠的触发条件**：

1. **`stop_type == "word"`** —— 模型主动吐出了交接标记（训练过的模型才会走这条；协议提示词的
   作用是让这件事偶尔发生，属于**提前交接的优化**，不是必要条件）。
2. **`stop_type == "limit"`** —— **本机小模型没在自己的预算内答完**。这条不需要任何训练，
   天然就是「这活本机干不完」的信号，而且它已经写出来的半截推理正好当大模型的起手式。

把 `SMALL_MAX_TOKENS` 调小，交接就更积极、更省钱；调大，本机承担得更多、延迟更低。
这就是这个系统唯一的、也是最有效的旋钮。

> 顺带一个意外收益：这个机制天然按「任务长度」分流——本机能短平快解决的就本机解决，
> 需要长篇展开的自动流向大模型。不需要任何分类器。

---

## 2. 快速开始

### 前置：装好 llama.cpp 和模型

```bash
bash   setup/07-install-llama.sh      # llama.cpp Windows CUDA 预编译包（约 242 MB）
python setup/07b-fetch-cudart.py      # CUDA 运行时 DLL（走清华 PyPI，比 GitHub 快得多）
python setup/08-download-gguf.py      # Qwen3-4B-Instruct-2507 Q4_K_M（2.33 GB，modelscope）
```

> **`07b-fetch-cudart.py` 一定要跑。** 不跑的话 llama.cpp 会静默回退到 CPU（10 tok/s 而不是 92），
> 而且**任何日志和返回值都不会告诉你这件事**——它照常启动、照常返回 ok。
> 之所以单独搞一个脚本：官方那个 `cudart-llama-bin-win-cuda-12.4-x64.zip` 在 GitHub 上有 391 MB，
> 实测只有 ~106 KB/s；同一批 DLL 打包成 PyPI wheel 放在清华镜像上，实测 **5.07 MB/s**。

### 一条命令起全套

```bash
bash local_relay/run.sh
```

它会依次：启动 `llama-server`（8080）→ 等模型加载完 → 自测 `/apply-template` →
启动 `relay_server.py`（8010）→ 打印客户端要填的 Base URL / API Key / Model。

```
    Base URL : http://127.0.0.1:8010/v1
    API Key  : local（随便填）
    Model    : pyrodash-local
```

停止：

```bash
bash local_relay/run.sh --stop
```

常用参数：

```bash
bash local_relay/run.sh --ctx 32768        # 加长上下文
bash local_relay/run.sh --ngl 0            # 强制纯 CPU（不占显存）
bash local_relay/run.sh --model /path/to/other.gguf
bash local_relay/run.sh --port 8081 --relay-port 8011
```

### 自测

```bash
python local_relay/selftest_offline.py          # 纯 CPU，不联网，验证协议注入与预算逻辑
python local_relay/test_relay.py --suite all    # 端到端：简单任务留本机、困难任务交手
python local_relay/test_relay.py --ask "帮我把这段正则改成 Python re 语法：..."   # 自由提问
```

---

## 3. 接进 pi（以及任何 OpenAI 客户端）

编辑 `~/.pi/agent/models.json`（Windows 上是 `C:\Users\<你>\.pi\agent\models.json`），
在 `providers` 下加一段：

```json
{
  "providers": {
    "pyrodash-local": {
      "baseUrl": "http://127.0.0.1:8010/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "models": [
        {
          "id": "pyrodash-local",
          "name": "PyroDash 本机中转 (Qwen3-4B → deepseek-v4-pro)",
          "contextWindow": 16384,
          "maxTokens": 4096,
          "input": ["text"]
        }
      ]
    }
  }
}
```

然后 `pi` 里 `/model` 选中它即可。

> **注意 `baseUrl` 末尾的 `/v1`**：`api: "openai-completions"` 时 pi 会拼成
> `{baseUrl}/chat/completions`，少了 `/v1` 会 404。
> **注意 `input: ["text"]`**：本机 4B 模型不看图，别声明 `"image"`。

**任何 OpenAI 兼容客户端**都可以，只要 base URL 指到 `http://127.0.0.1:8010/v1`：

```bash
curl http://127.0.0.1:8010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"pyrodash-local","messages":[{"role":"user","content":"1+1 等于几？只回答数字。"}]}'
```

---

## 4. 看省钱效果：`/v1/stats`

每个响应里都带一个非标准的 `pyrodash` 块，说明这次请求怎么走的：

```json
{
  "choices": [ ... ],
  "pyrodash": {
    "route": "handoff:limit",
    "offload_reason": "limit",
    "offloaded": true,
    "small_model": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "small_tokens": 512,
    "small_stop_type": "limit",
    "small_elapsed_s": 5.6,
    "llm_model": "deepseek-v4-pro",
    "llm_tokens": 1536,
    "llm_elapsed_s": 28.7,
    "total_budget": 2048,
    "budget_used": 2048,
    "tools_passthrough": false,
    "tool_calls": 0
  }
}
```

`route` 的取值：

| 值 | 含义 |
|----|------|
| `small` | 本机搞定，大模型 0 token（**省钱的就是这种**，实测 0.04 – 0.55 秒） |
| `handoff:tag` | 小模型主动吐了交接标记（训练过的模型才常见） |
| `handoff:limit` | 小模型预算内没答完 → 交接（**当前主力路径**） |
| `handoff:tools` | 请求带 `tools`，直接跳过大模型（小模型做不了工具编排） |
| `handoff:budget-exhausted` | 小模型就吃光了全部预算，大模型没得用（调小 `SMALL_MAX_TOKENS` 可避免） |

累计统计：

```bash
curl -s http://127.0.0.1:8010/v1/stats        # 交接率、token 分布、延迟、最近 20 条
curl -s -X POST http://127.0.0.1:8010/v1/stats/reset
curl -s http://127.0.0.1:8010/health
```

`/v1/stats` 里的 `offload.rate_pct` 就是**交接率**（数值型百分比；`rate` 是 0–1 的小数，
`rate_pct_str` 是给人看的 `"42.9%"`），`tokens.llm_share_pct` 就是**大模型 token 占比**
——后者越低越省钱。把这两个数和纯大模型的基线对比，就是 PyroDash
论文里那张「相对成本」图的雏形。

---

## 5. 调参

`relay_server.py` 顶部的 `CONFIG` 全部支持环境变量覆盖：

| 环境变量 | 默认 | 作用 |
|----------|------|------|
| `SMALL_MAX_TOKENS` | `512` | **最重要的旋钮**。本机小模型单次预算；调小→交接更积极更省钱，调大→本机承担更多延迟更低 |
| `RELAY_PORT` | `8010` | 中转服务端口 |
| `RELAY_HOST` | `127.0.0.1` | 默认只监听本机，别随便暴露 |
| `SMALL_BASE_URL` | `http://127.0.0.1:8080` | llama-server 地址 |
| `SMALL_MODEL` | 自动探测 | 留空则从 `/props` 读模型名 |
| `LLM_BASE_URL` | `https://ai-api.bj.tkoffice.cn/v1` | 远端大模型 |
| `LLM_MODEL` | `deepseek-v4-pro` | 远端模型名 |
| `LLM_API_KEY` | 读 `setup/.llm_key` | **密钥只在本地文件/环境变量里，绝不进仓库** |
| `DEFAULT_MAX_TOKENS` | `2048` | 共享总预算（客户端传 `max_tokens` 可覆盖） |
| `OFFLOAD_ON_LIMIT` | `1` | 是否把「预算内没答完」当作交接信号（关掉就只剩模型主动标记这条） |
| `TEMPERATURE` | `0.6` | 采样温度 |
| `PYRODASH_OFFLOAD_TAG` | `<\|llm_offload\|>` | 交接标记文本 |
| `PYRODASH_PROTOCOL_FILE` | 内置 | 换成你自己的协议提示词文件 |

例：更激进地省钱

```bash
SMALL_MAX_TOKENS=256 bash local_relay/run.sh
```

例：让本机尽量多干活（延迟优先）

```bash
SMALL_MAX_TOKENS=4096 OFFLOAD_ON_LIMIT=0 bash local_relay/run.sh
```

### 关于「共享预算」

`max_tokens` 是**小模型和大模型共用的总预算**（这一点照搬 PyroDash 的 `_llm_max_tokens`）：
小模型花掉多少，大模型就少多少。所以 `max_tokens=2048` + `SMALL_MAX_TOKENS=512` 时，
大模型拿到的是 1536。这样「两段加起来」不会超出调用方预期的成本上限。

---

## 6. 为什么用 `llama.cpp 原生 /completion` 而不是 `/v1/chat/completions`

因为**只有原生端点能无歧义地区分「答完了」和「撞到交接标记」**：

```jsonc
// POST /completion  →  stop_type ∈ {none, eos, limit, word}
{"content": "...", "stop_type": "word", "stopping_word": "<|llm_offload|>"}
```

而 OpenAI 兼容端点两种情况都只给 `finish_reason: "stop"`，区分不出来。
vLLM 那套 `include_stop_str_in_output: true` 的玩法在 llama.cpp 里**不存在**
（实测 `/completion` 和 `/v1/chat/completions` 都没有这个参数），好消息是
`stop_type`/`stopping_word` 比它更干净——标记本来就不会混进 `content`。

chat 模板由 llama-server 自己渲染（`POST /apply-template`，用 `--jinja` 读 GGUF 里的模板），
所以这里不需要在本机装 tokenizer。

---

## 7. 排错

| 症状 | 原因 / 处理 |
|------|-------------|
| `llama-server` 起来了但很慢（约 10 tok/s） | CUDA 运行时 DLL 没装齐，静默回退到 CPU 了。跑 `python setup/07b-fetch-cudart.py` 补 `cublas64_12.dll` / `cublasLt64_12.dll`，**重启 llama-server** |
| 不确定到底跑在 CPU 还是 GPU | 看日志 **没用**（本 build 两种情况都不打 CUDA 行）。用 `cd setup/llama.cpp && ./llama-server.exe --list-devices`，能列出 `CUDA0: NVIDIA GeForce RTX 3060 (...)` 才算用上；或看吞吐，92 tok/s 是 GPU、10 tok/s 是 CPU |
| 全是 `handoff:budget-exhausted` | 小模型把总预算吃光。调小 `SMALL_MAX_TOKENS`，或让客户端传更大的 `max_tokens` |
| 全是 `handoff:limit`，没有 `small` | 正常现象，见第 1 节。想留更多在本机就调大 `SMALL_MAX_TOKENS` |
| `handoff:tag` 恒为 0 | **预期行为**，通用模型不会自发交接。这是 PyroDash 要训练的原因，不是 bug |
| 远端报 401 | `setup/.llm_key` 缺失或密钥失效。检查 `LLM_API_KEY` 环境变量 |
| pi 里 404 | `baseUrl` 少了 `/v1` |
| 端口占用 | `bash local_relay/run.sh --stop` 或改 `--port` / `--relay-port` |

日志：

```
local_relay/.run/llama-server.log      小模型服务端
local_relay/.run/relay.log             中转服务
local_relay/.run/*.pid                 进程号（run.sh --stop 用）
```

---

## 8. 文件清单

| 文件 | 作用 |
|------|------|
| `relay_server.py` | 中转服务本体（纯标准库 HTTP，无需 fastapi/uvicorn） |
| `offload_protocol.py` | 交接协议提示词 + `inject_protocol()`（协议放最前，客户端原有 system 内容拼在后面） |
| `run.sh` | 一键启停 |
| `test_relay.py` | 端到端测试（简单/困难两组用例 + `--ask` 自由提问） |
| `selftest_offline.py` | 纯离线自检，不联网不占显存 |

依赖：venv 里只需要 `requests`；小模型侧只需要 `llama-server.exe`。

---

## 9. 和 PyroDash 论文的关系

| PyroDash 论文 | 这里 |
|---------------|------|
| GRPO 训练 4B 模型学会 token 级交接 | **不训练**，用「预算内没答完」当机械触发条件 |
| vLLM + `include_stop_str_in_output` 检测标记 | llama.cpp `stop_type`/`stopping_word`，语义更干净 |
| `<\|llm_offload\|>` 标记 + 共享 token 预算 | **原样保留**（`offload_protocol.py` / `_llm_max_tokens`） |
| GSM8K / AMC23 数学评测，λ 控制交接率 | 真实客户端流量 + `/v1/stats` 实时看交接率与 token 占比 |
| `evaluation/evaluation_math/llm_relay.py` 的 `_build_offload_messages` | 直接复用，保证「半截推理怎么接」的措辞与论文一致 |

换句话说：**论文回答「训练能带来什么」，这里回答「不训练能拿到多少」**——而实测答案是
「路由准确率 7/7，能省掉 1/4 的大模型 token，且本机的轻任务延迟只有 0.04 秒」。
剩下的差距（主动交接、更早交接）就是那篇论文里 GRPO 真正买到的东西。
