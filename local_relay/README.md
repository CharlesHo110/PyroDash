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
  答得完 ────┘                   └──── 答不完 / 分析类 / 带 tools
             │                            │
             ▼                            ▼
  ┌─────────────────────┐      ┌──────────────────────────┐
  │ llama-server :8080  │      │ deepseek-v4-flash (远端)   │
  │ Qwen3-4B  Q4_K_M    │      │ 收到半截推理，接着写        │
  │ 2.33 GB / 12GB 显存  │      │ 独立预算：客户端给的完整预算 │
  └─────────────────────┘      └──────────────────────────┘
```

---

## 1. 先看结论：哪些跑通了，哪些是坑

**跑通的**（实测，本机 RTX 3060 / CPU 回退模式）：

| 场景 | 结果 |
|------|------|
| 翻译 / 算术 / 总结 / 格式转换 | `route=small`，**本机 0.16 – 0.32 秒**答完，大模型 0 token |
| 多步推理 / 长代码 / 领域知识 | `route=handoff:limit`，本机给 512 token 半截推理 → 大模型接力 |
| **分析类任务**（「分析一下…」「explain why…」、超长需求） | `route=handoff:analysis`，**直接跳过小模型**，本机 0 token（实测 2.8 – 6.1 秒） |
| **带 `tools` 的请求**（pi 的工具调用） | `route=small`，**小模型自己产出 `tool_calls`**（实测 `get_weather({"city":"北京"})`）；给不出就 `handoff:tools-fallback` |
| 显式 `x-pyrodash-task` | 完全按声明走：`routine`→本机先试；`analysis`→直连云端；`tool`→小模型执行 |
| 路由是否准确 | **9/9 全部符合预期** |

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
> 需要长篇展开的自动流向大模型。

### 1.1 路由策略：小模型做「工具 + 日常事务」，分析类直接上大模型

上面那个「按长度自动分流」是**机械触发**的。后来我们又往它上面加了一层**任务分类**，
因为一条实测结论：**分析类任务不该给小模型试。**

理由很直接：分析类任务（推理、证明、权衡、架构、调试）几乎必然会被 512 预算截断，
那次尝试的唯一产出是一段**被丢弃的半截推理**——本机算力白烧，还多花 1–6 秒。
旧版共享预算下它还会挤占大模型腿的额度。

所以现在是三分：

| 任务类型 | 判据 | 路由 |
|---|---|---|
| `tool` | 请求带 `tools` | **小模型自己执行**（llama.cpp `--jinja` 渲染工具模板并解析 `tool_calls`）；
给不出可用调用则回退大模型 `handoff:tools-fallback` |
| `routine` | 翻译/格式化/提取/总结/改写等关键词；或长度 < `ANALYSIS_MIN_CHARS` | 小模型先答，预算内答完就本机搞定，否则 `handoff:limit` |
| `analysis` | 分析/推理/证明/对比/架构/调试等关键词；或长度 ≥ `ANALYSIS_MIN_CHARS`（默认 400） | **直接 `handoff:analysis` 交大模型**，本机 0 token |

判定优先级从高到低：

1. **客户端显式声明** —— body 里的 `pyrodash_task`，或 HTTP header `x-pyrodash-task`
   （值 `tool` / `routine` / `analysis`）。**这是唯一权威信号**，只有调用方知道自己要什么。
   body 里的值优先于 header。
2. 请求带 `tools` → `tool`。
3. 关键词 / 长度启发式 → `analysis` 或 `routine`。
4. 全没命中 → `UNKNOWN_TASK`（默认 `routine`）。

> **启发式一定会判错。** 所以它的作用被刻意限制为「决定要不要让小模型先试一下」：
> 判成 `analysis` 只是跳过本机尝试（最多多花几秒云端时间），判成 `routine` 也只是让本机
> 试一次、答不完照样交接。**判错的最坏结果是一次多余的本地尝试，不会产出错误答案。**
> 要精确控制就用第 1 条显式声明。

另外一个反直觉的点：**工具调用不但不该跳过大模型，反而是小模型最适合干的活。**
早期版本把带 `tools` 的请求直接透传云端（`handoff:tools`），理由是「4B 做不了工具编排」；
实测下来 4B 在 `--jinja` 下能稳定产出正确的 `tool_calls`，而工具选择本来就不需要深度推理。
现在工具验证交本机、真正的分析交云端，边界更合理。

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
          "name": "PyroDash 本机中转 (Qwen3-4B → deepseek-v4-flash)",
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
    "task": "routine",
    "task_source": "declared",
    "small_model": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "small_tokens": 512,
    "small_stop_type": "limit",
    "small_elapsed_s": 4.3,
    "llm_model": "deepseek-v4-flash",
    "llm_tokens": 752,
    "llm_elapsed_s": 10.6,
    "total_budget": 2048,
    "llm_budget": 2048,
    "budget_used": 1264,
    "tools_passthrough": false,
    "tool_calls": 0,
    "tool_calls_from": null
  }
}
```

`route` 的取值：

| 值 | 含义 |
|----|------|
| `small` | 本机搞定，大模型 0 token。含两种：事务性任务本机答完，**或工具调用由本机小模型产出** |
| `handoff:limit` | 小模型预算内没答完 → 交接（`routine` 的主力路径） |
| `handoff:analysis` | 判定为分析类任务，**小模型完全不参与**，直接转大模型 |
| `handoff:tools-fallback` | 带 `tools` 但小模型没给出可用 `tool_calls` → 回退大模型 |
| `handoff:tag` | 小模型主动吐了交接标记（训练过的模型才常见） |
| `handoff:budget-exhausted` | 小模型吃光了全部预算，大模型没得用（**只在 `SHARED_BUDGET=1` 下可能出现**） |

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

`recent` 是最近 20 次请求的**紧凑摘要**（路由/两侧 token/两侧耗时），不是完整响应体 ——
后者会让这个端点涨到几百 KB，而它的用途只是看占比。要读全文看服务端日志。

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
| `SMALL_TEMPLATE_KWARGS` | 空 | 透传给 `/apply-template` 的 `chat_template_kwargs`（JSON）。**换开思考的基座时必须用它关思考**，否则 100% 交接、中转退化成纯代理——见 [BASE_MODEL_EVAL.md](BASE_MODEL_EVAL.md) §7 |
| `LLM_BASE_URL` | `https://ai-api.bj.tkoffice.cn/v1` | 远端大模型 |
| `LLM_MODEL` | `deepseek-v4-flash` | 远端模型名 |
| `LLM_API_KEY` | 读 `setup/.llm_key` | **密钥只在本地文件/环境变量里，绝不进仓库** |
| `LLM_ENABLE_THINKING` | `0` | 远端模型是否开思考。**默认关**，原因见下 |
| `DEFAULT_MAX_TOKENS` | `2048` | 默认总预算（客户端传 `max_tokens` 可覆盖） |
| `SHARED_BUDGET` | `0` | **`0`=两腿各自独立预算**（小模型花的不占云端额度）；`1`=旧行为「总预算 − 小模型已用」 |
| `LLM_MAX_TOKENS` | `0` | 大模型腿单独预算；`0` 表示就用客户端的 `max_tokens` |
| `ANALYSIS_MIN_CHARS` | `400` | 超过这个字符数就判为分析类任务 |
| `UNKNOWN_TASK` | `routine` | 启发式全没命中时的兜底任务类型 |
| `OFFLOAD_ON_LIMIT` | `1` | 是否把「预算内没答完」当作交接信号（关掉就只剩模型主动标记这条） |
| `TEMPERATURE` | `0.6` | 采样温度 |

### 5.1 为什么默认换成 `deepseek-v4-flash`，以及为什么要把思考关掉

这两个决定是连在一起的，而且都是实测出来的。

**先把结论摆上**（同一道分析题，`max_tokens=2048`，`temperature=0`）：

| 模型 | 传参方式 | finish | 输出 tok | 思考字符 | 正文字符 | 耗时 |
|---|---|---|---|---|---|---|
| `deepseek-v4-pro` | 不传 | `length` | 2048 | 3757 | **0** | 47.5s |
| `deepseek-v4-pro` | `chat_template_kwargs.enable_thinking=false` | `length` | 2047 | 3759 | **0** | 42.3s |
| `deepseek-v4-pro` | `thinking={"type":"disabled"}` | `stop` | 1400 | 0 | 2636 | **25.3s** |
| `deepseek-v4-flash` | 不传 | `length` | 2048 | 3705 | **0** | 15.1s |
| `deepseek-v4-flash` | `chat_template_kwargs.enable_thinking=false` | `length` | 2048 | 7805 | **0** | 21.2s |
| `deepseek-v4-flash` | `thinking={"type":"disabled"}` | `stop` | 1298 | 0 | 2650 | **8.0s** |

两个坑：

**坑一：`chat_template_kwargs.enable_thinking` 是无效的。** relay 原本用的就是它
（继承自 PyroDash 的 `_call_dashscope_chat`），实测内网网关**静默忽略**这个参数，
`enable_thinking` 无论真假都不管用（传 `false` 时 flash 的思考反而涨到 7805 字符）。
网关只认 **`thinking: {"type": "disabled"}`**。这是已经修掉的 bug——
在那之前 `LLM_ENABLE_THINKING` 一直是个摆设。

**坑二：这两个模型的思考都远超 2048 预算。** 开着思考时，它们会把全部预算烧在
reasoning 上、正文一字未出就被截断（表里那些「正文 0 字符」就是），而且 `finish_reason`
会被报成 `length`——这是另一处已修的 bug（旧版恒报 `stop`，把截断静默掩盖了）。
所以**默认关思考**是必须的：关了之后 flash 8.0s / pro 25.3s 就能正常答完。

**那为什么选 flash？** 同预算同题目下 flash 比 pro 快 2–3 倍（8.0s vs 25.3s），
单位 token 吞吐也高大约一倍。代价是 flash 的思考更长（3705 vs 3757 字符，同量级）。
要开思考就得同时把预算抬到 ≥8192（实测 flash 需要 ~6240 tok 才答完）。

```bash
# 想开思考：必须同时给更大预算，否则 100% 被截断
LLM_ENABLE_THINKING=1 LLM_MAX_TOKENS=8192 bash local_relay/run.sh
```

### 5.2 独立预算：为什么不再从总预算里扣小模型的

旧行为（`SHARED_BUDGET=1`）沿用 PyroDash 的 `_llm_max_tokens`：大模型只能用
「客户端总预算 − 小模型已用」。小模型先花掉 512，云端就只剩 1536，
而开着思考的模型写不完推理+代码就被 `length` 截断。

实测证据（`HumanEval/130`）：

- 共享预算 2048 → 云端只拿到 1536 → **失败**
- 总预算 8192 → 云端拿到 2157 → **通过**

小模型的 token 是本机算的、不花钱，没有理由去占客户端付钱的远端额度。
所以默认 `SHARED_BUDGET=0`：两腿各自独立，云端直接拿客户端给的完整 `max_tokens`。
要复现论文里那种「总预算约束」，把 `SHARED_BUDGET` 设回 `1` 即可。
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

### 关于「共享预算」（旧行为，现在默认关闭）

PyroDash 原版的 `_llm_max_tokens` 把 `max_tokens` 当作**两腿共用的总预算**：
小模型花掉多少，大模型就少多少（`max_tokens=2048` + `SMALL_MAX_TOKENS=512` 时
大模型只剩 1536）。好处是「两段加起来」不超过调用方预期的成本上限。

但实测证明这在代码类任务上**直接损害正确率**（见 §5.2 的 `HumanEval/130`：
只剩 1536 时写不完推理+代码就被 `length` 截断）。所以现在默认
`SHARED_BUDGET=0`，即两腿各自独立。想复现论文里那种「总预算约束」把它设回 `1`。

### 5.3 `SMALL_MAX_TOKENS` 扫描：这个旋钮的真实曲线

只讲道理不够，我们把它测了。164 道 HumanEval，`max_tokens=2048`、`temperature=0`、
云端 `deepseek-v4-flash` 关思考，三组结果如下。

**先讲一件必须知道的事：默认策略下这个旋钮对代码题几乎不起作用。**
HumanEval 载荷中位 **563 字符**，`ANALYSIS_MIN_CHARS=400` 会把 **140/164（85%）**
判成 `analysis` 而直接跳过小模型。所以直接扫会得到一条假平线 ——
不是旋钮没用，是新路由策略本来就不让代码题碰到它。

#### 基线（两臂各自 164 题）

| 臂 | pass@1 | 均耗时 |
|---|---|---|
| `small` 纯本机 Qwen3-4B | **82.9%**（136/164） | 1.94s |
| `cloud` 纯云端 flash（关思考，2048） | **95.7%**（157/164） | 1.53s |

> 注意云端那 95.7% 是**在 2048 预算下达成的**。换 flash + 关思考之前，同样 2048
> 只有 83.5%（28/164 被 `length` 截断）；要 8192 才能到 95.7%。也就是说
> **flash + 关思考 让 2048 预算就够用了**，不再需要为了避开截断而抬高预算。

#### 组 A：默认策略（新路由）

| 指标 | 值 |
|---|---|
| pass@1 | **96.3%**（158/164） |
| 均耗时 | 1.61s |
| task 分布 | `analysis` 140 / `routine` 24 |
| route 分布 | `handoff:analysis` 140 / `small` 24 |
| 分路由通过率 | `handoff:analysis` 134/140（96%）· `small` **24/24（100%）** |
| 本机 / 云端 token | 2543 / 20905（云端占比 **89.2%**） |
| `finish_reason` | 164 题全部 `stop`（无截断） |

两个结论：

1. **准确率不再低于纯云端**：96.3% vs 95.7%，差 1 题，在噪声内。
   旧策略（所有 `tools`/长请求都透传）只有 85.4%。
2. **但在代码类任务上省不下来**：85% 的题被路由直接交出去，云端占比 89.2%。
   这里是**取舍而不是免费午餐** —— 代码题天生就被判成分析类。

值得注意的是那 24 道走了本机的题：**24/24 全对**，只花了 2543 个本机 token。
它们都是短题（载荷 < 400 字符），正好落在 4B 的舒适区。

#### 组 B ★：强制小模型路径，隔离出旋钮本身

用 `pyrodash_task=routine` 强制小模型先答（评测端设 `BENCH_RELAY_TASK=routine`），
这样 `SMALL_MAX_TOKENS` 就是唯一变量：

| `SMALL_MAX_TOKENS` | pass@1 | 交接率 | 本机 tok | 云端 tok | 云端占比 | 均耗时 |
|---|---|---|---|---|---|---|
| **128** | **90.9%** | 53.0% | 17667 | 16246 | 47.9% | 2.40s |
| 256 | 87.2% | 9.1% | 22858 | 3866 | 14.5% | 2.08s |
| 512（默认） | 84.8% | 1.2% | 23839 | 883 | 3.6% | 1.91s |
| 1024 | 84.8% | 0.0% | 24188 | **0** | **0.0%** | 1.87s |

单调、干净、可解释：**预算越小 → 交接越积极 → 准确率越高，代价是云端 token 和延迟**。

关键对比（相对纯本机 82.9%）：

- `SMALL_MAX_TOKENS=128`：准确率 **90.9%（+8.0 pp）**，只花 47.9% 的云端 token
- `SMALL_MAX_TOKENS=512`（默认）：84.8%（+1.9 pp），几乎没省
- `SMALL_MAX_TOKENS=1024`：84.8%，云端 token **归零** —— 这已经等于纯本机了

分路由看更有意思（`本机搞定` vs `交接后` 的通过率）：

| 扫描点 | 本机搞定 | 交接后 |
|---|---|---|
| 128 | 68/77（88%） | **81/87（93%）** |
| 256 | 128/149（86%） | **15/15（100%）** |
| 512 | 139/162（86%） | 0/2（0%，样本太小） |
| 1024 | 139/164（85%） | —（零交接） |

**交接那条路的通过率始终不低于本机那条**，而且本机通过率卡在 85–88% 不动。
反过来看，本机那列有个反直觉的事实：
**给 4B 更多预算并不能提高它的正确率** ——
77 → 149 → 162 → 164 道题走本机，通过率一直是 85–88%。
预算变大只是让它写得更长，不是写得更好；换来的是「本来能靠云端做对的题」
变成「本机自己做错」。

#### 结论：这个旋钮怎么调

| 你想要 | 怎么设 |
|---|---|
| 质量优先（推荐） | `SMALL_MAX_TOKENS=128` —— 90.9%，云端 token 减半 |
| 平衡 | `256` —— 87.2%，云端占比 14.5% |
| 成本优先 | `512`（默认）或更大 —— 84.8%，云端几乎不花钱 |

> **重要提醒：上表是「强制走小模型路径」下的数字。** 实际部署时默认策略生效
> （组 A，96.3%），代码/分析类题根本不进小模型，此时 `SMALL_MAX_TOKENS`
> 只影响那些被判定为 `routine` 的短任务。所以：
> **这个旋钮是给「短任务」调省钱幅度的，不是给「代码题」调正确率的。**
> 要让它对代码题生效，得先把 `ANALYSIS_MIN_CHARS` 调大（或显式声明 `routine`），
> 但那就等于放弃了 §1.1 那个「分析类别让小模型试」的优化——实测证明那个优化
> 在代码题上值 +11.5 pp（84.8% → 96.3%）。

原始数据：`setup/evidence/offload_sweep_humaneval164.json`；
复现：`bash setup/_t/bench/sweep.sh`（~55 分钟）。

> 📄 **完整分析（机制推导、置信区间、局限说明）见 [`ANALYSIS.md`](ANALYSIS.md)。**

### 5.4 换更小的基座值不值？

实测过：**不值。** 把小模型腿换成 MiniCPM5-2B（Q4_K_M 1.45 GB，比现役 Qwen3-4B 小 1/3、
生成快 1.6 倍），默认策略下中转 **96.3% → 93.9%**，云端 token 占比反而从 89.2% 升到 90.5% ——
本机答错的题最后还得云端重做，省下的本地 token 被多花的云端 token 吃回去了。

顺带挖出一个会**直接把中转废掉**的坑：现在主流小模型（MiniCPM5-2B、Nemotron 3 Nano、
Qwen3 全系）默认**开思考**，几千 token 的 reasoning 会把 `SMALL_MAX_TOKENS` 瞬间烧光 →
每道题都命中 `stop_type == 'limit'` → **100% 交接、本地零节省**，中转退化成纯代理。
换这类基座必须先关思考，而关思考要走上面表里的 `SMALL_TEMPLATE_KWARGS`。

> 📄 **完整评估（推理型模型兼容性、评估协议、局限说明）见 [`BASE_MODEL_EVAL.md`](BASE_MODEL_EVAL.md)。**

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
