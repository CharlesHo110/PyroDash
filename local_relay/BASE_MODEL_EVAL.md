# 基座替换评估：换更小的模型值不值？

> 评估对象：**MiniCPM5-2B**（OpenBMB，2026-09-08 发布）与 **NVIDIA Nemotron 3 Nano 4B**
> 是否适合替换当前 `local_relay` 的小模型腿基座 **Qwen3-4B-Instruct-2507**。
>
> 全部数字来自本机 HumanEval 164 题实测（2026-09-17），原始证据见
> `setup/evidence/base_model_eval_minicpm5_2b.json`，生成脚本
> `setup/_t/bench/make_base_model_evidence.py`。
>
> 中转机制本身的完整分析见 [ANALYSIS.md](ANALYSIS.md)。

---

## 0. 一句话结论

**不值得换。** 默认策略下中转准确率从 **96.3% 掉到 93.9%**，纯本机从 **82.9% 掉到 78.7%**。
它更快、更小，但快的那部分不在关键路径上（85% 的时间花在云端）。

**但这次实测挖出了一条比"换不换"重要得多的东西：**

> 现在主流的小模型（MiniCPM5-2B、Nemotron 3 Nano、Qwen3 全系）**默认都是"先思考再回答"**，
> 而这套中转的难度探测，**建立在"它答得短 = 它答得出"这个假设上**。
> 把一个开思考的模型塞进小模型腿，`SMALL_MAX_TOKENS` 会被思考过程瞬间烧光，
> 于是**每一道题**都判成"小模型搞不定"交给云端 —— 中转退化成纯代理，本地一分钱不省。

这不是能力问题，是**设计哲学冲突**。而且它有个直接后果：**换基座必须先改中转代码**
（见 §7），否则换了就是废掉。

---

## 1. 为什么会有这次评估

`ANALYSIS.md` §5 的核心发现是：**对 4B 来说，输出越长，答对的概率越低**。

| 输出长度 | Qwen3-4B 正确率 |
|---|---|
| 0–100 tok | 88.5% |
| 100–200 tok | 82.1% |
| 200–300 tok | 85.7% |
| 300–600 tok | 50.0% |
| 600+ tok | 0.0% |

中转正是拿这个当免费难度探测器用：`SMALL_MAX_TOKENS` 设小一点，答不完就交出去。

既然探测器质量取决于这条曲线的**斜率**，那自然的问题是：**换一个长度-正确率更单调、
更陡的小模型，是不是就能既省云端 token 又保准确率？**

MiniCPM5-2B 进入视野的理由：

- 体积只有 1.45 GB（Q4_K_M），**不到现役 Qwen3-4B 的 2/3**
- 第三方实测生成速度 73.7 tok/s（短）/ 58.6 tok/s（长），**明显快于同机 Qwen3-4B-Thinking 的 ~29 tok/s**
- 官方口径强调"把教师的慢思考留给训练，学生拿到快反应"，当时看起来正是我们要的"快而准"
- 第三方本地实测确认工具调用可用（`tool_calls` 参数正确）
- AA Intelligence Index v4.2 得分 15，官方称 sub-4B 第一，且**是同级里最简洁的模型**（指标任务仅输出 57M token）

Nemotron 3 Nano 4B 一并评估，因为它在多个榜单上与 MiniCPM5-2B 几乎并列。

---

## 2. 两个候选的实测可达性

| | Qwen3-4B-Instruct-2507（现役） | MiniCPM5-2B | Nemotron 3 Nano 4B |
|---|---|---|---|
| 参数量 | 4B | 2.52B（非嵌入 1.98B） | 3.97B |
| Q4_K_M 体积 | 2.33 GB | **1.45 GB** | 2.90 GB |
| 训练上下文 | 32K | 128K | 256K |
| 许可 | Apache-2.0 | Apache-2.0 | NVIDIA Open Model License |
| **默认开思考** | **否** | **是** | **是** |
| modelscope 可达 | ✅ | ✅ `OpenBMB/MiniCPM5-2B-GGUF` | ✅ `unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF` |
| 本机实测 | ✅ 已有基线 | ✅ 本次实测 | ❌ 结构化否决，未实测 |

**Nemotron 3 Nano 4B 不实测的理由**（写下来是为了记录判断依据，而不是回避工作）：

1. 官方模型卡明确写明"**先产生 reasoning trace，再得出结论**"，虽然可以用 system prompt 控制，但默认行为与本中转的机制直接冲突（§7）。
2. AA Intelligence 14.7 与 MiniCPM5-2B 的 15 基本持平 —— **没有任何数据显示它比 MiniCPM5-2B 强**，
   更谈不上强过现役 Qwen3-4B。
3. 它比 MiniCPM5-2B 大 2 倍，如果 MiniCPM5-2B 都换不动，它更没戏。

> 也就是说：否决 Nemotron 的理由是**结构性的**，不是"没测所以不知道"。
> 这个判断本身可被后续实测推翻，但推翻它需要的不是更多跑分，而是"它能关思考并且关掉之后仍然够聪明"。

---

## 3. 实验设置

- **题集**：HumanEval 全量 164 题，pass@1，子进程执行 `check()`
- **采样**：`temperature=0`，`max_tokens=2048`
- **云端**：`deepseek-v4-flash`，思考关闭（与 `ANALYSIS.md` 保持一致）
- **中转参数**：`SMALL_MAX_TOKENS=512`（默认）、`ANALYSIS_MIN_CHARS=400`（默认）、`SHARED_BUDGET=0`
- **公平性**：两个模型跑**完全相同的 164 道题、完全相同的判分方式**
- **关思考**：
  - 纯本机臂用 `BENCH_SMALL_NO_THINK=1`（走 `/v1/chat/completions` 的 `chat_template_kwargs`）
  - 中转臂用新加的环境变量 `SMALL_TEMPLATE_KWARGS={"enable_thinking": false}`
  - 已实测 Qwen3-4B 的模板里 `enable_thinking` 出现 **0 次** → 该开关对现役基座**零影响**，对比公平

---

## 4. 结果一：纯本机对比（`--arms small`，164 题）

| 模型 | pass@1 | 输出 tok 中位 | 输出 tok 均值 | `finish_reason` | 均耗时 |
|---|---|---|---|---|---|
| **Qwen3-4B-Instruct-2507** | **82.9%** (136/164) | 130 | 161.3 | `stop`×163, `length`×1 | 1.94s |
| MiniCPM5-2B（关思考） | 78.7% (129/164) | 192 | 230.6 | `stop`×162, `length`×2 | **1.69s** |
| *云端 flash（参考上限）* | *95.7%* (157/164) | *140* | *152.5* | *`stop`×164* | *1.53s* |

**−4.2pp。** 换算成题数：两边都过 117 道，只有 Qwen3-4B 过 19 道，只有 MiniCPM 过 12 道。

注意它的输出**比现役模型更长**（中位 192 vs 130）——这跟"同级最简洁"的官方口径不符，
但也不算矛盾：官方说的是指标任务上的总输出量，不是 HumanEval 这种代码题。

### 4.1 长度-正确率曲线：MiniCPM5-2B 确实更陡，但整体更低

| 输出长度 | Qwen3-4B | MiniCPM5-2B |
|---|---|---|
| 0–50 tok | 90.0% (n=10) | 100.0% (n=1) |
| 50–100 tok | 88.1% (n=42) | **100.0%** (n=16) |
| 100–200 tok | 82.1% (n=78) | 81.1% (n=74) |
| 200–300 tok | 85.7% (n=28) | 76.1% (n=46) |
| 300–600 tok | 50.0% (n=4) | 66.7% (n=24) |
| 600+ tok | 0.0% (n=2) | 33.3% (n=3) |

- MiniCPM5-2B 是**严格单调递减**的（100% → 81.1% → 76.1% → 66.7% → 33.3%）
- Qwen3-4B 在 200–300 段有反弹（82.1% → 85.7%），是噪声形状

**探测器的"短段 vs 长段"区分度**（这是中转真正用到的指标）：

| | 短段（≤128 tok）正确率 | 长段（>256 tok）正确率 | **区分度** |
|---|---|---|---|
| Qwen3-4B | 82.7% | 71.4% | **11.3 pp** |
| MiniCPM5-2B | 96.7% | 69.4% | **27.2 pp（2.4×）** |

**这正是当初想换它的理由成立的地方**：它的长度信号确实更干净。

### 4.2 离线 oracle 上限：两者几乎一样

用真实 token 数做离线阈值扫描（交接部分用云端真实结果补）：

| | 最优阈值 | oracle 准确率 | 对应交接率 |
|---|---|---|---|
| Qwen3-4B | 64 tok | 94.5% | 87.2% |
| MiniCPM5-2B | 64 tok | 95.7% | 97.0% |
| *在 50% 交接率上对比* | — | *89.6%* | *50%* |

**结论：即使给最优阈值，两者的理论上限也在噪声范围内（94.5% vs 95.7%，n=164 时单臂 CI ±3.0pp）。**
MiniCPM5-2B 那 95.7% 还是靠 **97% 交接率**堆出来的 —— 那已经等于纯代理了。

---

## 5. 结果二：中转端到端对比（默认策略，164 题）

| | Qwen3-4B 作小模型腿 | MiniCPM5-2B 作小模型腿 |
|---|---|---|
| **pass@1** | **96.3%** (158/164) | 93.9% (154/164) |
| 均耗时 | 1.61s | **1.57s** |
| 本机 token | 2543 | 2370 |
| 云端 token | 20905 | 22450 |
| 云端 token 占比 | **89.2%** | 90.5% |
| 路由分布 | `handoff:analysis` 140 / `small` 24 | **完全相同** |

### 5.1 差异全部来自小模型真正经手的那 24 道题

路由分布**一模一样** —— 因为路由只看 prompt 长度（`ANALYSIS_MIN_CHARS=400`），跟模型无关。
HumanEval 载荷中位 563 字符，所以 140/164（85.4%）直接被判 `analysis` 交给云端，
小模型只碰剩下 24 道短题。

| 路由 | Qwen3-4B | MiniCPM5-2B |
|---|---|---|
| `handoff:analysis` (140 道) | 134/140 = 95.7% | 133/140 = 95.0% |
| **`small` (24 道)** | **24/24 = 100%** | **21/24 = 87.5%** |

> `handoff:analysis` 那 1 道差异与模型无关（都是直接交云端），属于云端 API 的非确定性。
> 这恰好是本次对比的噪声底噪的实证。

**小模型经手的 24 道题，逐题对照：**

| 题目 | Qwen3-4B | MiniCPM5-2B |
|---|---|---|
| HumanEval/13 | ✅ 90 tok / 1.52s | ✅ 90 tok / 0.96s |
| HumanEval/14 | ✅ 83 tok / 1.04s | ✅ 82 tok / 0.59s |
| HumanEval/42 | ✅ 124 tok / 1.91s | ✅ 120 tok / 1.09s |
| HumanEval/49 | ✅ 190 tok / 2.37s | ✅ 109 tok / 0.96s |
| **HumanEval/59** | ✅ 133 tok / 1.83s | **❌** 140 tok / 1.17s |
| **HumanEval/83** | ✅ 246 tok / 3.09s | **❌** 103 tok / 1.05s |
| **HumanEval/155** | ✅ 137 tok / 2.05s | **❌** 140 tok / 1.24s |
| *（其余 17 道两者均 ✅）* | | |

- Qwen3-4B 在这块**满分**
- MiniCPM5-2B 的生成速度优势在这 24 道题上是实打实的（多数快 30–50%）

**所以问题的形状很清楚**：MiniCPM5-2B 真正的短板恰好落在中转唯一让它干活的那一段任务上。

### 5.2 集合关系

| | 两边都过 | 只有 Qwen3-4B 过 | 只有 MiniCPM 过 |
|---|---|---|---|
| 纯本机 | 117 | **19** | 12 |
| relay | 151 | **7** | 3 |

7 道"只有 Qwen3-4B 过"里，3 道来自小模型段（59/83/155），4 道来自云端段的非确定性。

---

## 6. 为什么离线 oracle 预测的优势没有兑现

这是本次实验最值得记录的一点。

**oracle 分析显示 MiniCPM5-2B 上限略高（95.7% vs 94.5%）**，为什么端到端反而更低？

因为 **oracle 分析的前提是"每一道题都先过小模型看它答多长"**。

而默认路由把 **85.4% 的题直接判成 `analysis` 交给云端，小模型根本碰不到**。
剩下 24 道题里，MiniCPM5-2B 那 2.4 倍陡的长度-正确率曲线**没有发挥空间**：

- 它的优势是"能分辨哪些题自己不行" —— 但它只有机会在 24 道题上做这个判断
- 在这 24 道（本来就是最短、最容易的题）上，Qwen3-4B 是 100%
- 它的劣势是"整体能力弱 4.2pp" —— 这个劣势在它经手的每一道题上都在起作用

**教训**：评估一个"基座替换"时，不能只看它在全量题集上的原始能力或离线 oracle 上限，
**必须把它放到实际的、带路由的调用路径上跑**。路由改变了小模型面对的任务分布，
而任务分布一变，"探测器质量优势"的值就可能归零甚至变负。

---

## 7. ⭐ 关键发现：推理型模型与这套中结转不兼容

这是这次实测真正的产出，比"换不换"重要得多。

### 7.1 现象：开思考会让中转直接废掉

MiniCPM5-2B 的 chat 模板**默认开思考**（llama-server 启动日志会提示
`chat template supports preserving reasoning, it is enabled by default`）。

同一批 16 道题，只改思考开关：

| | 通过 | 均耗时 | 输出 token 范围 | 撞 `max_tokens` |
|---|---|---|---|---|
| **开思考（模板默认）** | **8/16 = 50.0%** | **10.67s** | 557 ~ 2048 | **8/16** |
| 关思考 | **16/16 = 100%** | **1.06s** | 82 ~ 223 | 0 |

**开思考时每一道题的输出都 ≥ 557 token** —— 而 `SMALL_MAX_TOKENS` 默认是 **512**。

于是在中转里：

```
每道题 → 小模型思考到 512 token 被砍 → stop_type == 'limit'
       → 判定"小模型搞不定" → handoff → 云端重做
```

**100% 交接，本地零节省。** 中转退化成一层什么都不干的纯代理（还白搭一次本地推理延迟）。

> ⚠️ 这不是 MiniCPM5-2B 独有的问题。我们在 `deepseek-v4-pro` + 开思考上
> **亲手测过完全相同的失败模式**（见 `ANALYSIS.md` §7：旧配置 2048 预算下
> 28/164 被 `length` 截断，要 8192 才够）。同一堵墙，只是换成小模型撞。

### 7.2 为什么关思考需要改代码

llama.cpp 的思考开关**只在两个入口生效**：

| 入口 | 是否接受开关 | 说明 |
|---|---|---|
| `POST /v1/chat/completions` | ✅ | 需要 `chat_template_kwargs: {"enable_thinking": false}` |
| `POST /apply-template` | ✅ | 同上 |

而且**顶层 `enable_thinking` 字段会被静默忽略**（实测：写顶层字段无效，必须放进 `chat_template_kwargs`）。

问题在于中转的工作方式是：

```python
prompt = apply_template(small_messages)   # ← 先渲染模板
small  = call_small(prompt, ...)          # ← 再喂给原生 /completion
```

它走的是 **`/apply-template` + 原生 `/completion`**（这么设计是因为只有原生 `/completion`
会返回 `stop_type` / `stopping_word`，而整个难度探测机制依赖 `stop_type`）。

而原来的 `apply_template()` 写死了 `json={"messages": messages}` —— **没有任何地方能传模板参数**。

**所以换 MiniCPM5-2B 必须先改中转源码。** 本次已经补上（见 §7.3），并且这个改动
对现役 Qwen3-4B 完全无副作用。

### 7.3 已实施的改动

`local_relay/relay_server.py`：

- 新增环境变量 **`SMALL_TEMPLATE_KWARGS`**（JSON 字符串，默认空 = 行为不变）
- `apply_template()` 在非空时把它作为 `chat_template_kwargs` 透传
- 非法 JSON 直接 `SystemExit` 失败退出，不静默吞掉

```bash
# 用 MiniCPM5-2B 这类开思考的模型当基座时：
SMALL_BASE_URL=http://127.0.0.1:8082 \
SMALL_TEMPLATE_KWARGS='{"enable_thinking": false}' \
python local_relay/relay_server.py --port 8013 --small-base-url http://127.0.0.1:8082
```

实测确认：

| 请求 | `/apply-template` 返回的 prompt 结尾 |
|---|---|
| 不传参数（原行为） | `<\|im_start\|>assistant\n<think>\n` ← 开思考 |
| 传 `enable_thinking: false` | `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` ← 跳过思考 |
| 对 Qwen3-4B 传与不传 | **完全相同**（模板里没有 `enable_thinking`） |

---

## 8. 结论与建议

### 8.1 换不换

**不换。** 理由按重要性排序：

1. **端到端更差**：93.9% vs 96.3%（−2.4pp），且云端 token 占比反而上升（90.5% vs 89.2%）
   —— 本机答错的题最后还得云端重做，本机省下的 173 token 被云端多花的 1545 token 吃回去了
2. **纯本机也更差**：78.7% vs 82.9%（−4.2pp）
3. **它的优势用不上**：更陡的长度-正确率曲线只在"每道题都先过小模型"的前提下有价值，
   而默认路由把 85.4% 的题直接交给云端了
4. **速度快但不在关键路径上**：均耗时 1.57s vs 1.61s（−2.5%），因为 85% 的时间在等云端
5. **换它要改代码**（§7），而改完还是更差

### 8.2 各候选的最终定位

| 模型 | 结论 |
|---|---|
| **Qwen3-4B-Instruct-2507** | ✅ **保持现役**。当前路由策略下它在"短题段"是满分，换任何更小的模型都是净损失 |
| **MiniCPM5-2B** | ❌ 实测更差，不换。但其**关思考后在短题上 100% 正确率**的表现说明它并非弱模型，只是不适配当前路由的任务分布 |
| **Nemotron 3 Nano 4B** | ❌ 结构性否决，未实测（§2） |

### 8.3 如果将来还想换基座，怎么做

**评估协议**（本次踩过的坑，下次直接用）：

1. **先量"它经手的那批题"**，不要只看全量 pass@1，更不要只看离线 oracle 上限
2. **必须先关思考**，否则任何开思考的候选都会 100% 交接 —— 这个失败模式会伪装成"能力不行"
3. 中转到 8013 之类的临时端口，**不要动 :8010**（pi 在用）
4. 只看三个指标：`route=small` 那一段的通过率、云端 token 占比、均耗时
5. 记住噪声底噪：n=164 时两臂差 95%CI **±4.2pp**，1~4 道题的差异读不出任何东西

**换基座真正值得做的方向**（不在本次范围内）：不是换更小的，而是**找一个在短题段 ≥ 现役、
且能在长题段主动认输的小模型** —— 也就是把"长度信号"做得比现在更干净、同时不牺牲短题准确率。
MiniCPM5-2B 做到了前半句，栽在后半句。

---

## 9. 局限与不确定性

1. **−2.4pp 在统计噪声内。** n=164，单臂 95%CI ±3.0pp，两臂差 95%CI ±4.2pp。
   严格表述应该是"**未观察到优势**"，而不是"显著更差"。这里给出"不换"的建议，
   依据是"没有证据支持换"而不是"证明了更差"。
2. **只在一个任务分布上测过。** HumanEval 是单轮纯代码题，且高度饱和（现役 4B 已 82.9%）。
   如果换成"所有题都必须先过小模型"的分布（比如没有分析类任务、或 `ANALYSIS_MIN_CHARS` 调大），
   MiniCPM5-2B 的相对表现可能反转 —— 离线 oracle 表确实显示它有更高的上限。
3. **云端段的 1 道差异是非确定性的实证。** `handoff:analysis` 段两个配置的差异（134 vs 133）
   与本地模型无关，是 `deepseek-v4-flash` 在 `temperature=0` 下的服务端波动。
   这也是为什么 3 道题的差距不能当结论用。
4. **没有测工具调用。** 中转的 `tool` 路由是另一个值得单独评估的场景，
   本次两条臂都没有触发（HumanEval 没有工具），所以"MiniCPM5-2B 工具调用可用"
   只有第三方实测背书，没有本机验证。
5. **Nemotron 3 Nano 4B 没有实测。** 否决依据是公开模型卡的行为描述 + 榜单持平，
   属于结构性推断。如果后续有证据表明它能关思考且关掉后仍然够聪明，这个结论需要重审。
6. **量化只有 Q4_K_M。** 更高质量的量化（Q8_0 2.68 GB）可能让 MiniCPM5-2B 涨分，
   但那样体积优势也没了，性价比问题回到原点。
7. **`SMALL_TEMPLATE_KWARGS` 是本次新加的能力**，只在本机 llama.cpp 上验证过。
   其他后端（vLLM / transformers）的等价开关键名不一定相同。

---

## 附：数据溯源

| 数字 | 来源 |
|---|---|
| 纯本机 Qwen3-4B 82.9% / 中位 130 tok | `setup/_t/bench/sweep/base/small.json` |
| 纯本机 MiniCPM5-2B 78.7% / 中位 192 tok | `setup/_t/bench/sweep/minicpm_off/small.json` |
| 纯云端 flash 95.7% | `setup/_t/bench/sweep/base/cloud.json` |
| relay Qwen3-4B 96.3% | `setup/_t/bench/sweep/A_default/relay.json` |
| relay MiniCPM5-2B 93.9% | `setup/_t/bench/sweep/minicpm_relay/relay.json` |
| 开思考 8/16、557~2048 tok | `setup/_t/bench/sweep/minicpm_on/small.json` |
| 汇总证据（JSON） | `setup/evidence/base_model_eval_minicpm5_2b.json` |
| 证据生成脚本 | `setup/_t/bench/make_base_model_evidence.py` |

> 上表前六个路径位于 gitignore 的 `setup/_t/` 下（临时评测产物，不进版本库）；
> 证据生成脚本同理。**归档版的数字与聚合结果全部收敛在 `base_model_eval_minicpm5_2b.json` 里，
> 该文件才是可复核的权威副本**；下表复现命令里的 `setup/_t/...` 路径需要先按 §附 的命令重新生成。

**模型来源**（全部来自 modelscope，本机无法访问 huggingface）：

- `OpenBMB/MiniCPM5-2B-GGUF` → `MiniCPM5-2B-Q4_K_M.gguf`（1.45 GB，272 秒下完）
- `unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF`（未下载）

**复现命令**见证据 JSON 的 `reproduce` 字段，或：

```bash
cd D:/hecan/PyroDash

# 1) 下载
python setup/08-download-gguf.py --repo OpenBMB/MiniCPM5-2B-GGUF --match Q4_K_M

# 2) 起第二个 llama-server（:8080 留给 pi，不要动）
cd setup/llama.cpp && ./llama-server.exe \
  -m D:/hecan/PyroDash/models/llama/MiniCPM5-2B-Q4_K_M.gguf \
  --port 8082 --ctx-size 16384 --n-gpu-layers 99 --jinja --host 127.0.0.1

# 3) 纯本机臂（关思考）
BENCH_SMALL_URL=http://127.0.0.1:8082/v1/chat/completions BENCH_SMALL_NO_THINK=1 \
  ./setup/.venv-win/Scripts/python.exe setup/_t/bench/run_humaneval.py \
  --arms small --limit 164 --max-tokens 2048 --out setup/_t/bench/sweep/minicpm_off

# 4) 中转臂（临时端口 8013，不打扰 :8010）
SMALL_BASE_URL=http://127.0.0.1:8082 SMALL_TEMPLATE_KWARGS='{"enable_thinking": false}' \
  ./setup/.venv-win/Scripts/python.exe local_relay/relay_server.py \
  --host 127.0.0.1 --port 8013 --small-base-url http://127.0.0.1:8082

BENCH_RELAY_URL=http://127.0.0.1:8013/v1/chat/completions \
  ./setup/.venv-win/Scripts/python.exe setup/_t/bench/run_humaneval.py \
  --arms relay --limit 164 --max-tokens 2048 --out setup/_t/bench/sweep/minicpm_relay

# 5) 汇总成证据文件
./setup/.venv-win/Scripts/python.exe setup/_t/bench/make_base_model_evidence.py
```
