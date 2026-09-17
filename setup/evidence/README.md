# 验证证据（对应用户计划 §4 验证清单）

这些 JSON 是脚本真实跑出来的原始输出，可直接复核，不是手写数字。

| 文件 | 对应条目 | 命令 |
|---|---|---|
| `smoke_gsm8k3.json` | §4 项 2 | `bash setup/06-run-windows-native.sh 3` |
| `compare_lambda0.05_gsm8k50.json` | §4 项 3 | `bash setup/06-run-windows-native.sh --compare 50 --tag l05` |
| `compare_lambda0.6_gsm8k50.json` | §4 项 3 / 项 4 | `PYRODASH_MODEL=models/PyroDash-4B-GRPO-Lambda-0.6 bash setup/06-run-windows-native.sh --compare 50 --tag l06` |
| `offload_sweep_humaneval164.json` | 方案 1：`SMALL_MAX_TOKENS` 扫描 | `bash setup/_t/bench/sweep.sh`（~55 分钟） |
| `base_model_eval_minicpm5_2b.json` | 基座替换评估：MiniCPM5-2B 值不值 | `./setup/.venv-win/Scripts/python.exe setup/_t/bench/make_base_model_evidence.py` |

> ⚠️ 上面后两条用的脚本在 gitignore 的 `setup/_t/` 下（临时评测产物，不进版本库）。
> JSON 本身是自包含的：完整复现命令在它的 `reproduce` 字段里。


复算任一摘要：

```bash
setup/.venv-win/Scripts/python.exe -c "
import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8'))
print(json.dumps(d.get('arms') or d.get('summary'), ensure_ascii=False, indent=2))
" setup/evidence/compare_lambda0.05_gsm8k50.json
```

## 结果一览（GSM8K 前 50 题，三臂同一批题，远端 `deepseek-v4-pro`）

| λ | 臂 | 准确率 | offload 率 | 远端 token |
|---|---|---|---|---|
| λ=0.05 | 纯小模型下限 | 4.00% (2/50) | 96.0% | 0 |
| λ=0.05 | **PyroDash** | **98.00% (49/50)** | 96.0% | 41,916 |
| λ=0.05 | 纯大模型上限 | 96.00% (48/50) | — | 33,366 |
| λ=0.6 | 纯小模型下限 | 90.00% (45/50) | 2.0% | 0 |
| λ=0.6 | **PyroDash** | **90.00% (45/50)** | **0.0%** | **0** |
| λ=0.6 | 纯大模型上限 | 96.00% (48/50) | — | 33,063 |

分析见 `../STATUS.md` 的「对照实验结果」。

---

## 结果二：HumanEval 164 题 + offload 旋钮扫描（`local_relay/` 中转）

远端 `deepseek-v4-flash` + **关思考**（开思考时 2048 预算全烧在 reasoning 上、正文被截断）。
题数 164，`max_tokens=2048`，`temperature=0`，pass@1。

| 配置 | pass@1 | 交接率 | 云端 token 占比 | 均耗时 |
|---|---|---|---|---|
| `small` 纯本机 Qwen3-4B | 82.9% (136) | — | 0% | 1.94s |
| `cloud` 纯云端 flash | 95.7% (157) | — | 100% | 1.53s |
| **默认策略 relay** | **96.3% (158)** | 85.4% | 89.2% | 1.61s |

旋钮扫描（用 `pyrodash_task=routine` 强制走小模型路径，隔离出变量）：

| `SMALL_MAX_TOKENS` | pass@1 | 交接率 | 云端 token 占比 |
|---|---|---|---|
| 128 | **90.9%** | 53.0% | 47.9% |
| 256 | 87.2% | 9.1% | 14.5% |
| 512（默认） | 84.8% | 1.2% | 3.6% |
| 1024 | 84.8% | 0.0% | **0.0%** |

关键结论：

1. **默认策略下代码题不进小模型** —— HumanEval 载荷中位 563 字符，`ANALYSIS_MIN_CHARS=400`
   把 140/164（85%）判成 `analysis` 直接交云端。所以代码题上默认策略是**拿省钱换正确率**，
   而不是两全：准确率追平纯云端（96.3% vs 95.7%），但云端占比高达 89.2%。
2. **这个优化值 +11.5 pp** —— 强制小模型先答（组 B）只有 84.8%，默认策略 96.3%。
3. **`SMALL_MAX_TOKENS` 是单调的质量/成本旋钮**：越小越准、越费云端 token。
4. **给 4B 更多预算不能提高它的正确率** —— 走本机的题从 77 增到 164 道，
   通过率始终卡在 85–88%；多给预算只是让它写更长。且「交接后」的通过率始终 ≥「本机搞定」。

完整表与分析见 `../../local_relay/README.md` §5.3。

---

## 结果三：基座替换评估 —— MiniCPM5-2B 值不值（`base_model_eval_minicpm5_2b.json`）

同题集 164 道、同判分，只换小模型腿的基座：

| 配置 | 纯本机 pass@1 | 中转 pass@1 | 云端 token 占比 | 中转均耗时 |
|---|---|---|---|---|
| Qwen3-4B-Instruct-2507（现役） | **82.9%** | **96.3%** | **89.2%** | 1.61s |
| MiniCPM5-2B Q4_K_M（关思考） | 78.7% | 93.9% | 90.5% | **1.57s** |

路由分布两者完全相同（`handoff:analysis` 140 / `small` 24），差异全部来自小模型真正经手的 24 道：
**Qwen3-4B 24/24（100%）vs MiniCPM5-2B 21/24（87.5%）**。

另记录了一个会**直接废掉中转**的坑（`reasoning_trap` 字段）：
MiniCPM5-2B 模板**默认开思考**，同一批 16 题开思考 **8/16、均耗时 10.67s、输出 557～2048 token**，
关思考 **16/16、均耗时 1.06s、输出 82～223 token**。开思考时全部超过 `SMALL_MAX_TOKENS=512`
→ 在 relay 里 100% 命中 `stop_type=='limit'` → 每道题都交给云端 → **中转退化成纯代理、本地零节省**。

> ⚠️ 噪声提醒：n=164 时单臂 95%CI ±3.0pp、两臂差 ±4.2pp。-2.4pp 在噪声内，
> 严格表述是「**未观察到优势**」，而非「显著更差」。

完整评估（推理型模型兼容性、评估协议、7 项局限、复现命令）见
`../../local_relay/BASE_MODEL_EVAL.md`。
