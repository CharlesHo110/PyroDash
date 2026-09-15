# 验证证据（对应用户计划 §4 验证清单）

这些 JSON 是脚本真实跑出来的原始输出，可直接复核，不是手写数字。

| 文件 | 对应条目 | 命令 |
|---|---|---|
| `smoke_gsm8k3.json` | §4 项 2 | `bash setup/06-run-windows-native.sh 3` |
| `compare_lambda0.05_gsm8k50.json` | §4 项 3 | `bash setup/06-run-windows-native.sh --compare 50 --tag l05` |
| `compare_lambda0.6_gsm8k50.json` | §4 项 3 / 项 4 | `PYRODASH_MODEL=models/PyroDash-4B-GRPO-Lambda-0.6 bash setup/06-run-windows-native.sh --compare 50 --tag l06` |

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
