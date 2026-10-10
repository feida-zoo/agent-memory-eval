# agent-memory-eval

Agent 记忆系统评测工程：为多平台个人办公 Agent 提供可归因、可回归、可进 CI 的记忆质量评测。

## 仓库结构

```
membench/          # 评测工程（Python 包，CLI 驱动）
skill/             # agent-memory-eval skill 打包产物（.skill 文件）
docs/              # 深度研究报告与方法论文档
```

## 快速开始

```bash
cd membench
pip install -e .
python -m pytest tests/          # 240 项测试
python -m membench.cli run --adapter mock-strong --dataset membench/data/sample_dataset.jsonl
```

## 数据集

仓库**不内置**开源数据集，按需下载。`dataset fetch` 目前只支持 BEAM，其余数据集请从下列来源手动获取后用转换器转成统一格式：

```bash
python -m membench.cli dataset list              # 查看全部数据集与状态
python -m membench.cli dataset fetch beam_1m     # BEAM 1M 精简集
python -m membench.cli dataset fetch beam_10m    # BEAM 10M 全集
```

| 数据集 | 来源 | 获取方式 |
|---|---|---|
| BEAM | [github.com/mohammadtavakoli78/BEAM](https://github.com/mohammadtavakoli78/BEAM)（arXiv:2510.27246） | `dataset fetch` |
| LoCoMo | [github.com/snap-research/locomo](https://github.com/snap-research/locomo)，原始文件 `data/locomo10.json` | 手动下载 → `convert_locomo.py` |
| LongMemEval | [huggingface.co/datasets/xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) | 手动下载 → `convert_longmemeval.py` |
| GateMem | [github.com/rzhub/GateMem](https://github.com/rzhub/GateMem)（arXiv:2606.18829）/ [Hugging Face](https://huggingface.co/datasets/Ray368/GateMem) | 手动下载 → `convert_gatemem.py` |

内置的两个小数据集（`office_dataset.jsonl` / `sample_dataset.jsonl`）是自造的演示/冒烟数据，非开源数据集。

## 文档

- `membench/README.md` — 工程使用文档（适配器、命令、指标、门禁）
- `docs/Agent记忆系统深度研究报告.md` — 记忆系统评测方法论深度研究
