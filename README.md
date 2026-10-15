# agent-memory-eval

Agent 记忆系统评测工程：为多平台个人办公 Agent 提供可归因、可回归、可进 CI 的记忆质量评测。

## 仓库结构

```
membench/                    # 评测工程（Python 包，CLI 驱动）
skill/
├── src/                     # skill 源（SKILL.md / references / scripts / output / user_dataset）
├── build_skill.py           # 打包脚本：组装 src/ + membench/ + docs/ 为 .skill
└── agent-memory-eval.skill  # 打包产物（可分发 zip，顶层目录 agent-memory-eval/）
docs/                        # 深度研究报告与方法论文档（1 篇研究报告 + 3 张配图）
```

其中 `skill/` 的源码只维护 `src/`；`membench/` 与 `docs/` 都从仓库其他目录取，打包时一并组装进 `.skill`。

## 快速开始

```bash
cd membench
pip install -e .                 # 只装包本身（核心 CLI 纯标准库）
python -m pytest tests/          # 全量回归（358 项）
python -m membench.cli run --adapter mock-strong --dataset membench/data/sample_dataset.jsonl
```

依赖说明：`pip install -e .` **不装任何第三方库**——核心 CLI、六维指标、compare 门禁、doctor 自检全部基于标准库。可选 extra：

- `pip install -e ".[report]"`：matplotlib（报告配图，懒加载；缺它时 markdown 报告仍生成、仅不出图）；
- `pip install -e ".[http]"`：requests（HttpAdapter 真实调用被测 HTTP 服务）。

跑测试需要 `pytest` + `matplotlib` + `requests`，可 `pip install -e ".[dev]"`（等价于 `pip install pytest matplotlib requests`）。

## 数据集

仓库**内置**两个自造的小数据集（`office_dataset.jsonl` / `sample_dataset.jsonl`，分别用于业务验收与管线连通冒烟），**不内置**任何开源大数据集——其余全部按需下载（`dataset fetch`）后自动转换，默认落 `~/.membench/datasets/`（可用 `--out` 指定目录），需要能访问 GitHub / HuggingFace：

```bash
python -m membench.cli dataset info                      # 查看全部支持的数据集与状态（共 9 个）
python -m membench.cli dataset fetch locomo              # LoCoMo 10 个超长对话，1986 题
python -m membench.cli dataset fetch longmemeval_oracle  # LongMemEval oracle 版 ~5MB
python -m membench.cli dataset fetch longmemeval_s       # LongMemEval s 版 ~250MB
python -m membench.cli dataset fetch gatemem_office      # GateMem 办公三维
python -m membench.cli dataset fetch beam_1m             # BEAM 1M 精简集
python -m membench.cli dataset fetch beam_10m            # BEAM 10M 全集
```

| 数据集 | 来源 |
|---|---|
| BEAM | [GitHub](https://github.com/mohammadtavakoli78/BEAM) · [arXiv:2510.27246](https://arxiv.org/abs/2510.27246) |
| LoCoMo | [GitHub](https://github.com/snap-research/locomo) |
| LongMemEval | [HuggingFace](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) |
| GateMem | [GitHub](https://github.com/rzhub/GateMem) · [arXiv:2606.18829](https://arxiv.org/abs/2606.18829) · [HuggingFace](https://huggingface.co/datasets/Ray368/GateMem) |

内置的两个小数据集（`office_dataset.jsonl` / `sample_dataset.jsonl`）是自造的演示/冒烟数据，非开源数据集。

## 文档

- `membench/README.md` — 工程使用文档（适配器、命令、指标、门禁、产物布局、A/B 可比性校验）
- `docs/Agent记忆系统深度研究报告.md` — 记忆系统评测方法论深度研究
- `docs/benchmark_radar.png`、`docs/locomo_scores.png`、`docs/memory_timeline.png` — 研究报告配图
- `skill/README.md` — skill 源码结构与打包说明
