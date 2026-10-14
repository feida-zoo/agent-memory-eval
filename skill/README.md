# agent-memory-eval skill：源码与打包

本目录是 `agent-memory-eval` skill 的**源码与打包脚本**所在处。分发的 `.skill` 包是一个 zip，
顶层目录固定为 `agent-memory-eval/`。

## 目录构成

| 路径 | 说明 |
|---|---|
| `src/` | **skill 源**：`SKILL.md`、`references/`、`scripts/`、`output/.gitkeep`、`user_dataset/.gitkeep`。打包时原样拷入 |
| `build_skill.py` | 打包脚本（纯标准库，Python 3.9+ 可跑） |
| `agent-memory-eval.skill` | 打包产物（可分发 zip） |
| `../membench/` | 评测工程本体，**从仓库取**（不在本目录维护） |
| `../docs/` | 《Agent 记忆系统深度研究报告》及配图，**从仓库取** |

即：skill 源只维护 `src/`；`membench/` 与 `docs/` 都来自仓库其他目录，打包时一并组装进去。

## 打包

```bash
# 在仓库根目录执行
python skill/build_skill.py            # 默认 --repo-root . --out skill/agent-memory-eval.skill
python skill/build_skill.py --list     # 只列出将被打包的文件（含排除项提示），不写盘
python skill/build_skill.py --out /tmp/agent-memory-eval.skill
```

脚本只用标准库，系统 `python3` 即可运行。

### 组装规则

- `skill/src/**` → `agent-memory-eval/**`（原样，含 `.gitkeep`）；
- `<repo>/membench/` → `agent-memory-eval/membench/`（按下方排除策略裁剪）；
- `<repo>/docs/` → `agent-memory-eval/docs/`。

### 打包后**不该出现**的文件

| 类别 | 具体 |
|---|---|
| 按需下载的数据集 | `locomo.jsonl`、`longmemeval_*.jsonl`、`gatemem_office.jsonl`、`beam_*.jsonl`（用 `dataset fetch` 按需下载，默认落 `~/.membench/datasets/`；体积巨大，仓库 `.gitignore` 约定不入库） |
| 构建缓存 | `*.egg-info/`、`__pycache__/`、`*.pyc` |
| 运行期产物 | `runs/`、`output/` |
| 系统垃圾 | `.DS_Store` |

内置数据集中只有 `office_dataset.jsonl` 与 `sample_dataset.jsonl` 会进包。

> 打包脚本带**默认自检**：写完 zip 后复查包内是否出现被排除的数据集文件名，出现即报错退出并删掉产物，
> 防止"又把数据集打进去"的回归（历史上曾因此产出 83MB 的包）。

### 可复现构建

zip 内路径统一用正斜杠、UTF-8 文件名；条目按路径排序、时间戳固定为 1980-01-01，
因此同一份源码每次打出的包**字节一致**。

## 包结构（顶层目录名固定 `agent-memory-eval/`）

```
agent-memory-eval/
├── SKILL.md                 # skill 主文档（工作流与纪律）
├── references/              # adapter_guide / compare_workflows / optimize_workflow / http_bridge_guide / user_dataset_guide
├── scripts/                 # eval_two_commits.sh 等模板脚本
├── output/.gitkeep          # 评测报告输出目录（运行期写入）
├── user_dataset/.gitkeep    # 用户专属测试集目录（本地，不入库）
├── membench/                # 评测工程（CLI + 内置集 + 测试）
└── docs/                    # 深度研究报告及配图
```
