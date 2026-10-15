# AGENTS.md — agent-memory-eval 工程指引

面向在本仓库工作的 AI agent 与工程师。与 `.kimi-code/AGENTS.md`（通用质量要求：实事求是、区分事实与假设、信息不足如实说明）叠加生效。

## 这是什么

membench 是 **Agent 记忆系统评测框架**（Python 包 + CLI）：对被测记忆系统跑多会话数据集，产出六维指标报告（run）、多数据集宏平均全量评估（evaluate）、A/B 回归门禁（compare）、平台×commit 矩阵（matrix）。仓库同时维护一个 **agent skill 包**（`skill/agent-memory-eval.skill`），把 membench 与工作流封装给 agent 使用。

## 目录结构

```
membench/                 # 评测工程（Python 包工程根）
├── membench/             # 包目录
│   ├── adapters/         # BaseMemoryAdapter + mock / transcript / http
│   ├── data/             # schema / 加载器 / 转换器 / fetch（按需下载）/ info（数据集注册表）
│   ├── judges/           # heuristic / llm（OpenAI 兼容）/ client（复用 agent 的 LLM）
│   ├── metrics/          # token_f1 / bleu1 / recall@k + 六维汇总
│   ├── paths.py          # 【唯一】默认产物目录判定（output_root / runs_dir）
│   ├── layout.py         # 【唯一】实验目录命名与 materialize（eval_* / ab_* / matrix_*）
│   ├── agent_loop.py     # tool-call 循环复用（run_tool_loop / AgentLoopAdapter）
│   ├── doctor.py         # 三端点连通性与契约校验
│   ├── runner.py / evaluate.py / compare.py / matrix.py / fuse.py / report.py
│   ├── userdata.py       # 用户测试集 install / status / delete / resolve
│   └── cli.py            # CLI 入口
├── tests/                # pytest
├── examples/             # quickstart.py / optimize_loop.md / http_bridge_node（Node 模板）
└── README.md             # 工程使用文档
skill/
├── src/                  # skill 源：SKILL.md、references/、scripts/、空目录占位
├── build_skill.py        # 打包脚本（组装 src + membench + docs → .skill）
└── agent-memory-eval.skill   # 打包产物（提交进仓库）
docs/                     # 深度研究报告 + 配图
```

## 环境与自测（提交前必须做）

- 要求 Python **>= 3.10**（见 `membench/pyproject.toml`）。运行时仅标准库；`matplotlib`（出图/报告）、`requests`（HttpAdapter）为可选依赖。
- 建环境：

  ```bash
  python3 -m venv .venv && . .venv/bin/activate
  pip install -e membench pytest matplotlib requests
  ```

- 跑测试（提交前必须全绿）：

  ```bash
  cd membench && python -m pytest tests/ -q
  ```

  改动涉及面较广时不要只跑子集。
- **坑**：macOS 自带 `/usr/bin/python3` 可能低于 3.10 且缺 matplotlib，会让 report/图表相关测试大面积失败（`RuntimeError: 报告图表需要 matplotlib`）。这类失败先确认解释器与依赖，再怀疑代码。
- 测试不得在仓库内写产物：涉及默认目录的测试用 `monkeypatch.setenv("MEMBENCH_OUTPUT_DIR", str(tmp_path / ...))`。

## 核心约定（改代码前必读）

1. **一次实验一个自包含子目录**（报告与对应 run 就近归属，不再全局扁平化）：
   - `output/eval_<adapter>_<platform>_<ts>/`：`report.md`、`metrics.json`、`fig_*.png` + `runs/<run_id>/`
   - `output/ab_<tag>_<ts>/`：`A/`、`B/`（各含 `report.md`/`metrics.json`/`fig_*.png`/`runs/<run_id>/`）+ `compare_report.md`
   - `output/matrix_<ts>/`：报告 + `runs/<label>/`
   - CLI `--report-out DIR` 一律指**实验目录的父目录**。
   - 命名与落位**只走** `membench/layout.py`，不要在业务代码里手写目录名或手抄复制逻辑。
2. **默认目录只由 `membench/paths.py` 判定**（`output_root()` / `runs_dir()`；`MEMBENCH_OUTPUT_DIR` > skill 的 `output/` > 工程根）。历史上的多处 SKILL.md 探测已收敛到这里，不要再复制一份。
3. **可比性前置校验**：`compare` 按 `membench/compare.py` 的维度表核对 A/B 元数据（关键维度不一致 → 不可比）。新增关键维度要同步三处：维度表、`Runner` 的元数据写入、文档。**`commit` 不同是 A/B 的被测变量，不得判为不可比。** 纪律：compare 只测量差异，不判定两臂是否只差一个变量。
4. **数据集政策**：仓库**不内置**开源大数据集。`office_dataset.jsonl` / `sample_dataset.jsonl` 是自造演示集；`locomo` / `longmemeval_*` / `gatemem_office` / `beam_*` 一律 `dataset fetch` 到 `~/.membench/datasets/`。
   - `membench/.gitignore` 已忽略这些文件：**绝不**加入 git，也绝不让它们进入 skill 包（`skill/build_skill.py` 自带自检，混入即报错）。
5. **改 CLI 要三处同步**：`cli.py` 的参数定义、`cli.py` 顶部 docstring 用法、`membench/README.md`；`skill/src/SKILL.md` 里有命令示例时同样同步。
6. **`--shards` 的分片维度**：默认按题目分片（`i % N == shard_index`）。对「几乎每个对话都有题」的 ingest 密集型数据集（office / BEAM-1M），题目级分片会让每片仍对全部对话 ingest（成本 ×N）——用 `--shard-by dialogue`（整段对话归片、分片间不重复 ingest），再用 `fuse` 融合。两种维度都能融合，但**混维融合会被 fuse 拒绝**。

## skill 打包

- `skill/src/` 是 skill 源；`membench/` 与 `docs/` 在打包时从仓库取。
- 改了 `membench/**` 或 `skill/src/**` 后**必须**重建并提交产物：

  ```bash
  python3 skill/build_skill.py           # 生成 skill/agent-memory-eval.skill
  python3 skill/build_skill.py --list    # 只看将打包/被排除的文件
  ```

- 打包排除：`*.egg-info`、`__pycache__`、`.pytest_cache`、`runs/`、`output/`、`*.pyc`、`.DS_Store`，以及按需下载的数据集。历史上曾把 268MB 的 `longmemeval_s.jsonl` 打进包（83MB 的事故），自检逻辑就是为此存在。

## 代码风格

- 中文 docstring；类型注解；`from __future__ import annotations`；纯标准库优先。
- 注释只写非显然的"为什么"，不复述代码；改行为时同步更新相关 docstring 与文档。
- 可选依赖（matplotlib / requests）在 import 处兜底并给出可操作的报错，不要漏成裸 `ImportError` 堆栈。

## 提交前 checklist

- [ ] `cd membench && python -m pytest tests/ -q` 全绿
- [ ] 改了 `membench/**` 或 `skill/src/**` → 已重跑 `python3 skill/build_skill.py`
- [ ] `git status` 无大数据集、`output/`、`runs/`、`__pycache__`
- [ ] 新增命令/参数已写进 `membench/README.md`（必要时 `skill/src/SKILL.md`）
