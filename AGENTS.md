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
│   ├── paths.py          # 【唯一】默认产物根目录判定（output_root）
│   ├── layout.py         # 【唯一】任务目录 / 实验目录命名与路径 helper（task_subdir / init_task_dir / arm_dir / eval_* / compare_* / matrix_* / 分片账本 / resolve_dataset_dir）
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
- **坑**：macOS 自带 `/usr/bin/python3` 可能低于 3.10 且缺 matplotlib，会让 report/图表相关测试大面积失败（`RuntimeError: 报告图表需要 matplotlib`）。这类失败先确认解释器与依赖，再怀疑代码。本仓库开发机上验证过的解释器是 `/opt/homebrew/bin/python3.13`（已装 matplotlib），可直接用它替换上面的 `python`。
- 测试不得在仓库内写产物：涉及默认目录的测试用 `monkeypatch.setenv("MEMBENCH_OUTPUT_DIR", str(tmp_path / ...))`。

## 核心约定（改代码前必读）

1. **一次实验一个自包含任务目录**（报告与对应数据就近归属）：一次评测 = 一个**任务目录**，单臂任务（`arms=1`）就是 `eval_<name>_<platform>_<ts>`（自身即完整产物根），双臂 A/B 任务（`arms=2`）是 `compare_<name>_<platform>_<ts>`，`A/` 与 `B/` 是它下面的两个臂（**各为一个完整的单臂产物根**，含自己的 `runs/` 与 `reports/`）。骨架由 `layout.init_task_dir` 唯一决定（`task init --name NAME [--platform P] [--arms 1|2]`，stdout 只打印任务目录绝对路径）；`run`/`evaluate`/`evaluate-compare`/`compare`/`fuse` 都可用 `--task-dir DIR` **就地**写进/读已存在的任务目录（与 `--report-out` 互斥、不新建目录），`task status DIR` 是只读进度视图（骨架 / 分片账本 / 快照 / 下一步）：
   - `output/eval_<name>_<platform>_<ts>[_s<i>][_l<limit>]/`（`ts` = `YYYYMMDD-HHMMSS`；`_s<i>` = `--shards` 的分片序号、`_l<limit>` = `--limit`，无分片无 limit 时不带后缀，词表见 `layout.experiment_tags`）：
     - `runs/<dataset>/report/`：数据集级 `report.md`、`metrics.json`、`results.jsonl`、`fig_*.png`
     - `runs/<dataset>/shards.json`：**分片账本**（分片评测才有；进度与归属的唯一记录，读 / 写 / 校验见 `layout.read_shard_ledger` / `write_shard_ledger` / `check_shard_ledger`，`fuse` 与 `task status` 都读它）
     - `runs/<dataset>/<dialogue_id>/`：对话级 `runtime/`（桥的 reset 前备份落 `runtime/reset_backup/`；`MEMBENCH_RUN_DIR` 指本对话目录）、`memory_snapshot/`（**默认导出**，含框架写的 `manifest.json`；适配器未声明快照能力时警告一次并跳过，`metrics["state_snapshot"]["role"] == "unsupported"`）、`report/`（`report.md`、`metrics.json`、`results.jsonl`，不出图；含 `../memory_snapshot/` 的可点击链接与清单条目数 / 体积）
     - `reports/`：任务级报告（`report.md`、`metrics.json`、`evaluate_summary.json`、`fig_dataset_overview.png`——`evaluate_summary.json` **无条件产出**，不区分单/多数据集）
   - `output/compare_<name>_<platform>_<ts>/`：`A/`、`B/`（`layout.DIR_ARM_A` / `DIR_ARM_B`；各为完整 eval 实验形态，含自己的 `runs/` 与 `reports/`）+ `reports/{comparison.md,fig_*.png}`
   - `output/matrix_<ts>/`：`reports/{report.md,matrix_result.json,fig_matrix_accuracy.png}` + `runs/<label>/`
   - 分片（`shards>1`）**只写本片负责的对话目录 + 账本** `runs/<dataset>/shards.json`，不写数据集级 `report/` 与任务级 `reports/`（分片产物不是"完整实验"，一律由 `fuse` 补齐：`--task-dir DIR` 就地补齐，或 `--runs ... --out ...` 跨目录融合）；多片共用一个任务目录时**只允许 `--shard-by dialogue`**（题目级分片会让多片写同一对话目录互相覆盖），同一分片重复跑直接拒绝。
   - CLI `--report-out DIR` 一律指**实验目录的父目录**（缺省 `output_root()`）；`fuse --out` / `matrix --out` 指**完整实验目录**，`--task-dir` 指已存在的任务目录（或其臂目录）。
   - 「产物目录」入参（`--baseline` / `--candidate` / `--from-runs` / `--baseline-run` / `--candidate-run` / `fuse --runs` 各项 / `matrix --runs` 的 `=` 右值 / `--replay` / `--replay-verify-source`）统一接受**实验目录**或 `runs/<dataset>` 数据集目录，经 `layout.resolve_dataset_dir` 解析。`--from-runs` 的 `KEY` 另按数据集注册表核对身份（`office` ↔ `office_dataset` / `user_dataset`），key 认错直接报错——`resolve_dataset_dir` 的"提示不命中回退到唯一数据集目录"只服务真正的自定义 key。
   - 旧版产物（平铺 run 目录、缺 `report/` 层级、旧版对比实验目录）一律报错拒绝，不留兼容层；目录里**根本没有**数据集产物（空目录、跑到一半中断）走另一条文案（`layout.NO_DATASET_PRODUCT_MSG`），别把"没有产物"说成"版本旧"。命名与落位**只走** `membench/layout.py`，不要在业务代码里手写目录名或手抄复制逻辑；`run_id` 一律经 `layout.run_id` 组装（词表 `layout.run_id_tags`）。
   - CLI 各命令在创建实验目录后失败时要把**本次新建且仍为空**的目录清掉（`cli._no_leftover_experiment` / `_discard_empty_experiment`）：`Runner` 层"失败不留半成品"的约定不能被 CLI 的提前建目录破坏；有任何产物的目录保留现场。
2. **默认根目录只由 `membench/paths.py` 判定**（`output_root()`；`MEMBENCH_OUTPUT_DIR` > skill 的 `output/` > 工程根）。历史上的多处 SKILL.md 探测已收敛到这里，不要再复制一份。
3. **可比性前置校验**：`compare` 按 `membench/compare.py` 的维度表核对 A/B 元数据（关键维度不一致 → 不可比）。新增关键维度要同步三处：维度表、`Runner` 的元数据写入、文档。**`commit` 不同是 A/B 的被测变量，不得判为不可比。** 纪律：compare 只测量差异，不判定两组是否只差一个变量。
4. **数据集政策**：仓库**不内置**开源大数据集。`office_dataset.jsonl` / `sample_dataset.jsonl` 是自造演示集；`locomo` / `longmemeval_*` / `gatemem_office` / `beam_*` 一律 `dataset fetch` 到 `~/.membench/datasets/`。
   - `membench/.gitignore` 已忽略这些文件：**绝不**加入 git，也绝不让它们进入 skill 包（`skill/build_skill.py` 自带自检，混入即报错）。
5. **改 CLI 要三处同步**：`cli.py` 的参数定义、`cli.py` 顶部 docstring 用法、`membench/README.md`；`skill/src/SKILL.md` 里有命令示例时同样同步。
6. **`--shards` 的分片维度**：默认按题目分片（`i % N == shard_index`）。对「几乎每个对话都有题」的 ingest 密集型数据集（office / BEAM-1M），题目级分片会让每片仍对全部对话 ingest（成本 ×N）——用 `--shard-by dialogue`（整段对话归片、分片间不重复 ingest），再用 `fuse` 融合。两种维度都能融合，但**混维融合会被 fuse 拒绝**。**多个分片共用同一个任务目录（`--task-dir`）时只允许 `--shard-by dialogue`**（题目级分片会让多片写同一对话目录并互相覆盖），题目级分片请各片各落一个实验目录再跨目录 `fuse`。
7. **记忆快照默认启用**：每个对话在**全部 session ingest 完成后、第一道 query 之前**导出唯一一份到 `<对话目录>/memory_snapshot/`（框架写 `manifest.json`，适配器自己往该目录写内容；一个对话只导这一份，per-session 导出是没用的中间态）。适配器未声明快照能力时**警告一次并跳过**，`metrics["state_snapshot"]["role"] == "unsupported"`（此时 `manifest_sha256` / `format_version` 为 `null`，报告渲染成「不支持（适配器未声明快照能力）」而不是空值）；`--no-snapshot` 关闭（`--snapshot` 已删，不留兼容层）；回放模式自动不导出。对话级报告里给出 `../memory_snapshot/` 的可点击相对链接与清单条目数 / 体积——「跑完 ingest 就能翻记忆文件」是这个特性的第一价值，别把它藏进日志。

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
