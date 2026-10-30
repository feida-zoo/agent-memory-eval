# AGENTS.md — agent-memory-eval 工程指引

面向在本仓库工作的 AI agent 与工程师。与 `.kimi-code/AGENTS.md`（通用质量要求：实事求是、区分事实与假设、信息不足如实说明）叠加生效。

## 这是什么

membench 是 **Agent 记忆系统评测框架**（Python 包 + CLI）：对被测记忆系统跑多会话数据集，产出六维指标报告（run）、多数据集全量评估（evaluate：基准集按题数加权宏平均、办公域槽位单列业务结论）、A/B 回归门禁（compare）、平台×commit 矩阵（matrix）、多线逐步对比（series：N 个版本按时间顺序对比，线名从 commit 信息提取）。仓库同时维护一个 **agent skill 包**（`skill/agent-memory-eval.skill`），把 membench 与工作流封装给 agent 使用。

## 目录结构

```
membench/                 # 评测工程（Python 包工程根）
├── membench/             # 包目录
│   ├── adapters/         # BaseMemoryAdapter + mock / transcript / http
│   ├── data/             # schema / 加载器 / 转换器 / fetch（按需下载）/ info（数据集注册表）
│   ├── judges/           # heuristic / llm（OpenAI 兼容）/ client（复用 agent 的 LLM）
│   ├── metrics/          # token_f1 / bleu1 / recall@k + 六维汇总
│   ├── paths.py          # 【唯一】默认产物根目录判定（output_root）
│   ├── layout.py         # 【唯一】任务目录 / 实验目录命名与路径 helper（task_subdir / init_task_dir / arm_dir / eval_* / compare_* / matrix_* / 分片账本 / trace_state 路径 / resolve_dataset_dir）
│   ├── agent_loop.py     # tool-call 循环复用（run_tool_loop / AgentLoopAdapter）
│   ├── doctor.py         # 三端点连通性与契约校验
│   ├── runner.py / evaluate.py / compare.py / matrix.py / series.py / fuse.py / report.py
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
     - `runs/<dataset>/shards.<i>.json`（逐片条目，真源）+ `runs/<dataset>/shards.json`（物化视图）：**分片账本**（分片评测才有；进度与归属的记录，读 / 写 / 校验见 `layout.read_shard_ledger` / `update_shard_ledger` / `check_shard_ledger`，`fuse` 与 `task status` 都经 `read_shard_ledger` 读合并结果）
     - `runs/<dataset>/<dialogue_id>/`：对话级 `runtime/`（归适配器使用的运行时文件，空目录也建；`MEMBENCH_RUN_DIR` 指本对话目录；框架不内置 reset 前备份）、`memory_snapshot/`（**默认导出**，含框架写的 `manifest.json`；适配器未声明快照能力时警告一次并跳过，`metrics["state_snapshot"]["role"] == "unsupported"`）、`trace_state/t<as_of>/`（`run --trace-state`：interleaved 的每个有提问时点一份，含 `manifest.json`；sequence 对话不写，归因读末尾 `memory_snapshot/`）、`report/`（`report.md`、`metrics.json`、`results.jsonl`，不出图；含 `../memory_snapshot/` 的可点击链接与清单条目数 / 体积）
     - `reports/`：任务级报告（`report.md`、`metrics.json`、`evaluate_summary.json`、`fig_dataset_overview.png`——`evaluate_summary.json` **无条件产出**，不区分单/多数据集）
   - `output/compare_<name>_<platform>_<ts>/`：`A/`、`B/`（`layout.DIR_ARM_A` / `DIR_ARM_B`；各为完整 eval 实验形态，含自己的 `runs/` 与 `reports/`）+ `reports/{comparison.md,fig_*.png}`
   - `output/matrix_<ts>/`：`reports/{report.md,matrix_result.json,fig_matrix_accuracy.png}` + `runs/<label>/`
   - `output/series_<ts>/`：多线对比（`series`，只读产物）——`reports/{report.md,fig_series_accuracy.png}` + `runs/<线序>-<slug>/<dataset>/`（各线的数据集树副本）。`--line <commit-or-name>=<DIR>` 按时间顺序重复给出，第一条默认叫「基线」，其余线名从 `git -C <repo> log -1 --format=%s` 的 subject 提取（去 conventional 前缀、剥首尾标点、截断 24 字，取不到回退短 sha 并告警）；多数据集线按同一套融合口径（office 不参与）并给逐数据集表；任一线的门禁 FAIL → 退出码 1。
   - 分片（`shards>1`）**只写本片负责的对话目录 + 账本** `runs/<dataset>/shards.<i>.json`（逐片原子写；`shards.json` 是物化视图），不写数据集级 `report/` 与任务级 `reports/`（分片产物不是"完整实验"，一律由 `fuse` 补齐：`--task-dir DIR` 就地补齐，或 `--runs ... --out ...` 跨目录融合）；多片共用一个任务目录时**只允许 `--shard-by dialogue`**（题目级分片会让多片写同一对话目录互相覆盖），同一分片重复跑直接拒绝。
   - CLI `--report-out DIR` 一律指**实验目录的父目录**（缺省 `output_root()`）；`fuse --out` / `matrix --out` 指**完整实验目录**，`--task-dir` 指已存在的任务目录（或其臂目录）。
   - 「产物目录」入参（`--baseline` / `--candidate` / `--from-runs` / `--baseline-run` / `--candidate-run` / `fuse --runs` 各项 / `matrix --runs` 的 `=` 右值 / `--replay` / `--replay-verify-source`）统一接受**实验目录**或 `runs/<dataset>` 数据集目录，经 `layout.resolve_dataset_dir` 解析。`--from-runs` 的 `KEY` 另按数据集注册表核对身份（`office` ↔ `office_dataset` / `user_dataset`），key 认错直接报错——`resolve_dataset_dir` 的"提示不命中回退到唯一数据集目录"只服务真正的自定义 key。
   - 旧版产物（平铺 run 目录、缺 `report/` 层级、旧版对比实验目录）一律报错拒绝，不留兼容层；目录里**根本没有**数据集产物（空目录、跑到一半中断）走另一条文案（`layout.NO_DATASET_PRODUCT_MSG`），别把"没有产物"说成"版本旧"。命名与落位**只走** `membench/layout.py`，不要在业务代码里手写目录名或手抄复制逻辑；`run_id` 一律经 `layout.run_id` 组装（词表 `layout.run_id_tags`）。
   - CLI 各命令在创建实验目录后失败时要把**本次新建且仍为空**的目录清掉（`cli._no_leftover_experiment` / `_discard_empty_experiment`）：`Runner` 层"失败不留半成品"的约定不能被 CLI 的提前建目录破坏；有任何产物的目录保留现场。
   - **每个实验目录（含作为命令容器的臂目录）自带 `runtime/`**（`commands.jsonl` + `logs/` + `env.json`，与 `runs/`、`reports/` 平级；记录命令、脱敏后的参数、起止 / 耗时 / 退出码、环境快照与全量日志）——排查"这次到底怎么跑的"先看它；它与 `runs/<dataset>/<对话>/runtime/`（对话级运行时数据）不是一回事。
2. **默认根目录只由 `membench/paths.py` 判定**（`output_root()`；`MEMBENCH_OUTPUT_DIR` > skill 的 `output/` > 工程根）。历史上的多处 SKILL.md 探测已收敛到这里，不要再复制一份。
3. **可比性前置校验**：`compare` 按 `membench/compare.py` 的维度表核对 A/B 元数据（关键维度不一致 → 不可比）。新增关键维度要同步三处：维度表、`Runner` 的元数据写入、文档。**`commit` 不同是 A/B 的被测变量，不得判为不可比。** 纪律：compare 只测量差异，不判定两组是否只差一个变量。
4. **数据集政策**：仓库**不内置**开源大数据集。`office_dataset.jsonl` / `sample_dataset.jsonl` 是自造演示集；`locomo` / `longmemeval_*` / `gatemem_office` / `beam_*` 一律 `dataset fetch` 到 `~/.membench/datasets/`。
   - `membench/.gitignore` 已忽略这些文件：**绝不**加入 git，也绝不让它们进入 skill 包（`skill/build_skill.py` 自带自检，混入即报错）。
5. **改 CLI 要三处同步**：`cli.py` 的参数定义、`cli.py` 顶部 docstring 用法、`membench/README.md`；`skill/src/SKILL.md` 里有命令示例时同样同步。
6. **`--shards` 的分片维度**：默认按题目分片（`i % N == shard_index`）。对「几乎每个对话都有题」的 ingest 密集型数据集（office / BEAM-1M），题目级分片会让每片仍对全部对话 ingest（成本 ×N）——用 `--shard-by dialogue`（整段对话归片、分片间不重复 ingest），再用 `fuse` 融合。两种维度都能融合，但**混维融合会被 fuse 拒绝**。**多个分片共用同一个任务目录（`--task-dir`）时只允许 `--shard-by dialogue`**（题目级分片会让多片写同一对话目录并互相覆盖），题目级分片请各片各落一个实验目录再跨目录 `fuse`。
7. **记忆快照默认启用**：每个对话在**全部 session ingest 完成后、第一道 query 之前**导出唯一一份到 `<对话目录>/memory_snapshot/`（框架写 `manifest.json`，适配器自己往该目录写内容；一个对话只导这一份，per-session 导出是没用的中间态）。适配器未声明快照能力时**警告一次并跳过**，`metrics["state_snapshot"]["role"] == "unsupported"`（此时 `manifest_sha256` / `format_version` 为 `null`，报告渲染成「不支持（适配器未声明快照能力）」而不是空值）；`--no-snapshot` 关闭（`--snapshot` 已删，不留兼容层）。对话级报告里给出 `../memory_snapshot/` 的可点击相对链接与清单条目数 / 体积——「跑完 ingest 就能翻记忆文件」是这个特性的第一价值，别把它藏进日志。
   **回放任务用任务级内容寻址快照池**：`task init --replay` 在任务根建 `snapshots/`；回放（`restore()`/`rebuild()` 完成后、第一道 query 之前）导出到临时目录 → 按内容 sha256 前 12 位定桶 → 桶落位只靠一次 `os.replace` 判"本次创建 / 别人已建好"（成功=新写；失败则读桶内 `manifest.json` 的 `entry.sha256`，一致就删临时目录记 `reused`，不一致即撞车/损坏直接 `SystemExit`）→ 原子写**逐对话条目** `snapshots/entries/<臂>/<数据集>/<对话>.json`（空臂键的目录名为 `_`；条目自足，含 digest / 状态哈希 / 字节数 / 文件数 / 身份 / 来源 / reused / run_id / written_at；同一对话重复写必须 digest 一致，否则报错）。`snapshots/index.json` 是**派生视图**（`buckets` / `entries` / `arms` 三段），真源是 `entries/` 树 + 各桶清单，由 `layout.build_snapshot_index` / `materialize_snapshot_index` 重算；`fuse --task-dir` 收尾刷新，`task status` 从真源重算并 best-effort 刷新。**一对话一文件 ⇒ 对话级分片的多个进程只写自己的对话，并行回放不需要锁，也不能在写入路径上读-改-写索引。** 池属于任务：`out_dir` 是 `<任务>/A|B` 时用父任务的池、臂键为目录名，单臂/普通实验根的臂键为 `""`；A/B 回放同一源时两臂指向同一批桶（只存一份），内容不同才各存一份——**差异本身就是回放不保真的证据**。`--no-snapshot` 回放不写桶也不写条目/索引。`--replay` 接受回放任务：优先读条目树（旧产物没有 `entries/` 才回退读索引），传臂目录按臂取桶，传任务根时池里只有一个臂才自动用它、多个臂报错要求改传具体臂目录；链式回放（回放任务再作源）必须可用。池的路径 / 条目 / 索引读写全部走 `layout` 的 `snapshot_pool_root` / `snapshot_arm_key` / `snapshot_entry_path` / `read_snapshot_entry` / `write_snapshot_entry` / `snapshot_bucket_dir` / `read_snapshot_index` / `write_snapshot_index` / `materialize_snapshot_index`，别在业务代码里手拼。
**版本标识（`membench/version.py` 是唯一规则来源）**：`metrics.membench_version = {version, hash, hash_source, stale_build}`。`hash` 在**打包时定稿**——`skill/build_skill.py` 先算包内 `*.py` 的 sha256 前 12 位，写进 `membench/membench/_build_hash.py`（该文件不参与哈希计算，否则自引用），再打包；运行时直接读常量。安装后的 skill 没有 `.git`，所以**不要**用 git commit 当版本标识。核对：`hash_source=build` 即读的是打包值；`computed` 表示开发态或打包后改过代码（`stale_build=true` 会提示重新打包）。run / fuse / evaluate / evaluate-compare 报告头部都显示它；`fuse` 记**当前**代码版本（指标是当前代码重算的）。改判分口径后跨版本分数不可直接比较，靠这个标识对账。

8. **摄入策略（`Dialogue.ingest_strategy`）与失败归因（`--trace-state`）**：`sequence`（默认）整段 ingest 后提问，`as_of` 不强制；`interleaved`（GateMem）按 `Question.as_of`（已 ingest 的消息条数，0..N）升序分组、边 ingest 边提问，中间**不 reset**，落盘仍按原始题序。`interleaved` + 题目级分片、`interleaved` + `--replay` 都在开跑前 `SystemExit`（前者各片会各自 reset，后者只有末尾快照）；`--snapshot` 允许，但只在全部 ingest 完成后导末尾态。`run --trace-state` 在每个有提问的 as_of ingest 完成后、query 之前导出 `<对话目录>/trace_state/t<as_of>/`（`layout.dialogue_trace_state_dir` / `dialogue_trace_state_path`；适配器不支持快照时警告一次并跳过），sequence 对话的提问时点就是末尾 `memory_snapshot/`。**`trace_state` 是三态、默认 `None`=自动**：本次要跑的对话里存在 interleaved 且含（未忽略）safety 题的对话时自动开启，并打印本次**确切片数**（未忽略题的 distinct `as_of` 数，不是题数）与单价告警阈值（safety 的「真遗忘」只能读提问时点的状态判定；不开时 `n_checked=0` 会被读成"safety 全过"），`True`（CLI `--trace-state`）强制、`False`（CLI `--no-trace-state`）关闭；`evaluate` / `evaluate-compare` / `run_task.py` 只透传。`results.jsonl` 每题记 `as_of` 与 `evidence_in_state`（扫描该时点状态目录里的 UTF-8 文本文件，跳过二进制与 `manifest.json`；needle 含证据本身、`gold_answer`、`judge_spec.include` 的正则与字面写法（见 `runner.include_literal_forms`）、`judge_spec.leak_targets`（≥8 字符、非哨兵；leak_targets 不依赖 evidence 选中的是哪一轮——safety 题的 evidence 常只剩删除指令那轮），长证据按句取至多 5 句；无快照 / 无 evidence / 状态不可读记 `null`，**不猜**）。`metrics.attribution` 与报告「失败归因」小节给四桶 `not_written / not_retrieved / wrong_answer / unknown`，只统计判错、带 evidence 且**非拒答**的题（拒答题由 `state_leak` 解释，另记 `n_wrong_abstention`；无 evidence 的判错题单独计数）。**`evidence_hit=true` 的判错题直接落 `wrong_answer`**——证据已进回答上下文，失败只可能在回答，不需要状态快照，因此没开 `--trace-state` 时该桶仍有值。**safety 题**在状态可读且 `evidence_in_state=True` 时直接判错（"删除后未遗忘"红线），并落 `metrics.state_leak {n_safety,n_checked,n_leak,rate}`；报告给覆盖率 `n_checked/n_safety`，`n_checked=0` 时按 `trace_state.role` 写明原因（`disabled` 显式关闭 / `unsupported` 适配器不支持 / 有字段但读不到 / 根本没有该字段=旧版本）；显式 `--no-trace-state` 且本次确有 safety 题时 run 打印醒目告警并落 `trace_state {enabled:false, role:"disabled"}`；状态不可读不改判、记 `unknown`。**拒答识别**：`contains_refusal` 词表覆盖常见英文拒答（`can't answer|provide|confirm|recover`、`no longer`、`not available`、`isn't recoverable`、`doesn't contain|hold`、`was deleted` 等）；`judge=llm` 且 `gold=<ABSTAIN>` 时词表判「未拒答」的题再发一次 YES/NO 复核（配了 LLM 端点才发，异常静默回退词表），reason 标 `拒答判定=llm|lexicon`；`yes/no/same/true/false` 这类答案形状的 `leak_targets` 在已拒答时不计泄露（`Yes.` 这种直接给答案仍判红线）。**近失复核**：`judge=llm`、非拒答、rubric 判错但 `include` 命中率 ≥50% 且无 `not_include`/`leak_targets` 硬违规时，再发一次 PASS/FAIL 复核（语义覆盖即可 PASS，翻判为对，reason 标 `近失复核=llm(...)`；异常回退 rubric）。`results.jsonl` 另落 `evidence`（每题最多 5 条 × 200 字符）与 `include_hit_rate`；`overall` 另给诊断项 `rubric_item_recall`、`n_empty_retrieved`、`retrieved_median`，均不参与评级、不进融合。`ingest_strategy` 是 `compare` 的**关键维度**（缺省归一 `sequence`，旧产物不误伤）。

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
