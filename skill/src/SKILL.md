---
name: agent-memory-eval
description: >
  对有源码的 Agent（尤其是通用办公助手类 Agent）进行记忆系统评测与优化迭代的工程化技能。
  内置 membench 评测工程（adapter 协议 + 六维指标 + 回归门禁 + 报告生成）与办公域黄金测试集。
  触发场景：评测/测试 Agent 的记忆系统、记忆能力打分、记忆系统优化前后对比（A/B）、
  两个 commit 的记忆回归对比、多平台（如企微/飞书/钉钉接入）记忆表现对比、平台×commit 矩阵对比、
  记忆系统优化闭环（诊断→建议→plan→执行→AB→报告）、
  基于用户真实对话日志构造/删除用户专属测试集。
  关键词：记忆评测、memory benchmark、记忆回归、A/B 对比、stale_rate、黄金集、优化闭环、可比性校验。
---

# agent-memory-eval：Agent 记忆系统评测技能

本技能封装 membench 评测工程（`membench/`，自带 README 与测试），覆盖**接入 → 评测 → 对比 → 优化闭环**。**所有评测报告必须输出到本 skill 的 `output/`（不传 `--report-out` 时 `output_root()` 同样是它）；一次实验一个自包含任务目录（单臂 `eval_*` / 双臂 A/B `compare_*` / `matrix_*`，目录名自带 `YYYYMMDD-HHMMSS`），报告开头注明所用测试集（membench 自动生成，勿删）。**

评测口径是**单用户个人 Agent**：GateMem 的 140 道 privacy-refuse 题已标 `ignore=true` 由引擎自动跳过（原始数据保留、口径可追溯）；222 道 safety 题全部参评；对外输出泄漏、提示注入等残余风险现有评测集未覆盖，可在用户测试集补题。执行位置：`scripts/*` 在 skill 根目录执行；`membench.cli` 在 `membench/` 目录内执行（下文命令省略 `cd membench`）。

## 我该走哪条路

| 用户想干什么 | 用哪条命令（默认路径） | 产物在哪 |
|---|---|---|
| 跑一次评测（单集 / 多集） | `scripts/run_task.py eval` | `output/eval_<name>_<platform>_<ts>/` |
| 全量评估（基准集加权宏平均 + 办公域单列） | 原生 `membench.cli evaluate` | 同上，任务级 `reports/evaluate_summary.json` |
| 公开基准常用集（无 office / lme_s） | 原生 `membench.cli evaluate --common`；可分片：`run_task.py eval --common` | 同 `eval_*` |
| 两个 commit 或两版系统 A/B | `scripts/run_task.py ab` | `output/compare_<name>_<platform>_<ts>/` |
| 跨数据集 A/B（融合 diff） | 原生 `membench.cli evaluate-compare` | 同 `compare_*`（含 `fig_fused_diff.png`） |
| 优化闭环（诊断→建议→plan→执行→AB→报告） | `scripts/run_task.py eval` / `ab` | 基线 `eval_*` + 对比 `compare_*` |
| 平台对比（同 commit 换平台） | 原生 `run` 两次 + `compare` | `compare_*` |
| 平台 × commit 矩阵 | 原生 `membench.cli matrix` | `output/matrix_<ts>/` |
| 回放验证（改读取侧免重跑 ingest） | 原生 `run --replay` | 回放任务目录（含 `snapshots/`） |
| 用户测试集管理 | 原生 `user-data`（install / status / delete） | `user_dataset/` |

两个脚本的完整参数以 `--help` 为准，本技能只给默认路径的最小调用；`scripts/eval_two_commits.sh` 是原生 CLI 的参考脚本（非默认路径）。

**产物定位**（完整布局与各文件含义见 `membench/README.md`「产物布局」）：

| 任务形态 | 目录 | 关键产物 |
|---|---|---|
| 单臂 | `eval_<name>_<platform>_<ts>/` | `runs/<dataset>/report/` + 任务级 `reports/` |
| 双臂 A/B | `compare_<name>_<platform>_<ts>/` | `A/`、`B/` + `reports/comparison.md` |
| 矩阵 | `matrix_<ts>/` | `reports/report.md` + `runs/<label>/` |
| 回放任务 | 上述目录 + `snapshots/` | 条目 `entries/` 是真源，`index.json` 是派生视图 |
| 默认根 | `output_root()` | `MEMBENCH_OUTPUT_DIR` > skill `output/` > 工程根 |

报告解读：单集报告给四级评价与综合评级；多数据集 `evaluate` 给**基准评级**（不含办公域槽位，默认按题数加权、封顶 500）+ **业务结论单列**（office 的准确率与 stale 红线，`stale_rate ≤0.03` 一票否决）；token_f1 / bleu1（字面重合，非语义正确性）/ recall_at_5 / over_refusal_rate 为诊断指标，仅展示不参与评级；细节见 `membench/README.md`「指标说明」。

## 标准流程

### 1. 评测（单集 / 多集）

```bash
python scripts/run_task.py eval --adapter <适配器名> --dataset <key|路径> --recall-mode llm --judge llm
```

- 多数据集重复 `--dataset`；`--dataset` 接受数据集 key（如 `--dataset office`）或 JSONL 路径，按需下载集先 `dataset fetch <key>`；全量宏平均走原生 `evaluate`。`--recall-mode` 与 `--judge` 必填，`--judge heuristic` 只能配 `--limit`。
- 脚本串起 `task init → run（含分片）→ fuse → verify_artifacts`；按需下载集先 `dataset fetch <key>`。
- 首次接入先读 `references/adapter_guide.md`（必需三挂点 + 可选快照/自描述 / HTTP 端点 / 持久化 / 验收）。

### 2. A/B（两个 commit、两版系统）

```bash
python scripts/run_task.py ab --baseline-adapter <基线> --candidate-adapter <候选> --dataset <数据集> --recall-mode llm --judge llm --name <标签> --baseline-version <版本A> --candidate-version <版本B> --driver-llm-a none --driver-llm-b none
```

- 两个 commit：先 `git worktree` 各检出并构建，再传两个 adapter（步骤见 `references/compare_workflows.md`）。
- 脚本把两臂写进同一棵 `compare_*` → 各自 `fuse` → `compare --task-dir` 出 `reports/comparison.md`。
- 门禁 accuracy 降 >0.02 或 stale_rate 升 >0.02 → FAIL（退出码 1）；两侧判分器、数据集、驱动模型、适配器版本必须一致。
- `--baseline-version` / `--candidate-version` / `--driver-llm-a` / `--driver-llm-b` 是**必填**（strict 可比性的关键维度）：同一系统只改读取侧时两侧版本填同一值；没有驱动 LLM 填 `none`。缺了会判"无法核验"→ FAIL，所以脚本直接不给默认。

### 3. 分片（大数据集）

```bash
python scripts/run_task.py eval --adapter <适配器名> --dataset ~/.membench/datasets/beam_1m.jsonl --shards 4 --recall-mode llm --judge llm
```

- 分片由脚本自动编排（共用一棵任务目录 → `fuse` 补齐），你只需给 `--shards`。
- 共享任务目录只允许对话级分片（脚本默认 `--shard-by dialogue`）；题目级分片属非默认路径，见 `membench/README.md`「大数据集：冒烟与分片」。
- 每片独占记忆实例（HTTP 桥 = 独立进程 + `BRIDGE_PORT` / `BRIDGE_DATA_DIR`）；分片产物只有对话目录 + `shards.<i>.json`，不是完整实验。

### 4. 回放（改了读取侧，免重跑 ingest）

```bash
python -m membench.cli run --adapter <适配器名> --dataset <同一数据集> --replay ../output/eval_<名>_<平台>_<ts> --replay-verify-source ../output/eval_<名>_<平台>_<ts>
```

- 快照默认导出（每对话 ingest 后一份），`--no-snapshot` 关闭；回放导出落任务级内容寻址池 `snapshots/`。
- `--replay-verify-source` 是保真地板：未改代码应逐题一致，不一致先修恢复再谈 A/B。
- A/B 回放同一源用 `task init --replay --arms 2`，状态按内容去重只存一份；`task status` 看两臂是否逐对话一致。
- 并行回放：同一臂按对话级分片多进程跑（各写自己的条目、无锁），收尾 `fuse --task-dir <臂>`；`run_mode` 是可比性维度，回放分与全量分不可比。

### 5. 平台 × commit 矩阵

```bash
python -m membench.cli matrix --runs "wx@<shaA>=<产物目录>" "fs@<shaA>=<产物目录>" "wx@<shaB>=<产物目录>" "fs@<shaB>=<产物目录>"
```

- 四个组合必须同测试集、同判分器、同驱动模型、同 adapter 版本，否则矩阵不可归因。
- 任一两两门禁 FAIL → 退出码 1；解读顺序：同平台跨 commit（回归）→ 同 commit 跨平台（平台差）→ 对角线（交互效应）。

## 三个检查点

### 检查点 1：改动做完，先问是否 A/B

任何优化/改动落地后，**先问用户是否现在跑 A/B、用哪个数据集**，得到明确答复再开跑。不做的后果：烧掉整晚评测成本，且"改完就自评"没有基线，结论不可归因。你是"提方案并按确认执行"的角色，不是"改完就自信跑分"的角色。

### 检查点 2：启动前确认范围与召回口径

启动任何评测/对比前，逐项与用户确认：**数据集范围**（单集 / 多集 / 全量）、**A/B 分组**（基线 vs 候选）、**平台/标签**、**判分器**，以及**召回口径必须显式**（`--recall-mode` 无默认；原生参数叫 `--evidence-match`，默认 `substring` 容易被静默带过）。不做的后果：口径或平台错配会让对比结论不可归因，召回口径选错只能整晚重跑：

| 口径 | 判定方式 | 适用与选错代价 |
|---|---|---|
| `substring` | 证据逐字出现在检索结果中 | 记忆存原文时用；摘要式存储会整晚系统性虚低 |
| `overlap` | 内容 token 包含度 ≥ 阈值 | 容错标点/包裹/轻微改写；仍偏字面 |
| `llm` | 逐题语义判定（每题一次 LLM 调用） | 摘要/改写式存储必选；慢，配 `--jobs 8~16` |

两侧必须同口径；换口径后与历史分数不可比；选错只能重跑。用户不确定时给判断依据（存原文→`substring`/`overlap`，存摘要/改写→`llm`），或先 `--limit 20` 冒烟看命中再定。不传 `--dataset` 时按**用户测试集 > 内置 office** 回退，要向用户说明实际用了哪个。

### 检查点 3：交付前必须自检

交付前必须运行 `python scripts/verify_artifacts.py <任务目录>`，**把输出贴给用户**；退出码非 0 时不得宣布完成，先按报告补齐或重跑。不做的后果：partial、死链、题数不一致的产物被当成交付，用户无法复核。自检覆盖：骨架与产物齐备、数据集级与逐对话题数一致、partial 不许冒充完整、报告内链无死链、compare 有无门禁结论。

## 长跑：后台执行 + 定时监控

```bash
python scripts/run_task.py eval ... --background      # stdout 只一行任务目录
python scripts/watch_task.py <任务目录> --until-done --interval 300
```

- 默认后台跑，**不要在同一轮里前台阻塞**等整场评测；每 5–10 分钟看一次进度。
- 三个卡死/失败信号：`orchestrator.json` 的 pid 已不在且无 `exit_code`（stalled）；分片账本长时间无新增；`runtime/logs/` 最近日志出现 traceback。
- `watch_task.py` 退出码：0 完成且成功、1 完成但失败、2 超时/无法判断；`--json` 出机器可读结构。

## 失败排查

| 看到什么 | 怎么办 |
|---|---|
| "旧版产物" / 缺 `report/` 层级 | 该实验是旧格式，重跑；不要手改目录冒充新格式 |
| "没有产物"（空目录 / 中途中断） | 确认路径没传错，重跑 |
| "未融合分片"（只有账本与对话目录） | `python -m membench.cli fuse --task-dir <任务目录>` 补齐 |
| "实验目录已存在且非空" / 同名实验已存在 | 换 `--name` 或换父目录；不要复用别人的分片 |
| 共享任务目录报"只允许 dialogue 分片" | 加 `--shard-by dialogue`，或各片各落实验目录再 `fuse --runs` |
| 回放报桶冲突 / 条目损坏 | 别手改 `snapshots/`；换新任务目录重跑回放 |
| `compare` / `matrix` 退出码 1 | 门禁 FAIL，不是崩溃：看 `comparison.md` / `matrix_result.json` 的结论与逐指标 diff |
| fuse / beam 报 stale 门禁 | 非办公域数据集超阈值只记 `INFO`（仅诊断、退出码 0），不是失败；真正否决看办公域槽位那条门禁（`FAIL`） |
| 检索指标全 0 | 先查口径（摘要式存储必须 `llm`）与 `retrieved` 是否回填了注入原文 |

`run_task.py` 的退出码：0 成功、1 A/B 门禁 FAIL、2 流程失败（看最后一条命令的报错）、3 交付自检不过。

## 目录与 references 索引

| 路径 | 用途 / 何时读 |
|---|---|
| `membench/` | 评测工程本体（CLI `python -m membench.cli`）；产物布局、指标说明、接入章节以其 README 为准 |
| `references/adapter_guide.md` | 接入被测系统时读：必需三挂点与可选接口、真实性红线、快照回放、验收与自查 |
| `references/compare_workflows.md` | 做 commit / 平台 / 矩阵对比与可比性排查时读 |
| `references/optimize_workflow.md` | 走优化闭环（诊断→建议→plan→执行→AB→报告）时读 |
| `references/http_bridge_guide.md` | 被测系统非 Python、要写 HTTP bridge 时读 |
| `references/user_dataset_guide.md` | 用户要求构造或删除专属测试集时读 |
| `scripts/run_task.py` | 标准流程编排器（eval / ab），退出码 0/1/2/3 见上；`--background` 派生到后台 |
| `scripts/watch_task.py` | 长跑进度轮询（后台任务配它），退出码 0/1/2；不要自己解析账本 |
| `scripts/verify_artifacts.py` | 交付前自检：骨架、产物、题数一致、partial、内链、门禁结论 |
| `scripts/eval_two_commits.sh` | 原生 CLI 的参考脚本（非默认路径），演示 worktree 双 commit |
| `membench/examples/quickstart.py` | 写真实 adapter 的参考实现 |
| `membench/examples/optimize_loop.md` | 优化闭环入门教程 |
| `membench/examples/http_bridge_node/` | Node/Electron HTTP bridge 可复制模板 |
| `user_dataset/` | 用户专属测试集（与内置集物理隔离，不打包） |
| `runtime/`（实验 / 臂目录下） | 每次命令的 `commands.jsonl`、`logs/`、`env.json`；排查问题先看它（与对话级 runtime 不同） |
| `output/` | 默认产物根（`MEMBENCH_OUTPUT_DIR` 可覆盖） |
| `docs/` | 《Agent 记忆系统深度研究报告》与配图 |

数据集口径：`office` / `sample` 内置；`locomo` / `longmemeval_*` / `gatemem_office` / `beam_*` 一律 `dataset fetch` 到 `~/.membench/datasets/`（全量清单与基准线用 `dataset info` 查），不内置、不进 skill 包。
