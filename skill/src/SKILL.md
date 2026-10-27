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

本技能封装 membench 评测工程（`membench/`，自带 README 与测试），覆盖**接入 → 评测 → 对比 → 优化闭环**的完整工作流。
**所有评测报告必须输出到本 skill 的 `output/` 目录（`--report-out <skill目录>/output`；不传时缺省 `output_root()` 同样是它）。产物一次实验一个自包含任务目录（单臂 `eval_*` / 双臂 A/B `compare_*` / `matrix_*`，目录名自带 `YYYYMMDD-HHMMSS` 时间戳；一次评测的多个分片、A/B 两臂都可以用 `--task-dir` 写进同一个任务目录），报告开头注明所用测试集（membench 自动生成，勿删）。**

## ⛔ 两个强制检查点（不得跳过）

1. **任何优化/改动完成后，先询问用户是否执行 A/B 对比，不得径自开跑。** 你是"提方案并按用户确认执行"的角色，不是"自动改完就自信跑分"的角色。
2. **启动 A/B 前，先与用户确认数据集范围（单集 / 多集 / 全量）、A/B 分组（基线 vs 候选）与平台/标签。** 数据集口径不一致、平台/标签错配，都会让对比结论不可归因。

> 纪律背景：`compare` 只**测量差异**，不**判定**两组是否只差一个变量。可比性必须由你（优化者）自己保证——固定模型与 harness、只切优化开关，并用 `--require-comparable` 做前置校验。

## 被测对象定位：单用户个人 Agent

**目标被测系统是单用户个人 Agent，记忆属于用户个人。** 该定位影响评测口径：

- **多用户权限题已屏蔽**：GateMem 内置集中 140 道 privacy-refuse 题（测"同一记忆对不同身份的选择性披露"）与单用户场景无关，已在数据集内标记 `ignore=true`，评测引擎自动跳过——不删除原始数据，口径可追溯，未来若产品演变为多用户形态可随时恢复。
- **safety（删除后遗忘）依然重要**：用户说"忘掉我上周的体检结果"，agent 之后不应再记起——这是数据主权问题，与单/多用户无关，GateMem 的 222 道 safety 题全部保留参评。
- **残余风险缺口**：单用户场景下的隐私风险以"对外输出泄漏"（agent 帮你写邮件时带出私密记忆）和"提示注入套取"形式存在，现有评测集未覆盖；如需要可在用户测试集中补充这类题。

## 目录约定

| 路径 | 用途 |
|---|---|
| `membench/` | 评测工程本体（CLI: `python -m membench.cli ...`） |
| `membench/membench/data/office_dataset.jsonl` | **内置**办公域黄金集（20 对话 / 86 题 / 7 类题型），业务验收门禁用 |
| `membench/membench/data/sample_dataset.jsonl` | **内置**冒烟迷你集（仅验证管线连通，不作评估结论） |
| `~/.membench/datasets/` | **按需下载**数据集的落盘目录（`locomo` / `longmemeval_oracle` / `longmemeval_s` / `gatemem_office` / `beam_1m` / `beam_10m`） |
| `user_dataset/` | 用户专属测试集存放目录（与内置集物理隔离） |
| `output/` | 所有评估/对比报告的输出目录（任务目录：单臂 `eval_*`、双臂 A/B `compare_*`、`matrix_*`；多分片/两臂可用 `--task-dir` 写进同一棵 `task init` 建的任务目录；可用 `MEMBENCH_OUTPUT_DIR` 覆盖） |
| `docs/` | 《Agent 记忆系统深度研究报告》及配图，供随时查阅背景知识 |
| `references/adapter_guide.md` | 接入指南详解（三挂点、真实性红线与降级口径流程、验收 checklist、判分器） |
| `references/compare_workflows.md` | commit / 平台 / 矩阵对比详解 + 可比性前置校验 |
| `references/optimize_workflow.md` | 记忆系统优化闭环操作手册（六阶段） |
| `references/http_bridge_guide.md` | 非 Python 系统 HTTP bridge 接入 checklist |
| `references/user_dataset_guide.md` | 用户测试集构造指南 |
| `scripts/eval_two_commits.sh` | 两个 commit 的 A/B 评测模板脚本 |
| `membench/examples/quickstart.py` | 真实 adapter 编写参考（把假想记忆系统包成 BaseMemoryAdapter） |
| `membench/examples/optimize_loop.md` | 优化闭环教程（基线 → 单变量改动 → compare 门禁 → 迭代） |
| `membench/examples/http_bridge_node/` | 非 Python（Node/Electron）HTTP bridge 可复制模板 |

## 工作流 0：接入被测 Agent（首次使用必做）

1. 阅读 `references/adapter_guide.md`（含第 3 节真实性红线：真实写入/读取路径是完整评测的唯一前提；旁路不得自行采用，走不通时停下、由用户决定降级口径），在被测 Agent 源码中定位三个挂点：记忆清空、记忆写入、记忆读取+回答。
2. 实现 `BaseMemoryAdapter` 三方法（`reset` / `ingest` / `query`），或在 Agent 侧暴露 `POST /reset|/ingest|/query` 三个端点后用内置 HttpAdapter（`run --adapter http --adapter-arg base_url=<服务地址>`，无需再包适配器）。
   - **`query` 必须回填 `retrieved`**：= 实际拼进回答 prompt 的**记忆条目原文**，按注入优先级排序，多轮检索**先合并去重**再返回，报**注入的那一份全量**（框架不合并、不去重、不截断；返回空列表 = 静默按 0 召回计）。它决定 `evidence_recall` / `recall@5` 是否成立，细则见 `references/adapter_guide.md` §2.3。
   - 需要「快照回放」（改读取侧免重跑 ingest）时，额外实现可选接口 `snapshot_capabilities()` / `snapshot()` / `restore()`（`rebuild()` 可选，覆盖跨版本/重建派生结构）——框架只传目录与 tag，导出什么由你决定；细则见 `references/adapter_guide.md` §2.7。**快照默认导出**（每对话 ingest 完成后、第一道 query 之前一份，落该对话目录的 `memory_snapshot/`），适配器没声明快照能力时会警告一次并跳过、评测照常完成；不需要快照加 `--no-snapshot` 关闭（`--snapshot` 已删除）。跑完 ingest 就能在对话级 `report.md` 里点开 `../memory_snapshot/` 翻记忆文件。回放时同一份 `snapshot()` 会被再次调用（`restore()`/`rebuild()` 之后），导出落任务级内容寻址池——适配器无需感知落点是逐对话目录还是池。
   - **若被测 Agent 的记忆能力本身就是"LLM + 工具调用"**（ingest 与 query 是同构的 tool-call 循环），不要各写一遍骨架——直接用 `membench.agent_loop.AgentLoopAdapter` 复用循环（见「本波新能力」）。
3. **生成的适配器必须显式持久化**（否则跨 session 丢失）：
   ```bash
   cd <skill目录>/membench
   python -m membench.cli adapter install /path/to/your_adapter.py --name <适配器名>
   # 落盘到 ~/.membench/adapters/，之后任何 session 直接 --adapter <适配器名> 可用
   # 查看：python -m membench.cli adapter list；删除：adapter remove <名>
   ```
   临时/CI 场景也可用环境变量 `MEMBENCH_ADAPTER_MODULES=module.path`（同名时环境变量优先于持久化）。
4. 冒烟验证（**membench 命令一律在 `<skill目录>/membench/` 项目根内执行**）：
   ```bash
   cd <skill目录>/membench
   python -m membench.cli run --adapter <你的adapter> --dataset membench/data/office_dataset.jsonl --judge heuristic
   ```
5. **验收标准**：pipeline 跑通且 `stale_rate` 与 `abstention_accuracy` 非零缺陷暴露合理；若全部满分，警惕 adapter 接错（如 query 没走记忆路径而是直接问底座模型）。

## 工作流 1：用户测试集管理（严格显式触发）

**铁律：构造和删除都必须由用户显式要求，绝不自动执行。评测时若用户测试集存在则默认使用，但不得主动建议之外的任何写操作。**

- **构造**（用户说"用我的真实数据构造测试集"等）：按 `references/user_dataset_guide.md` 将用户对话日志合成为 membench JSONL（schema 校验 + evidence 原文校验），然后：
  ```bash
  cd <skill目录>/membench
  MEMBENCH_USER_DATA_DIR=<skill目录>/user_dataset python -m membench.cli user-data install <合成文件>
  ```
- **删除**（用户说"删除我的测试集"等）：先向用户确认，再执行 `user-data delete`。
- **查看**：`user-data status`。

## 工作流 1.5：数据集介绍与按需下载

用户问"支持哪些测试集 / 这个分数算好还是差"时，用内置介绍命令（不要凭记忆回答，以命令输出为准）：

```bash
cd <skill目录>/membench
python -m membench.cli dataset info                 # 列举全部数据集：来源、特点、基准线要点
python -m membench.cli dataset info locomo          # 单个数据集详细介绍（含实际统计与用法示例）
python -m membench.cli dataset fetch locomo         # 下载并转换落盘到 ~/.membench/datasets/
python -m membench.cli dataset fetch longmemeval_oracle
python -m membench.cli dataset fetch beam_1m
```

**数据集口径（务必记牢，别再按记忆误答）：**

| 数据集 | 状态 | 说明 |
|---|---|---|
| `office` | **内置** | 办公域黄金集（`membench/data/office_dataset.jsonl`），业务验收门禁 |
| `sample` | **内置** | 冒烟迷你集（`membench/data/sample_dataset.jsonl`） |
| `locomo` | **按需下载** | `dataset fetch locomo`（10 对话 / 1986 题，英文日常对话，横向对标） |
| `longmemeval_oracle` | **按需下载** | `dataset fetch longmemeval_oracle`（500 题，只含证据会话） |
| `longmemeval_s` | **按需下载** | `dataset fetch longmemeval_s`（500 题，~50 会话/题，~268MB，检索压力测试） |
| `gatemem_office` | **按需下载** | `dataset fetch gatemem_office`（17 episodes / 547 检查点，utility/safety 两维有效） |
| `beam_1m` / `beam_10m` | **按需下载** | 百万/千万 token 级压力测试（体积大，不内置） |
| `statemembench` | 暂不可用 | 数据未公开，需向作者索取 |

- **只有 `office` / `sample` 是内置的**，其余全在 `~/.membench/datasets/`（可用 `MEMBENCH_DATASETS_DIR` 覆盖）。
- 每个条目的「基准线」小节说明看到评分如何判断好坏（如 LoCoMo 人类上限 F1≈87.9%、近年记忆系统 ACC 57%~77% 参考区间）。

## 工作流 2：运行评测，输出评估报告

```bash
cd <本skill目录>/membench
export MEMBENCH_USER_DATA_DIR=<skill目录>/user_dataset
python -m membench.cli run --adapter <adapter> --judge heuristic --report-out ../output/
# 或指定测试集 / 平台 / commit / 驱动 LLM / 适配器版本：
python -m membench.cli run --adapter <adapter> --dataset <路径> --platform <平台名> --commit <sha> \
  --driver-llm <驱动模型标识> --adapter-version <适配器版本> --report-out ../output/
# 用按需下载的基准（先 fetch；文件在 ~/.membench/datasets/）：
python -m membench.cli run --adapter <adapter> --dataset ~/.membench/datasets/locomo.jsonl --report-out ../output/
# 完整清单与介绍：python -m membench.cli dataset info
```

> **评测前置确认（强制）**：启动任何评测（`run` / `evaluate` / `evaluate-compare`，含单纯评测与对比）之前，必须先向用户确认**召回率口径**（`--evidence-match`），并说明选项与适用性：
>
> | 口径 | 判定方式 | 适用与代价 |
> |---|---|---|
> | `substring`（默认） | 证据逐字出现在检索结果中 | 记忆存原文/近原文时用；最快、与历史跑分可比 |
> | `overlap` | 内容 token 包含度 ≥ 阈值（容错标点/包裹/轻微改写） | 仍偏字面；免费 |
> | `llm` | 逐题语义判定（每题一次 LLM 调用） | **摘要/改写式存储必选**——选错时整晚评测的召回率会系统性偏低；慢，配 `--jobs 8~16` |
>
> - 对比类（`evaluate-compare`、A/B 两次 run）**两侧必须同口径**；
> - 换口径后与历史 substring 跑分不可直接对比（可比性校验会给 WARN）；
> - 用户不确定时不要干问：给判断依据（记忆系统存原文→`substring`/`overlap`；存摘要/改写→`llm`），仍不确定就先 `--limit 20` 小样本冒烟看命中情况，再定完整评测口径；
> - 口径选错只能重跑（无自动补救）——这一步是整晚评测成本的关键闸门，必须落实。

**产物目录约定（一次实验一个自包含任务目录，报告与数据就近归属）：**

| 命令 | 实验目录（`--report-out DIR` 的产出） | 内含 |
|---|---|---|
| `task init --name NAME --arms 1 [--replay]` | `DIR/eval_<name>_<platform>_<ts>/`（stdout 只打印绝对路径，故 `TASK=$(...)` 可接走） | 骨架：`runs/` + 任务级 `reports/`（`--arms 2` 建 `A/` `B/` 两臂 + `reports/`）；`--replay` 额外建任务级快照池 `snapshots/`（见工作流 2.7） |
| `run --report-out DIR` | `DIR/eval_<adapter>_<platform>_<ts>[_s<i>][_l<limit>]/` | `runs/<dataset>/{report/{report.md,metrics.json,results.jsonl,fig_*.png},shards.<i>.json + shards.json(分片账本),<对话目录>/{runtime/,memory_snapshot/,report/}}` + 任务级 `reports/` |
| `run --task-dir DIR` | 不新建目录，直接写 `DIR`（跳过 `--report-out`） | 同上；多分片共享 `DIR` 时**只允许 `--shard-by dialogue`**，同片重复跑被拒 |
| `evaluate --report-out DIR` | `DIR/eval_<adapter>_<platform>_<ts>/` | 各数据集落 `runs/<dataset>/…`；任务级 `reports/{report.md,metrics.json,evaluate_summary.json,fig_dataset_overview.png}` |
| `compare` / `evaluate-compare` `--report-out DIR` | `DIR/compare_<name>_<platform>_<ts>/` | `A/`、`B/`（各一个完整 eval 实验形态，自带 `runs/` 与 `reports/`）+ `reports/{comparison.md,fig_metric_diff.png,fig_category_diff.png}`（evaluate-compare 为 `fig_per_dataset_diff.png`、`fig_fused_diff.png`） |
| `compare --task-dir DIR` | 不新建目录：就地读 `DIR/A`（基线）与 `DIR/B`（候选） | `reports/comparison.md` + 两张对比图（不复制两臂） |
| `matrix --report-out DIR` | `DIR/matrix_<ts>/` | `reports/{report.md,matrix_result.json,fig_matrix_accuracy.png}` + `runs/<label>/`（各 label 的数据集树副本） |
| `fuse --out DIR` / `fuse --task-dir DIR` | 前者：`DIR` 即完整实验目录；后者：就地补齐 `DIR` | `runs/<dataset>/…` + `reports/`（分片产物只有对话目录与账本，数据集级/任务级靠这一步补齐） |

- `<ts>` = `YYYYMMDD-HHMMSS`；`<platform>` 缺省 `default`；`--shards` 分片时目录名末尾加 `_s<分片序号>`、`--limit` 追加 `_l<limit>`（不分片无 limit 时不带后缀）——并行分片因此**可以共用同一个 `--report-out`**；同名且非空会被直接拒绝（不会静默覆盖）。
- **`--task-dir DIR`（与 `--report-out` 互斥）**：写进/读**已存在**的任务目录（`task init` 建的目录，或它的臂目录 `A/` / `B/`），不新建目录——同一次评测的多个分片、A/B 两臂因此落在同一棵树里。`task status DIR` 是只读进度视图（骨架 / 分片账本进度 / 快照数与体积 / 下一步该跑什么）；回放任务显示任务级池（桶数、体积、各臂复用/新写、两臂是否逐对话一致）。
- **记忆快照默认启用**：每个对话在 ingest 完成后、第一道 query 之前导出唯一一份到 `<对话目录>/memory_snapshot/`（框架另写 `manifest.json`）；跑完 ingest 就能在对话级 `report.md` 里点开 `../memory_snapshot/` 看记忆文件（报告给链接 + 清单条目数与体积）。适配器未声明快照能力时警告一次并跳过（评测照常完成，`state_snapshot.role="unsupported"`），不需要快照用 `--no-snapshot` 关闭（`--snapshot` 已删除）。**回放也导出**，但落任务级内容寻址池 `snapshots/`（同一状态只存一份；`--no-snapshot` 连池也不写）——见工作流 2.7。
- **读产物的入参**统一接受**实验目录**或**数据集目录**（`<实验目录>/runs/<dataset>`）：`compare --baseline/--candidate`、`evaluate --from-runs KEY=DIR`、`evaluate-compare --baseline-run/--candidate-run`、`matrix --runs`、`fuse --runs`、`run --replay`、`run --replay-verify-source`，都经 `membench.layout.resolve_dataset_dir` 解析（`run --replay` 另接受回放任务的任务根 / 臂目录）；旧版平铺 run 产物（缺 `report/` 层级）一律报错，要求重跑；目录里根本没有数据集产物（空目录、跑到一半中断）另有一条"没有产物"的报错，不要说成版本问题；**未融合的分片产物**（数据集目录下只有账本与对话目录）又会要求先 `fuse`——"没融合"不是"没产物"。
- 不传 `--report-out` / `--task-dir` 时用 `output_root()`：`MEMBENCH_OUTPUT_DIR` > skill 包 `output/` > 工程根 `membench/output/`。目录命名只走 `membench/layout.py`（`task_subdir` / `init_task_dir` / `arm_dir` / `eval_subdir` / `compare_subdir` / `matrix_subdir` / `experiment_tags` / `run_id` / `run_id_tags` / `shard_ledger_path`），不要自己拼目录名。

### 工作流 2.5：全量评估（多数据集融合总分）

用户问"整体打个分 / 全部数据集跑一遍"时用 `evaluate`：对办公域槽位（用户集>内置office）+ locomo + longmemeval_oracle + gatemem_office + LongMemEval-S + BEAM-1M 依次评测，跨数据集**宏平均**融合出综合评级，输出一份全量报告（**启动前先做「评测前置确认」选定召回口径**，见工作流 2）：

```bash
python -m membench.cli evaluate --adapter <adapter> --judge llm --report-out ../output/
python -m membench.cli evaluate --adapter <adapter> --no-lme-s   # 剔除 LongMemEval-S（~268MB、ingest 成本高）
python -m membench.cli evaluate --adapter <adapter> --no-beam    # 不跑 BEAM
python -m membench.cli evaluate --adapter <adapter> --beam-10m   # 1M 换成 10M 全集终验（与 --no-beam 互斥）
python -m membench.cli evaluate --adapter <adapter> --quick      # 跳过 BEAM 与 LongMemEval-S，分钟级日常回归
python -m membench.cli evaluate --adapter <adapter> \
    --dataset office --dataset locomo --dataset ~/data/my_set.jsonl  # 显式清单（与预设开关互斥）
python -m membench.cli evaluate --from-runs office=<run目录> --from-runs locomo=<run目录>
# ↑ 汇总模式：已有 run 复用、缺口自动补跑；全部已跑完时 --adapter/--judge 可省略
#   （同一侧 adapter / adapter_version / driver_llm / judge / judge_model / 检索口径 必须一致，
#    混入多个被测对象或补跑参数与已有 run 不一致 → 开跑之前直接拒绝；
#    仅 platform 差异会在控制台与报告顶部标注；与预设开关互斥）
# 默认清单含 LongMemEval-S 与 BEAM-1M（用 --no-lme-s / --no-beam 剔除）
```

用户问"两版系统 A/B 对比一下、跨数据集一次出结论"时用 `evaluate-compare`：两套 adapter × 同一组数据集，分别跑完后产出**逐数据集 diff + 逐数据集题型 diff + 跨数据集融合 diff + 融合/per-dataset 两层门禁 + 对比图**，落 `<报告父目录>/compare_<name>_<platform>_<ts>/` 自包含结构——`A/`、`B/` 各为完整 eval 实验形态，结论在 `reports/comparison.md`、两图（`fig_per_dataset_diff.png` / `fig_fused_diff.png`）同目录（两侧 capabilities 未落账时自动推断并在报告头标注来源；**启动前先做「评测前置确认」，两侧同口径**，见工作流 2）：

```bash
python -m membench.cli evaluate-compare \
    --baseline-adapter <adapter_A> --candidate-adapter <adapter_B> \
    --judge llm --report-out ../output/ --tag "优化前vs优化后"
# 两侧可分别指定 driver_llm / adapter-version 供 compare 可比性前置校验
# 检索口径默认 substring（逐字子串）；摘要/改写式记忆可换 --evidence-match overlap|llm
# （口径不同不可与历史 substring 跑分直接比较，两侧需同口径重跑）
# llm 口径为逐题 chat 调用（单题可达 10s+），大规模评测配 --jobs 8~16 并发跑

# 汇总模式：两侧各自复用+补跑（KEY=DIR 可重复，DIR = 实验目录或 runs/<dataset> 数据集目录；
# 某侧全部已跑完时可省略该侧 adapter；
# 每侧 adapter / adapter_version / driver_llm / judge / judge_model / 检索口径 必须一致，
# 混用或补跑参数与已有产物不一致 → 开跑之前直接拒绝）
python -m membench.cli evaluate-compare \
    --baseline-run office=<A 侧产物> --candidate-run office=<B 侧产物> \
    --baseline-run locomo=<A 侧产物> --candidate-run locomo=<B 侧产物> \
    --report-out ../output/ --tag "优化前vs优化后"
# 报告附「两侧可比性（逐数据集）」：dataset_sha256 / judge / judge_model / driver_llm / platform
# 不一致或两侧都未标注逐项列出（两侧都复用的数据集对在跑前先打印一份——别等跑完才发现）
```

> **全量评估前必须先 fetch 用到的按需下载数据集**（locomo / longmemeval_oracle / gatemem_office / longmemeval_s / beam_1m 等），否则会报错提示先 `dataset fetch <key>`。命令：`python -m membench.cli dataset fetch <key>`（落 `~/.membench/datasets/`）。

解读要点（向用户说明）：
- **宏平均**：每个指标先在各数据集内算分再跨集等权平均；指标只在具备对应标注能力的集上参与（无 gold_stale 的集不参与 stale_rate，无 evidence 标注的集不参与 evidence_recall），不会用"记 0"稀释总分。
- **检索指标全 0 时先查口径**：若报告提示"两项检索指标全为 0"，多半是 evidence 原文不在检索结果里（摘要式存储）；用 `--evidence-match overlap|llm` 两侧同口径重跑，或让适配器回填检索原始片段。
- **门禁独立于总分**：办公域槽位 stale_rate>0.03 一票否决，总评"不达标"、退出码 1，无论综合评级多高。
- BEAM 未 fetch 时会报错提示先 `dataset fetch beam_1m`；heuristic 判分跑 BEAM 会警告失真，正式全量评估用 `--judge llm`。

**判分器三选一**（按可信度和可用性排序）：
- `--judge llm`：第三方模型（OpenAI 兼容协议），**正式评估首选**，与公共基准口径一致。配 `--llm-base-url/--llm-api-key/--llm-model`。
- `--judge client`：复用 **agent 自己的 LLM** 做判分（自评，可能系统性偏高，但可用于快速验证）。配 `--client-llm "my_agent.llm:LLMClient"`（类路径，会被无参实例化）。适合记忆系统针对自家 LLM 做过优化、想先看自评效果的场景。
- `--judge heuristic`：零成本默认，规则判定。stale/abstention 红线指标始终由规则判定，不受 LLM 影响。

测试集选择优先级（membench 自动执行，向用户说明实际使用了哪个）：
**用户显式指定 > 用户测试集（如已构造）> 内置办公黄金集**。

### 工作流 2.6：大数据集分片与任务目录

大数据集（BEAM-1M / LongMemEval-S 这类）按分片并行跑时，让**同一次评测的所有分片写进同一个任务目录**，最后一步就地补齐产物：

```bash
cd <本skill目录>/membench
export MEMBENCH_USER_DATA_DIR=<skill目录>/user_dataset
# 1) 建任务目录：--arms 1 单臂（runs/ + reports/），--arms 2 A/B 双臂（A/ B/ + reports/）
#    stdout 只有任务目录的绝对路径，可直接接走；报告仍落在 <skill目录>/output/ 下
TASK=$(python -m membench.cli task init --name beam --arms 1 --report-out ../output/)

# 2) 各分片写进同一棵树（对话级分片：整段对话归片、分片间不重复 ingest；可并行/多机）
python -m membench.cli run --adapter <adapter> --dataset ~/.membench/datasets/beam_1m.jsonl \
    --task-dir "$TASK" --shards 4 --shard-index 0 --shard-by dialogue
# ... --shard-index 1..3 同样写进 "$TASK"（每片独占自己的记忆实例，见下）
python -m membench.cli task status "$TASK"   # 只读进度：哪片跑完、还缺哪片、下一步该跑什么

# 3) 就地融合：补齐数据集级与任务级产物（校验分片齐全/口径一致/题目无重复；指标重算 + 门禁）
python -m membench.cli fuse --task-dir "$TASK"
```

A/B 两侧共用一棵任务目录（多分片对比的官方通路）：

```bash
TASK=$(python -m membench.cli task init --name opt --arms 2 --report-out ../output/)   # 建 A/ B/
python -m membench.cli run --adapter <基线 adapter> --dataset <数据集> \
    --task-dir "$TASK/A" --shards 4 --shard-index 0 --shard-by dialogue
python -m membench.cli run --adapter <候选 adapter> --dataset <数据集> \
    --task-dir "$TASK/B" --shards 4 --shard-index 0 --shard-by dialogue
python -m membench.cli fuse --task-dir "$TASK/A" && python -m membench.cli fuse --task-dir "$TASK/B"
python -m membench.cli compare --task-dir "$TASK" --require-comparable   # 就地出 comparison.md
```

- **共享任务目录只允许 `--shard-by dialogue`**：题目级分片下同一对话会被多片写进同一个对话目录、互相覆盖，开跑前就被拒绝；题目级分片请各片各落一个实验目录（`--report-out`，目录名带 `_s<i>`），再用 `fuse --runs <各片目录>,... --out <完整实验目录>` 跨目录融合。
- 分片产物**只写本片负责的对话目录 + 账本**（逐片条目 `runs/<dataset>/shards.<i>.json`，`shards.json` 是物化视图），数据集级 `report/` 与任务级 `reports/` 一律由 `fuse` 补齐（所以分片产物不会被误读成一份完整实验）；账本记各片的 `status` / 负责的对话 / 题数 / `run_id`，口径（`shards` / `shard_by` / `limit` / 数据集指纹）不一致或同片重复跑都会被拒绝；逐片条目让多个分片进程可以并行写同一棵树而不会丢条目。
- 融合后的产物形态与不分片直跑一致，可直接进 `compare` / `evaluate` / `evaluate-compare` 或 `matrix`。
- 两侧/各片必须各自独占一个记忆实例（HTTP 桥 → 每片独立桥进程与 `BRIDGE_PORT` / `BRIDGE_DATA_DIR`；Python 适配器 → 每片独立进程），否则跨片串扰。

### 工作流 2.7：快照回放与回放任务的快照池

改了**读取侧**（检索/排序/回答）代码后，用上一次评测导出的状态快照回放，跳过 `reset` + `ingest`，只跑 query。快照默认导出，无需额外开关。

```bash
# 1) 全量跑一次（快照默认导出到各对话目录的 memory_snapshot/）
python -m membench.cli run --adapter <adapter> --dataset <数据集> --report-out ../output/

# 2) 回放 + 保真校验地板（未改代码时答案应逐题一致）
python -m membench.cli run --adapter <adapter> --dataset <同一数据集> \
    --replay ../output/eval_<名>_<平台>_<ts> \
    --replay-verify-source ../output/eval_<名>_<平台>_<ts>/runs/<dataset>

# 3) A/B 两臂回放同一源：用回放任务，状态按内容去重只存一份
TASK=$(python -m membench.cli task init --name replay --arms 2 --replay --report-out ../output/)
python -m membench.cli run --adapter <基线 adapter> --dataset <数据集> \
    --task-dir "$TASK/A" --replay ../output/eval_<名>_<平台>_<ts>
python -m membench.cli run --adapter <候选 adapter> --dataset <数据集> \
    --task-dir "$TASK/B" --replay ../output/eval_<名>_<平台>_<ts>
python -m membench.cli task status "$TASK"   # 桶数 / 各臂复用新写 / 两臂是否逐对话一致

# 3b) 同一臂多进程并行回放（大集）：对话级分片，各进程只写自己对话的条目
python -m membench.cli run --adapter <adapter> --dataset <数据集> \
    --task-dir "$TASK/A" --replay ../output/eval_<名>_<平台>_<ts> \
    --shards 2 --shard-index 0 --shard-by dialogue &
python -m membench.cli run --adapter <adapter> --dataset <数据集> \
    --task-dir "$TASK/A" --replay ../output/eval_<名>_<平台>_<ts> \
    --shards 2 --shard-index 1 --shard-by dialogue &
wait
python -m membench.cli fuse --task-dir "$TASK/A"   # 收尾：物化快照池派生索引
python -m membench.cli task status "$TASK"

# 4) 链式回放：上一轮回放任务本身可作源（传臂目录；任务根仅在池里只有一个臂时可用）
TASK2=$(python -m membench.cli task init --name replay2 --arms 2 --replay --report-out ../output/)
python -m membench.cli run --adapter <adapter> --dataset <数据集> \
    --task-dir "$TASK2/A" --replay "$TASK/A"
```

- **池的位置与结构**：`task init --replay` 在任务根建 `snapshots/`；回放导出（`restore()`/`rebuild()` 之后、第一道 query 之前）落 `snapshots/<状态内容 sha256 前 12 位>/`；**每个对话一个条目** `snapshots/entries/<臂>/<数据集>/<对话>.json`（真源，自足：digest / 状态哈希 / 体积 / 文件数 / 身份 / 来源 / reused / run_id / 写入时间）；`snapshots/index.json` 是派生视图（`buckets` / `entries[臂|数据集|对话 -> 桶]` / `arms[来源与计数]`），由条目 + 桶清单重算，`fuse --task-dir` 收尾与 `task status` 刷新。池属于任务：`--task-dir <任务>/A|B` 时就是父任务的池。
- **并行回放（无锁）**：条目一对话一文件、对话级分片下各进程只写自己的对话；桶落位只靠一次原子 `os.replace` 判"新建 / 复用"（失败方校验桶内哈希一致后记 `reused`）。共享任务目录**只允许 `--shard-by dialogue`**，题目级分片会被拒绝；跑完用 `fuse --task-dir <臂>` 收尾。
- **内容去重（核心价值）**：A/B 回放同一源、恢复出的状态相同 → 只产生一份桶，条目里两臂指向同一 digest；不同则各存一份——**差异本身就是"回放不保真"的证据**。`task status` 对"两臂共用同一批桶"给出人话（如 `A 臂与 B 臂快照 7/7 逐对话一致（共用 7 个桶）`），有差异则列出 `内容不同 / 仅 A / 仅 B` 的对话数。
- **冲突即报错**：桶已存在但 `entry.sha256` 与本次不一致（哈希前缀撞车 / 桶被改动 / 坏 manifest）直接失败，不静默复用；条目损坏 / 版本不认识同样拒绝（别手改 `entries/` 或 `index.json`）。
- **`--no-snapshot`**：回放也关掉池导出——不写桶、不写条目 / 索引；`metrics.state_snapshot` 里没有 `snapshots_dir` / `n_buckets`。
- **源解析**：`--replay` 接受实验目录 / 数据集目录 / 回放任务（带 `snapshots/` 的任务根或臂目录）。读源优先读条目树，只有旧产物（没有 `entries/`）才回退读 `index.json`。传任务根且池里有多个臂时会报错并要求改传具体臂目录（两臂状态可能不同，不能替用户选）。跑前校验强度不变（覆盖齐备、`complete=true`、数据集指纹、格式版本都按桶内 `manifest.json` 核）。
- **报告与口径**：回放任务的对话级报告把「本对话快照桶」链到 `snapshots/<digest>/` 并标「复用 / 新写」，数据集级报告给池目录与本 run 桶数。回放仍**固定写入侧**：结论只覆盖读取/检索/回答层，`run_mode` 是 compare 的关键维度（回放分与全量分不可比），token 账 ingest 记 0 并标 `ingest_skipped`。

## 工作流 3：两个 commit 的 A/B 对比

详见 `references/compare_workflows.md`。要点：用 `git worktree add` 分别检出两个 commit，各自构建并以 `--commit <sha>` 标签跑同一测试集，然后：

```bash
python -m membench.cli compare \
  --baseline <旧 commit 的实验目录或 runs/<dataset> 数据集目录> \
  --candidate <新 commit 的实验目录或 runs/<dataset> 数据集目录> \
  --require-comparable --tag <标签> --report-out ../output/
```

- **`--require-comparable`**：对比前做**前置校验**——驱动/判分模型、适配器实现版本、数据集指纹等关键维度不一致即判"不可比"，禁止当 A/B 用（避免"变量变了还照给 PASS"）。
- **召回口径一致性**：两次 run 必须同一 `--evidence-match` 口径（默认 substring；口径不同对比无意义，可比性校验会给 WARN）——启动评测前按「评测前置确认」（见工作流 2）先与用户敲定。
- **`--tag`**：给本次对比命名，决定实验目录 `compare_<name>_<platform>_<ts>/` 的 `<name>` 段（缺省 `<基线 adapter>_vs_<候选 adapter>`）。
- **`--task-dir <任务目录>`**：A/B 两臂本来就写在同一个任务目录里时（多分片 A/B 通路的常态，见工作流 2.6），用 `compare --task-dir <任务目录> --require-comparable` **就地**读 `A/` 与 `B/` 出报告，`comparison.md` 落该任务目录的 `reports/`——不新建 `compare_*`、也不复制两臂。
- 其余 commit 对比模板见 `scripts/eval_two_commits.sh`。

门禁：accuracy 降 >0.02 或 stale_rate 升 >0.02 → FAIL（退出码 1，可接 CI）。

## 工作流 4：记忆系统优化闭环

完整操作手册见 `references/optimize_workflow.md`，入门教程见 `membench/examples/optimize_loop.md`。六阶段：

```
诊断（读 report.md：六维指标 / 分题型 / 失败样例 Top10 / 门禁 → 结论"问题出在哪"）
  → 优化建议（可执行方向清单，按收益排序）
  → 优化 plan（拆成单变量改动、排优先级、界定验收指标）
  → 执行（落地单个改动；范围明确且独立时才交 subagent，默认当前会话执行）
  → AB 对比（消融 + 可比性前置校验）
  → 优化报告（方案 + 可归因效果，落对比实验目录的 reports/optimization_report.md）
```

| 阶段 | 动作与命令 | 产物 | 检查点 |
|---|---|---|---|
| 1 诊断 | 读基线 `report.md`（六维指标、by_category、失败样例 Top10、门禁），对照 `optimize_loop.md` 的"现象→短板维度"表定位 | 诊断结论（问题维度 + 证据） | — |
| 2 优化建议 | 给出可执行方向清单，按预期收益排序 | 建议清单 | — |
| 3 优化 plan | 每条拆成**单变量**改动、排优先级、界定验收指标（含门禁阈值） | plan（一人一步） | — |
| 4 执行 | 落地单个改动（只改一处；默认当前会话执行，范围明确且独立时才用 subagent） | 代码改动 | **⛔ 检查点①：改完后先问用户是否跑 A/B** |
| 5 AB 对比 | 固定模型/harness、只切优化开关；先 `--require-comparable` 校验再 `compare` | `compare_<name>_<platform>_<ts>/` | **⛔ 检查点②：跑前先确认数据集范围、A/B 分组与平台/标签** |
| 6 优化报告 | 记录方案 + 可归因效果（含门禁结论） | `compare_<name>_<platform>_<ts>/reports/optimization_report.md` | — |

- 每次只改一个变量；数据集与判分器在两次 run 之间保持不变（见工作流 2 / 3）。
- 失败/回退样例里若出现大量"答案其实对但判 False"，先切 `--judge llm` 再定胜负（见 `references/optimize_workflow.md`）。
- **`compare` 只测差异、不判定消融**；可比性由你自己保证（如上）。

## 工作流 5：两个平台对比

同一 commit、同一测试集，分别以 `--platform <平台A>` / `--platform <平台B>` 跑两次（adapter 内按平台切换接入配置），然后 `compare`。报告中明确指出平台差异最大的题型（by_category diff）。

## 工作流 6：平台 × commit 矩阵对比

四个组合各跑一次（标签 `平台@commit`），然后：

```bash
python -m membench.cli matrix \
  --runs "wx@<shaA>=<实验目录1 或 runs/<dataset>>" "fs@<shaA>=<实验目录2 或 runs/<dataset>>" \
         "wx@<shaB>=<实验目录3 或 runs/<dataset>>" "fs@<shaB>=<实验目录4 或 runs/<dataset>>" \
  --report-out ../output/
```

产出矩阵透视表（每指标一张 平台×commit 表）+ 全部两两门禁 + 分组条形图。
任一两两组合门禁 FAIL → 退出码 1（与 `compare` 同口径，可接 CI）。

## 本波新能力速查

### agent-loop 驱动复用（ingest 与 query 同一套循环）

被测 Agent 的 ingest（喂会话让 LLM 调写入工具落库）与 query（让 LLM 调检索工具作答）往往是**同一套 tool-call 骨架**。用 `AgentLoopAdapter` 把循环、统计与证据回填交给 membench，接入方只给 `{llm_client, tools, system_prompt}`：

```python
from membench.agent_loop import AgentLoopAdapter

adapter = AgentLoopAdapter(
    llm_client="my_agent.llm:LLMClient",   # module:ClassName 类路径（无参实例化），与 --judge client 同约定
    ingest_tools=[{"name": "write_memory", "schema": {...}, "handler": write_memory}],
    query_tools=[{"name": "retrieve", "schema": {...}, "handler": retrieve}],  # 缺省复用 ingest_tools
    system_prompt="你是一个带长期记忆的办公助手。",
)
```

- 确切签名：`AgentLoopAdapter(llm_client, ingest_tools, query_tools=None, system_prompt=None, name="agent-loop", max_rounds=8, retrieved_from=None, ingest_prompt=None, query_prompt=None)`。
- `llm_client` 可为对象 / 可调用 / `"module:ClassName"` 类路径；`tools` 元素为 `{"name","schema","handler"}`。
- membench 负责循环、`IngestStats`/`QueryResult` 统计与 `retrieved` 回填（缺省把检索类工具名含 `retrieve`/`search`/`recall` 的**多轮结果按调用顺序合并去重**；不满足时传自定义 `retrieved_from`）。
- 把实例交给 Runner API 即可评测（见 `membench/examples/quickstart.py` 的用法）。
- **适用边界**：仅当被测系统的写入/读取函数与 LLM 客户端能在 **Python 侧**调用时适用；非 Python 系统（Node/Electron）的 tool-call 循环必须留在被测进程内，membench 无法复用——走 HttpAdapter 三端点（见下一节）。

### HTTP bridge 与 doctor（非 Python 系统）

被测 Agent 不是 Python（Node/Electron）且记忆是库内能力时，写一个极薄 HTTP bridge 暴露三端点，再用内置 HttpAdapter。细节见 `references/http_bridge_guide.md`，可复制模板见 `membench/examples/http_bridge_node/`。

跑评测前先做连通性 + 契约校验：

```bash
python -m membench.cli doctor --target <url>     # 连通性 + 三端点契约字段校验（reset/ingest/query）

# 直连跑评测：桥地址用 --adapter-arg 传入（无需再包一层适配套件或 adapter install）
python -m membench.cli run --adapter http --adapter-arg base_url=<url> \
  --adapter-arg timeout=10 --dataset membench/data/office_dataset.jsonl \
  --judge heuristic --report-out ../output/
# HttpAdapter 接受的构造参数：base_url（必填）、api_key、timeout
```

- 校验 `/reset` 2xx、`/ingest` 2xx（含 `input_tokens`/`llm_calls` 数值校验）、`/query` 2xx + JSON + 字符串 `answer`（`retrieved` 若出现须为字符串列表）。
- 底层实现为 `membench.doctor.check_target(base_url, timeout=10.0, api_key=None)`，只依赖标准库。
- **清空前快照（可选，回溯用）**：框架在每次 `reset()` 前给出上下文（HTTP：`POST /reset` 请求体的 `run_dir` / `dialogue_id` / `dialogue_index`；Python 适配器：同名环境变量 `MEMBENCH_*`），框架**不读不写**记忆存储——要不要快照由接入侧决定。`MEMBENCH_RUN_DIR` / `run_dir` 现在指**该对话的目录**（`<实验目录>/runs/<dataset>/<对话目录>`，其 `runtime/` 框架已建好），约定把清空前的备份落 `<对话目录>/runtime/reset_backup/`；默认关闭（reset 每对话一次，长跑常开会把磁盘翻倍）。逐题 `retrieved` 已在 `results.jsonl` 里，快照只用于查内部存储状态。

### run 元数据与可比性

- `run` 新增 `--driver-llm` / `--adapter-version`；`metrics.json` 记 `driver_llm` / `judge_model` / `adapter_version` / `dataset_fingerprint` / `schema_version`，用于回溯"这次分数是用什么跑出来的"。
- `compare` 新增 `--require-comparable` / `--tag`：前者做可比性前置校验（关键维度不一致 → 不可比，禁止当 A/B），后者决定实验目录 `compare_<name>_<platform>_<ts>/` 的 `<name>` 段。
- 数据集指纹 `membench.data.schema.dataset_fingerprint(path) -> {"sha256", "bytes", "name"}`；可用 `membench.data.info.resolve_available_path(key)` 解析某数据集当前可用路径（内置优先，其次 `datasets_dir()`）。
- **token 消耗记账**：`metrics.json` 的 `token_usage` 块记全流程累计（`ingest_input_tokens` / `query_input_tokens` / `query_output_tokens` / `ingest_llm_calls` / `query_llm_calls` + `n_dialogues` / `n_questions`）；run 结束控制台直接打印一行摘要，run 报告有「Token 消耗（全流程）」表，`evaluate` 报告给分数据集消耗 + 合计（`evaluate_summary.json` 的 `token_usage`）。适配器不填 = 0 且标注「未上报」，不是真实零消耗；该块不参与评级 / 门禁 / 可比性校验。

### `--shards` 的分片维度

- `--shards N --shard-index I` 配 `fuse` 融合同一评测的多个分片 run；`--shard-by` 决定分片维度。分片产物**只写本片负责的对话目录 + 账本 `runs/<dataset>/shards.json`**，数据集级 `report/` 与任务级 `reports/` 一律由 `fuse` 补齐（分片产物不是"完整实验"）；跨目录分片时实验目录名末尾带 `_s<I>`（`--limit` 另带 `_l<N>`），**各分片可共用同一个 `--report-out`**（同秒起跑不撞名），同片重跑落同名目录仍被非空守卫拒绝。
- **多片共享一个任务目录**（`--task-dir`，见工作流 2.6）是另一种写法：不新建目录、只允许 `--shard-by dialogue`（题目级分片会让多片写同一对话目录互相覆盖），同片重复跑在同一棵树里也会被账本拒绝。
- 分片产物里的对话级报告**顶部标了「分片 i/N（按 dialogue|question）——仅覆盖部分题目」**，题目数写成 `14（分片 i/N，总量见融合报告）`——引用分数时别把本片题数当整场题数。
- `--shard-by question`（默认）按**全局题号**分片（`i % N == I`），不含入选题的对话整段跳过——适合"少量超长对话承载大量题"的集，ingest 真省。
- `--shard-by dialogue` 按**对话整段**分片（`d % N == I`），分片之间 ingest 互不重复——"几乎每个对话都有题"的 ingest 密集型数据集（如 office、BEAM-1M）**必须用这个**；用题目级分片会让每个分片仍对全部对话 ingest，成本 ×N 反而更慢。
- 两种维度都能用 `fuse` 融合，但**不许混维融合**（fuse 会明确报错）。
- **并行分片要实例隔离**：分片是"进程级"并行——每个分片必须独占一个记忆实例（HTTP 桥 = 每分片独立桥进程，`BRIDGE_PORT` / `BRIDGE_DATA_DIR` 各自隔离；Python 适配器 = 每分片独立进程）。runner 对每个 dialogue 调 `reset()`，共享同一实例并行会互相清库（跨片串扰）；只有一个实例时请串行跑分片。
- **分片数上限与 `--shards max`**：dialogue 模式上限=数据集对话数、question 模式上限=题目数；传超过上限的整数会**按上限执行**（不报错）；`--shards max` 自动按上限取最大；索引越过有效分片数的进程安全跳过（不产生空 run）。
- **分片 A/B（大数据集对比）官方通路**：`task init --arms 2` 建双臂任务目录 → 两侧各片 `run --task-dir <任务目录>/A|B --shards N --shard-by dialogue`（每片独立实例并行）→ 两侧各自 `fuse --task-dir <任务目录>/A|B` 就地补齐 → `compare --task-dir <任务目录>` 就地出对比报告；多数据集汇总仍可用 `evaluate-compare --baseline-run <key>=<A 侧完整实验目录或 runs/<dataset>> --candidate-run <key>=<B 侧同上>`。融合产物透传全部可比性元数据，可直接参与 `compare --require-comparable` 复验。
- **多数据集对比的可比性**：报告附「两侧可比性（逐数据集）」（两侧 `dataset_sha256 / judge / judge_model / driver_llm / platform` 不一致记「关键」、两侧都未标注记「关键缺失」、检索口径等记「次要」；`adapter / adapter_version` 与 `commit` 是被测变量，差异属预期）；两侧都复用的数据集对在**开跑前**先打印问题清单。要做成"能归因的 A/B"，两侧就都带上 `--driver-llm-a/-b` 与 `--adapter-version-a/-b`——缺了只能在报告里记「无法核验」，事后无法补记。

## 结果解读准则

- 报告中的比率型指标以 **"评价（百分比）"** 形式呈现（如 `优秀（92.3%）`），四级：优秀 / 良好 / 中等 / 较差；报告开头有 **综合评级**（六项质量指标连续点数均值，1.0~4.0，档内随数值线性变化，满分 4.0）。效率成本指标（延迟/token/调用次数）依赖部署环境，不参与评级；**诊断指标**（检索召回率@5 / 过度拒答率）仅展示与归因、不参与评级。各数据集特有的分数参照区间用 `dataset info <key>` 查询。
- 办公 Agent 的**一票否决指标是 `stale_rate`**（陈旧召回，目标 ≤0.03）；红线指标是 `abstention_accuracy`（PII/凭证拒答，目标 =1.0）。
- 先看 report.md 的"失败样例 Top10"定位短板题型，再决定优化方向；每次优化只改一个变量。
- 背景知识与指标方法论见 `docs/Agent记忆系统深度研究报告.md` 第 4 章。
