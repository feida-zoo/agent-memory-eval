# membench — Agent 记忆系统评测工程

membench 是一个面向 **agent 记忆系统** 的评测框架（Python 3.10+，纯标准库优先，仅依赖 matplotlib 出图；HTTP 适配器另需 requests）。它回答三个问题：

1. **run**：我的记忆系统在一套多会话对话数据集上表现如何？（六维指标 + 中文报告 + 图表）
2. **compare**：这次优化相比基线是进步还是回归？（可比性前置校验 + 逐指标 diff + PASS/FAIL 门禁）
3. **evaluate**：跨数据集全量评测，宏平均融合出综合评级（办公域 stale 门禁一票否决）。

设计要点：

- **六维指标**：正确性、检索质量、效率成本、记忆动态（知识更新/陈旧召回）、拒答、可扩展性。
- **判分双轨**：启发式判分（默认，零成本、确定性）+ LLM-as-Judge（可选，OpenAI 兼容接口，失败自动回退启发式）。
- **适配器插件化**：被测系统实现 3 个方法即可接入；内置 mock / transcript / http 参考适配器，若被测系统的记忆能力本身就是"LLM + 工具调用"且能在 Python 侧调用写入/读取函数，直接用 `AgentLoopAdapter` 复用循环（非 Python 系统见 HTTP 接入路径）。

## 目录结构

```
├── membench/
│   ├── adapters/          # BaseMemoryAdapter + mock / transcript / http 适配器
│   ├── data/              # 数据集 schema、JSONL 加载器、fetch（按需下载）、内置集
│   ├── judges/            # HeuristicJudge / LLMJudge
│   ├── metrics/           # token_f1 / bleu1 / recall@k + 六维汇总
│   ├── paths.py           # 产物根 / runs 目录判定（MEMBENCH_OUTPUT_DIR > skill > 工程根）
│   ├── layout.py          # 实验目录命名与落位（eval_* / ab_* / matrix_* 的公开 helper）
│   ├── runner.py          # Runner: 执行评测, 写 run 目录
│   ├── agent_loop.py      # AgentLoopAdapter: 复用 ingest/query 的 tool-call 循环
│   ├── doctor.py          # check_target: HTTP bridge 三端点契约自检（纯标准库）
│   ├── fuse.py            # fuse: 分片 run 合并为完整 run（指标重算 + 门禁）
│   ├── compare.py         # compare_runs: 可比性前置校验 + diff + 回归门禁
│   ├── matrix.py          # compare_matrix: 多 run 矩阵对比（platform × commit）
│   ├── userdata.py        # 用户测试集管理: install / status / delete / resolve
│   ├── report.py          # report.md + 图表（matplotlib Agg，可选）+ 报告导出
│   └── cli.py             # CLI: run / evaluate / fuse / compare / matrix / doctor /
│                          #      adapter / user-data / list-datasets / dataset info|fetch
├── tests/                 # pytest（含 mock-weak vs mock-strong 端到端）
└── examples/
    ├── quickstart.py      # 最小接入示例
    ├── optimize_loop.md   # 优化迭代闭环教程
    └── http_bridge_node/  # 非 Python（Node/Electron）系统 HTTP bridge 参考模板
```

## 快速开始

```bash
# 1. 用内置强 mock 适配器跑内置示例数据集（--report-out 落自包含实验目录）
python -m membench.cli run --adapter mock-strong \
    --dataset membench/data/sample_dataset.jsonl \
    --judge heuristic --out runs/ --report-out output/

# 2. 再跑弱 mock（模拟“优化前”）
python -m membench.cli run --adapter mock-weak \
    --dataset membench/data/sample_dataset.jsonl --out runs/ --report-out output/

# 3. 对比两个 run：可比性前置校验 + 门禁结论
python -m membench.cli compare --baseline runs/<run_weak> --candidate runs/<run_strong> \
    --require-comparable --tag weak-vs-strong --report-out output/

# 查看内置数据集
python -m membench.cli list-datasets
```

## 全量评估（多数据集融合总分）

`evaluate` 对所有数据集全量测试，跨数据集**宏平均**融合出综合评级：

```bash
# 默认全量 = 办公域槽位（用户集>内置office）+ locomo + longmemeval_oracle + gatemem_office + BEAM-1M
python -m membench.cli evaluate --adapter <你的adapter> --judge llm --report-out <报告父目录>

python -m membench.cli evaluate --adapter <a> --quick      # 跳过 BEAM，分钟级日常回归
python -m membench.cli evaluate --adapter <a> --beam-10m   # 用 BEAM-10M 替换 1M，全集终验
python -m membench.cli evaluate --adapter <a> --include-lme-s  # 追加 LongMemEval-S（大体积，与 oracle 分差反映检索/精确率损失）
```

## 双臂多数据集融合 A/B 对比

`evaluate-compare` 一次跑两套 adapter × 多数据集，直接产出**逐数据集 diff + 跨数据集融合 diff + 门禁 + 对比图**，落 `<报告父目录>/ab_<tag>_<ts>/` 自包含结构（A/B 两侧各自带各数据集 run 副本 + compare_report.md + 两张对比图）：

```bash
python -m membench.cli evaluate-compare \
    --baseline-adapter <adapter_A> --candidate-adapter <adapter_B> \
    --judge llm --report-out <报告父目录> --tag "优化前vs优化后"

# 同样支持 --quick / --beam-10m / --include-lme-s；两侧可分别指定 driver_llm / adapter_version
python -m membench.cli evaluate-compare \
    --baseline-adapter A --candidate-adapter B \
    --driver-llm-a gpt-4o --driver-llm-b gpt-4o-mini \
    --adapter-version-a v1 --adapter-version-b v2 \
    --quick
```

门禁口径与 `compare` 一致：accuracy 下降 > `--acc-threshold`（默认 0.02）→ FAIL；stale_rate 上升 > `--stale-threshold`（默认 0.02）→ FAIL；latency_p95 相对上升 > `--latency-warn`（默认 30%）→ WARN。报告默认显示**逐数据集 diff 表** + **逐数据集题型 diff** + **双臂融合 diff 表** + **per-dataset 门禁明细**（单数据集 spike 不被融合平均抹平）+ **未定义指标标注**（避免把"数据集未标注 evidence"误读为"召回率为 0"）；比率型指标以百分比呈现（如 `81.5%`），差值为百分点（如 `+2.3pt`）。指标"未计算"统一记 `null`，与真实 0 区分；诊断指标（检索召回率@5 / 过度拒答率）在 diff 与对照表中可见但不参与评级。旧版 run 的 capabilities 未落账时报告会按数据集/逐题记录自动推断并在头部标注来源。

融合口径（重要，读总分前必读）：

- **宏平均**：六项质量指标（accuracy/token_f1/bleu1/evidence_recall/abstention_accuracy/stale_rate）先在各数据集内算分，再跨集等权平均；**指标只在具备对应标注能力的数据集上参与**——无 gold_stale 的集（locomo/longmemeval）不参与 stale_rate，无 evidence 标注的集（longmemeval/beam）不参与 evidence_recall，无 abstention 题的集不参与 abstention_accuracy。绝不用"记 0"稀释。此外**诊断指标**（检索召回率@5 需 has_evidence；过度拒答率全数据集参与）同样跨集融合展示，但不参与综合评级与门禁。
- **门禁一票否决独立于总分**：办公域槽位 stale_rate > 0.03 → 总评"不达标"，退出码 1（可接 CI），无论综合评级多高。
- **按需下载的公开集必须先 fetch**：`locomo` / `longmemeval_oracle` / `gatemem_office` / `beam_1m`（或 `beam_10m`）都没有内置文件，`evaluate` 从 `~/.membench/datasets/` 解析可用路径，缺失时报错提示先 `python -m membench.cli dataset fetch <key>`（`--quick` 不检查 BEAM 槽位）。judge=heuristic 跑 BEAM 会警告（rubric 型题目规则判分失真，建议 llm/client）。
- **产物**：`--report-out DIR`（DIR 是实验目录的**父目录**）时落 `DIR/eval_<adapter>_<platform>_<ts>/`，内含 `evaluate_report.md`、`evaluate_summary.json`、`fig_dataset_overview.png` 与 `runs/<各数据集 run_id>/`（自包含）；不传时保持历史位置 `<out_dir>/evaluate_<ts>/`。

run 产物（`runs/<run_id>/`）：`results.jsonl`（逐题原始记录）、`metrics.json`、`report.md`、`fig_category_accuracy.png`、`fig_latency_hist.png`。
report.md 中比率型指标以 **"评价（百分比）"** 呈现（如 `优秀（92.3%）`，四级：优秀/良好/中等/较差），开头附 **综合评级**（六项质量指标连续点数均值，1.0~4.0，档内随数值线性变化，阈值 3.5/2.5/1.5）；效率成本指标不参与评级（依赖部署环境）；**诊断指标**（检索召回率@5、过度拒答率）仅展示与归因，不参与评级——前者是前 5 条检索的证据覆盖率，后者是非拒答题的误拒比例，与 `evidence_recall` / `abstention_accuracy` 配合分诊（检索 vs 生成、漏拒 vs 误拒）。评级规则见 `membench/ratings.py` 的 `METRIC_THRESHOLDS` 表，可按团队标准调整。
compare 产物（`output_root()/ab_<tag>_<ts>/`，见「产物布局」）：A/B 各自带 run 副本 + `compare_report.md` + `fig_metric_diff.png`（质量指标高亮明显变化 + 效率成本）+ `fig_category_diff.png`（分题型准确率对比）。

使用 LLM 判分（可选）：

```bash
python -m membench.cli run --adapter mock-strong \
    --dataset membench/data/sample_dataset.jsonl \
    --judge llm --llm-base-url https://api.example.com/v1 \
    --llm-api-key $API_KEY --llm-model gpt-4o-mini \
    --llm-timeout 120   # 可选：判分请求超时秒数（默认 90s，适配推理模型）
```

题目级并发（可选）：`run` / `evaluate` 均支持 `--jobs N`（默认 1）。ingest 阶段
始终串行（对话时序与记忆状态敏感），仅 query+judge 阶段并发；`results.jsonl`
的记录顺序与 `--jobs 1` 完全一致。`--jobs > 1` 要求你的 `adapter.query` 与
judge 线程安全（内置 mock adapter 与 LLMJudge 均线程安全）。

## 大数据集：冒烟与分片

冒烟（快速验证链路，可选）：`run --limit N` 每个数据集只取前 N 道题。
run_id 带 `_limit{N}` 后缀、metrics.json 记入 `"limit": N`，冒烟结果不会
被误当全量结果对比。

数据集级分片（BEAM / LongMemEval 这类大集，可选）：

```bash
# 先下载（落 ~/.membench/datasets/beam_1m.jsonl）：
#   python -m membench.cli dataset fetch beam_1m
# 按对话分 4 片（整段对话归片 d % 4，分片间不重复 ingest；可在多台机器上并行）
python -m membench.cli run --adapter my-sys --dataset ~/.membench/datasets/beam_1m.jsonl --shards 4 --shard-index 0 --shard-by dialogue
...
python -m membench.cli run --adapter my-sys --dataset ~/.membench/datasets/beam_1m.jsonl --shards 4 --shard-index 3 --shard-by dialogue

# 合并为完整 run（校验分片齐全、adapter/dataset/shard 维度一致、题目无重复；
# 指标重新计算，stale 门禁对融合结果同样生效，FAIL 时退出码 1）
python -m membench.cli fuse --runs runs/..._dshard0of4,runs/..._dshard1of4,... --out runs/beam_1m_full
```

`--shards` 与 `--limit` 可组合（先分片再截断），用于分片链路的廉价验证。

**分片维度怎么选（`--shard-by`，重要）**：

- `--shard-by question`（默认）按全局题号 `i % N == shard_index` 分片；不含入选题的对话整段跳过（ingest 也省掉）。适合"少量超长对话承载大量题"、能整块跳过对话的场景。
- `--shard-by dialogue` 整段选取 `d % N == shard_index` 的对话、保留其全部题目，**分片之间 ingest 互不重复**。"几乎每个对话都有题"的 ingest 密集型数据集（内置 office：20 对话 / 86 题；BEAM-1M：35 对话 × 20 题）必须用它——题目级分片会让每个分片仍对全部对话 ingest，成本 ×N，分片反而更慢。
- 两种维度的分片都能用 `fuse` 融合回完整 run，但不许混维融合（fuse 会明确报错）。

## 可复现性元数据

每个 run 目录的 `metrics.json` 除了六维指标，还记一组元数据，回答"这次分数是用什么跑出来的"：

- `schema_version`：元数据 schema 版本（当前 `"0.2"`）；
- `adapter` / `platform` / `commit` / `dataset` / `dataset_path` / `dataset_source`；
- `driver_llm`：驱动被测系统的 LLM 标识（`--driver-llm`，自由文本）；
- `judge` / `judge_model`：判分器名与判分模型（`judge_model` 缺省时自动取 judge 对象的 `.model`）；
- `adapter_version`：适配器实现版本 / commit（`--adapter-version`）；
- `dataset_fingerprint`：`{"sha256": 内容 sha256 前 16 位, "bytes": 字节数, "name": 文件名}`，`compare` 的可比性校验据此机检"数据集是否变了"；
- `adapter_meta`：`--adapter-meta "model=deepseek-v3;temp=0"` 自由文本，原样记入；
- `adapter_describe`：adapter 定义了可选 `describe() -> dict` 时自动调用记入（调用失败记 null，不中断评测）；
- `experiment_dir`：仅当 `--report-out` 给定时有，指向本次实验的自包含目录。

```bash
python -m membench.cli run --adapter my-system --dataset membench/data/office_dataset.jsonl \
    --driver-llm deepseek-v3 --adapter-version $(git rev-parse --short HEAD) \
    --judge llm --llm-model gpt-4o-mini --report-out output/
```

建议至少把驱动模型、判分模型、适配器版本、数据集指纹落全——`compare --require-comparable` 靠这些维度做前置校验，字段缺失会记入"无法核验"。

## A/B 可比性前置校验

`compare` 只**测量差异**，不**判定**两臂是否只差一个变量。为了让"驱动/判分模型、适配器、数据集变了却照给 PASS"不再发生，`compare` 在算 diff 之前先做**可比性前置校验**：逐维度比对两个 run 的 metrics.json 元数据，结论作为门禁列表的**首位**（`comparability`）。

维度分三级：

| 级别 | 维度 | 不一致的后果 |
|---|---|---|
| 关键 | `adapter`、`adapter_version`、`dataset_sha256`（取 `dataset_fingerprint.sha256`）、`judge`、`judge_model`、`driver_llm`、`platform` | 非 strict 下门禁降级 WARN；`--require-comparable`（strict）下直接 FAIL |
| 次要 | `limit`、`shards`、`shard_index`、`n_questions`、`adapter_meta` | 降级 WARN，不 FAIL |
| 提示 | `commit`、`dataset_path`、`dataset` | 仅记录（`same=false`），不影响结论 |

- 关键维度**任一值缺失**（如旧 run 没有 `driver_llm`）记入 `unverifiable`：非 strict 下不判不可比，strict 下与不一致同等处理（判 FAIL）——没填就核验不了，strict 模式不允许"没填当通过"。
- `commit` 不同是 A/B 的被测变量本身，**不算**不可比。

门禁语义：

- **PASS**：关键维度全部一致（且无次要不一致）；
- **WARN**：存在关键维度不一致（非 strict）或次要维度不一致——结果可疑但可继续；
- **FAIL**：strict 下存在关键维度不一致或无法核验。

不可比原因在两处可见：CLI 输出逐条 `[membench] 不可比原因: [关键] <维度>: <说明>`；`ab_<tag>_<ts>/compare_report.md` 的「可比性前置校验」小节给出逐维度对照表（A / B / 是否一致）与「不可比原因」列表。

```bash
python -m membench.cli compare --baseline runs/<run_a> --candidate runs/<run_b> \
    --require-comparable --tag my-opt --report-out output/
```

`--tag` 决定 `ab_<tag>_<ts>/` 目录名（缺省由两个 run_id 拼）。退出码：verdict=FAIL → 1，否则 0。

## 判分健壮性说明

- LLM judge 输出按三级解析：整体 JSON → 抠取首个 JSON 对象 → 回退启发式判分并打 `llm-fallback` 标记。
- 使用 LLM 判分时，metrics.json 顶层记录 `judge_fallback: {count, ratio}`；报告头部展示回退比例，**ratio > 5% 时报告输出警示行**（判分质量可能退化，建议抽查复核）。
- stale 红线判定为确定性规则：旧值命中且新值未命中才判 stale（回答中引用旧值作为变更上下文、同时给出新值的，不判陈旧）。

## 用户测试集管理

`--dataset` 现在是可选参数。不传时按优先级自动选择测试集：

1. 显式 `--dataset <路径>`（来源标签 `user-specified`）；
2. 已安装的用户测试集（来源标签 `user`）；
3. 内置 `office_dataset.jsonl`（来源标签 `builtin-office`）。

run 开始时会在日志中打印实际选用的测试集及其来源，并写入 metrics.json 的
`dataset_source` 字段与 report.md 的「评测配置」表。

```bash
# 安装用户测试集（先按 schema 完整校验，失败则不写入）
python -m membench.cli user-data install /path/to/my_dataset.jsonl
# 查看状态（路径 / 安装时间 / 来源 / dialogue 与题目数）
python -m membench.cli user-data status
# 删除用户测试集
python -m membench.cli user-data delete
```

用户测试集默认放在 `./user_dataset/`（`user_dataset.jsonl` + `metadata.json`），
可用环境变量 `MEMBENCH_USER_DATA_DIR` 改到别的目录。

## 平台与 commit 维度

跨平台 / 跨版本对比时，用 `--platform` 与 `--commit` 给 run 打标签：

```bash
python -m membench.cli run --adapter mock-strong --platform wx --commit a1b2c3
```

- `run_id` 变为 `{adapter}_{platform}_{dataset}_{timestamp}`（缺省 platform 为 `default`）；
- `metrics.json` 元数据新增 `platform` / `commit` / `dataset_source` 字段；
- `report.md` 开头固定有一张「评测配置」表：测试集路径、测试集来源、平台、commit、判分器。

跨版本（commit）对比时，把两次 run 的 `--driver-llm` / `--adapter-version` 也填上，`compare` 的可比性前置校验才能区分"结果变化来自被测系统"还是"来自换了驱动模型"。

## 矩阵对比（多 run / platform × commit）

`matrix` 子命令对多个带标签的 run 做矩阵对比：

```bash
python -m membench.cli matrix \
  --runs "wx@a1b2c3=runs/dir1" "fs@a1b2c3=runs/dir2" \
         "wx@d4e5f6=runs/dir3" "fs@d4e5f6=runs/dir4" \
  [--out output/matrix_xxx] [--report-out output/]
```

- `--out` 直接指定完整实验目录（优先）；`--report-out DIR` 是实验目录的**父目录**（给出时落 `DIR/matrix_<ts>/`）；都不给时落 `output_root()/matrix_<ts>/`。

label 推荐 `platform@commit` 格式（也接受任意字符串）。实验目录自包含（各臂 run 副本落 `runs/<label>/`），产出 `matrix_report.md` +
`matrix_result.json` + `fig_matrix_accuracy.png`：

1. 若标签恰好覆盖 2 平台 × 2 commit，对 accuracy / stale_rate / evidence_recall /
   abstention_accuracy / latency_p95_ms 各生成一张 platform × commit 透视表；
2. 所有 label 的逐指标汇总表（行=指标，列=label）；
3. 所有两两组合的门禁判定（复用 compare 的 PASS/WARN/FAIL 规则）；
4. accuracy 分组条形图（x=commit、hue=platform，或反之取合理者）。

## 产物布局（一次实验一个自包含子目录）

报告与其对应的 run 就近归属在同一个实验目录里，不再与全局 run 混放。产物根由 `membench.paths.output_root()` 统一判定：`MEMBENCH_OUTPUT_DIR` > skill 的 `output/` > 工程根 `output/`；run 默认落 `membench.paths.runs_dir()`（开发环境保持历史行为：工程根 `runs/`）。

```bash
# 单次评测（run / evaluate --report-out DIR）：DIR 是实验目录的父目录
DIR/eval_<adapter>_<platform>_<ts>/
├── report.md / metrics.json / results.jsonl / fig_*.png
└── runs/<run_id>/                      # 本次评测的 run 副本（含分片）
# evaluate 特有：evaluate_report.md / evaluate_summary.json + runs/<各数据集 run_id>/

# A/B 对比（compare --report-out DIR）
DIR/ab_<tag>_<ts>/
├── A/{report.md, metrics.json, fig_*.png, runs/<run_id>/}
├── B/{report.md, metrics.json, fig_*.png, runs/<run_id>/}
├── compare_report.md / fig_metric_diff.png
└── optimization_report.md              # 可选，由你的优化流程写入

# 矩阵对比（matrix：--report-out DIR 为父目录，--out 直接指定完整实验目录）
DIR/matrix_<ts>/
├── matrix_report.md / matrix_result.json / fig_matrix_accuracy.png
└── runs/<label>/                       # 各臂 run 副本
```

- 不传 `--report-out` 时：`run` 只写 `runs/<run_id>/`；`compare` 落 `output_root()/ab_<tag>_<ts>/`；`evaluate` 保持历史位置 `<out_dir>/evaluate_<ts>/`；`matrix` 落 `output_root()/matrix_<ts>/`。
- 命名与落位 helper 是**公开 API**（不必自己重写约定）：`membench.layout` 的 `subdir_for` / `ab_subdir` / `matrix_subdir` / `experiment_path` / `copy_runs_into` / `materialize_run_experiment` / `materialize_ab_experiment` / `materialize_matrix_experiment`；`membench.paths.output_root()` / `runs_dir()`。
- Runner API 使用者传 `Runner(..., report_out=DIR)` 即可让 Runner 自己建 `eval_<adapter>_<platform>_<ts>/`、把 run 复制进 `runs/<run_id>/`，并把实验目录写回 `metrics.json.experiment_dir` 且打印出来。

## 如何接入自己的记忆系统

四条路径，按推荐程度排序：

### 路径一：继承 BaseMemoryAdapter（推荐，最灵活）

只需实现 `reset / ingest / query` 三个方法：

```python
from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
from membench.runner import Runner

class MyAdapter(BaseMemoryAdapter):
    name = "my-system"

    def reset(self): self.client.clear()                      # 每个 dialogue 前清空
    def ingest(self, messages):                               # 写入一段会话
        self.client.add(messages)
        return IngestStats(input_tokens=0, llm_calls=0, latency_ms=0.0)
    def query(self, question):                                # 基于记忆回答
        ans, docs = self.client.ask(question)
        return QueryResult(answer=ans, retrieved=docs,
                           latency_ms=0.0, input_tokens=0,
                           output_tokens=0, llm_calls=0)

Runner(MyAdapter(), "membench/data/sample_dataset.jsonl").run()
```

完整可运行示例见 `examples/quickstart.py`。token / llm_calls 填不了就填 0（报告中视为 n/a）。

### 路径二：实现 REST 契约，用 HttpAdapter

如果你的记忆系统已有 HTTP 服务，实现三个端点即可，无需写 Python：

- `POST /reset`：清空全部记忆状态
- `POST /ingest`：body `{"messages": [{"role", "content", "timestamp"}]}`
- `POST /query`：body `{"question": "..."}`，返回 `{"answer": str, "retrieved": [str], ...}`

然后在 Python 侧用内置 `HttpAdapter` 指向你的服务地址，照常跑 `Runner`。

#### 接入自检（doctor）

正式评测前，先用 `doctor` 做连通性 + 契约校验，避免把"适配器/契约写错"误判成"记忆能力差"：

```bash
python -m membench.cli doctor --target http://127.0.0.1:8787 [--timeout 10.0] [--api-key K]
```

按 `reset → ingest → query` 顺序探测：

- `reset`：2xx → PASS；
- `ingest`：2xx → PASS；响应为 JSON dict 时额外校验可选字段 `input_tokens` / `llm_calls` 为数值（缺失可接受）；
- `query`：须 2xx + JSON dict + 字符串 `answer`；`retrieved` 若出现且非 null 须为字符串列表。

单端点失败**不抛异常、不短路**（`reset` 未实现但读写正常不会误判为不可用）。底层为纯标准库的 `membench.doctor.check_target(base_url, timeout=10.0, api_key=None) -> {"ok", "base_url", "checks": [{"name","status","detail"}]}`（`status` ∈ `PASS`/`FAIL`/`SKIP`，`ok=True` 表示无任何 FAIL）。CLI 逐项打印 `[membench] doctor <name>: <status> — <detail>`，`ok` → 退出码 0，否则 1。

非 Python（Node/Electron）系统的完整 checklist 与可复制模板见 `examples/http_bridge_node/`（skill 包内配套文档 `references/http_bridge_guide.md`）。

### 路径三：复用 agent-loop 循环（`AgentLoopAdapter`）

若被测系统的 ingest（喂会话让 LLM 调写入工具落库）与 query（让 LLM 调检索工具作答）本身就是**同一套 tool-call 骨架**（拼 prompt → 调模型 → 分发工具 → 累计 usage → 回填证据），不要各写一遍，直接用 `membench.agent_loop.AgentLoopAdapter`，只给 `{llm_client, tools}`：

> **仅适用 Python 侧**：LLM 客户端与工具 handler 都必须是 Python 可调用对象；非 Python（Node/Electron）系统的 tool-call 循环留在其自身进程内，membench 无法复用，走路径二的 HTTP 接入。

```python
from membench.agent_loop import AgentLoopAdapter
from membench.runner import Runner

adapter = AgentLoopAdapter(
    llm_client="my_agent.llm:LLMClient",   # module:ClassName（延迟导入、无参实例化），与 --judge client 同约定
    ingest_tools=[{"name": "write_memory", "schema": {...}, "handler": write_memory}],
    query_tools=[{"name": "retrieve_memory", "schema": {...}, "handler": retrieve}],  # 缺省复用 ingest_tools
    system_prompt="你是一个带长期记忆的办公助手。",
)
Runner(adapter, "membench/data/office_dataset.jsonl", judge="heuristic",
       driver_llm="deepseek-v3").run()
```

- 确切签名：`run_tool_loop(client, messages, tools, max_rounds=8, system_prompt=None) -> dict`；`AgentLoopAdapter(llm_client, ingest_tools, query_tools=None, system_prompt=None, name="agent-loop", max_rounds=8, retrieved_from=None, ingest_prompt=None, query_prompt=None)`。
- `llm_client` 三选一：带 `chat_with_tools(messages, tools)` 的对象（优先，走原生 tool-call）> 带 `chat` / `complete` 的对象或可调用对象（走纯文本 JSON 约定）> `"module:ClassName"` 类路径字符串。`tools` 元素为 `{"name", "schema", "handler"}`；handler 收解析后的 `arguments` dict，返回 str 或 dict（handler 抛异常会被折叠成观察文本回填，**不中断**循环）。
- membench 负责循环、`IngestStats` / `QueryResult` 统计与 `retrieved` 回填。回填由 `retrieved_from(question, loop_result) -> list[str]` 决定，缺省 `default_retrieved_from` 识别工具名含 `retrieve` / `search` / `recall` 的调用并取其结果——这决定 `evidence_recall` 等证据指标是否有意义：**检索工具名要对得上关键词、结果要是被检索到的记忆原文**（JSON 列表或一行一条），否则请自传 `retrieved_from`。
- `AgentLoopAdapter.describe()` 返回驱动 LLM 标识与工具名列表，自动落入 run 元数据 `adapter_describe`。

### 路径四：导出对话日志，用 TranscriptAdapter

最低成本路径：把你系统线上真实的「记忆写入/检索」日志导出为离线 transcript（对话 + 每题实际检索到的记忆条目），用 `TranscriptAdapter` 回放注入。适合**不改系统、先量现状**的场景，也适合复现线上 badcase。

### 适配器持久化（adapter install）

业务适配器（如 `my-office-adapter`）写好一次后即可落盘，之后任何 session 无需重新生成或设环境变量即可使用：

```bash
# 落盘到持久化目录（默认 ~/.membench/adapters/，可用 MEMBENCH_ADAPTERS_DIR 覆盖）
python -m membench.cli adapter install ./my_office_adapter.py [--name my-office-adapter]

# 查看内置 + 持久化适配器（表格含注册名/类名/安装时间/来源路径）
python -m membench.cli adapter list

# 删除
python -m membench.cli adapter remove my-office-adapter
```

`adapter install` 只负责校验（语法可解析）+ 落盘 `.py` 与元信息 `.json`，并提示"下次运行时自动加载"；此后每次 `membench` 命令启动都会自动加载持久化目录中的全部 `*.py` 适配器（按文件名排序；单个损坏的适配器仅打印 stderr 警告，不阻塞其他适配器）。

环境变量方式（`MEMBENCH_ADAPTER_MODULES=mod1,mod2`）仍保留，用于临时/CI 场景。**优先级：后加载的覆盖先加载的——先加载持久化目录，再加载环境变量模块，因此同名注册时环境变量方式优先**，便于临时调试覆盖已安装的适配器。

## 指标说明

| 维度 | 指标 | 定义 | 目标基线 |
|---|---|---|---|
| 正确性 | accuracy | judge 判对比例（heuristic 或 LLM） | ≥ 0.85 |
| 正确性 | token_f1 / bleu1 | 预测与 gold 的 token 级重合度 | 参考项，不门禁 |
| 检索质量 | evidence_recall | 非 abstention 题中，证据子串命中任一检索条目的比例 | ≥ 0.90 |
| 效率成本 | latency_p50 / p95 | query 延迟分位数（毫秒） | 按系统预算自定 |
| 效率成本 | avg_input_tokens / avg_llm_calls | 每题平均注入 token 与 LLM 调用数 | 越低越好 |
| 记忆动态 | stale_rate | knowledge_update 题中命中过期事实且未包含更新值的比例 | ≤ 0.03 |

> 注：stale 的"包含更新值"接受两种口径——gold 片段逐字覆盖，或 gold 内容 token 召回 ≥ 0.8（容忍同义改写，如"下周一下午"→"周一下午"）。旧值仅作为变更上下文被引用、且新值已正确给出时不计为 stale。
| 拒答 | abstention_accuracy | abstention 题中正确拒答的比例 | ≥ 0.80 |
| 可扩展性 | （随数据规模观察上述指标的退化曲线） | 用更大数据集多跑几轮对比 | — |

## 优化迭代工作流

```
run 基线（记全 --driver-llm / --adapter-version / 数据集指纹）
  → 看 report.md 失败样例找短板 → 单变量改动
  → run 候选（同数据集 / 同判分 / 同驱动模型）
  → compare --require-comparable 做可比性前置校验 + 看门禁（PASS/FAIL/WARN）
  → 固化或回滚
```

门禁规则（可用 `--acc-threshold` 等参数覆盖）：

- accuracy 下降 > 0.02 → **FAIL**
- stale_rate 上升 > 0.02 → **FAIL**
- latency_p95 上升 > 30% → **WARN**
- 其余指标变动仅报告

> `compare` 只测量差异，不判定两臂是否只差一个变量——必须自己固定模型 / harness / 数据集，只切优化开关，并用 `--require-comparable` 让 compare 帮你拦下"变量不一致"的对比。

完整教程（含如何扩充黄金集、何时切换 LLM 判分、六阶段优化闭环）见 `examples/optimize_loop.md`。

## 接入公开基准（LoCoMo / LongMemEval / BEAM / GateMem）

membench 内置公开记忆基准的格式转换器，并支持按需下载。查看全部支持的数据集（来源/特点/基准线）：

```bash
python -m membench.cli dataset info            # 列举全部 9 个数据集及简介
python -m membench.cli dataset info locomo     # 单个数据集详细介绍（含基准线：如何解读分数）
python -m membench.cli list-datasets           # 仅列内置数据集
```

**内置 JSONL（开箱即用）只有两个**：`membench/data/office_dataset.jsonl`、`membench/data/sample_dataset.jsonl`。其余全部**按需下载**（`dataset fetch`，默认落 `~/.membench/datasets/`，可用 `MEMBENCH_DATASETS_DIR` 覆盖）：

| key | 简介 | 下载源（文件，2026-10 核实） |
|---|---|---|
| `locomo` | LoCoMo 10 段超长对话 / 1986 题 | GitHub raw `snap-research/locomo/data/locomo10.json` |
| `longmemeval_oracle` | LongMemEval oracle 版 500 题（仅证据会话） | HF `xiaowu0162/longmemeval-cleaned` 的 `longmemeval_oracle.json` |
| `longmemeval_s` | LongMemEval-S 500 题（~50 会话/题，~268MB） | HF 同名 repo 的 `longmemeval_s_cleaned.json`（注意 S 变体文件名是 `..._cleaned.json`，`longmemeval_s.json` 不存在） |
| `gatemem_office` | GateMem 办公三维 17 episodes / 547 检查点 | GitHub raw `rzhub/GateMem/bench/data/office/{episodes,checkpoints}.jsonl` |
| `beam_1m` | BEAM 1M：35 chats × ~1M token，约 700 题 | GitHub raw `mohammadtavakoli78/BEAM/chats/1M/...` |
| `beam_10m` | BEAM 10M：10 chats × ~10M token，约 200 题（全集终验） | GitHub raw `mohammadtavakoli78/BEAM/chats/10M/...` |

（另有 `statemembench` 暂不可用：数据未公开，无法下载。）

```bash
python -m membench.cli dataset fetch locomo            # 逐个下载；幂等，--force 可重下
python -m membench.cli dataset fetch longmemeval_oracle
python -m membench.cli dataset fetch beam_1m
# 全部可下载 key：locomo / longmemeval_oracle / longmemeval_s / gatemem_office / beam_1m / beam_10m
```

`dataset fetch` 幂等：产物已存在且能被 `load_dataset` 解析时直接返回；未知 key → 退出码 2（并列出可下载的 key）；下载失败 → 打印消息、退出码 1。`evaluate` 全量评估会从下载目录解析这些集，缺失时报错提示先 `dataset fetch <key>`。

手动转换（自选数据源时）：

```bash
# -o 与 dataset fetch 的落盘位置保持一致（~/.membench/datasets/<key>.jsonl），
# 这样 evaluate / resolve_available_path 能直接解析到（<key> 用注册表 key）。

# LoCoMo（Snap Research，10 个超长对话，1986 题，5 类题型含对抗/拒答题）
curl -fsSL -o /tmp/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
python -m membench.data.convert_locomo /tmp/locomo10.json -o ~/.membench/datasets/locomo.jsonl

# LongMemEval（ICLR 2025，500 题，5 类能力；oracle 版 ~5MB 适合冒烟，
# s 版 ~250MB 是公开榜单标准，m 版 ~2GB）
curl -fsSL -o /tmp/longmemeval_oracle.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
python -m membench.data.convert_longmemeval /tmp/longmemeval_oracle.json -o ~/.membench/datasets/longmemeval_oracle.jsonl

# BEAM（arXiv:2510.27246）与 GateMem（办公三维）：见 dataset fetch 与转换器 docstring
python -m membench.data.convert_beam <chats根目录> --scale 1m -o ~/.membench/datasets/beam_1m.jsonl
python -m membench.data.convert_gatemem episodes.jsonl checkpoints.jsonl --ignore-privacy-refuse -o ~/.membench/datasets/gatemem_office.jsonl

# 评测（LoCoMo 为英文长对话，建议先用 --max-dialogues 1 冒烟）
python -m membench.cli run --adapter mock-strong --dataset ~/.membench/datasets/locomo.jsonl --out runs/
```

题型映射与已知限制：

| 基准 | 题型映射 | 已知限制 |
|---|---|---|
| LoCoMo | cat1→multi_hop, cat2→temporal, cat3→open_ended, cat4→single_hop, cat5→abstention（gold 为 `<ABSTAIN>` 哨兵） | evidence 自动从 dia_id 解析为原文；约 6% 标注质量争议是数据集本身已知问题 |
| LongMemEval | user/assistant→single_hop, multi-session→multi_hop, temporal-reasoning→temporal, knowledge-update→knowledge_update, preference→preference | oracle 版无 per-turn 证据（recall 恒 0）；**S 版 evidence 已从 answer_session_ids 解析为 session 级原文**；无 gold_stale（stale_rate 恒 0）；cleaned 版无拒答题 |
| BEAM | 10 类题型→membench 7 类（information_extraction→single_hop、multi_session_reasoning→multi_hop、event_ordering/temporal_reasoning→temporal、knowledge_update/contradiction_resolution→knowledge_update、instruction_following/preference_following→preference、summarization→open_ended、abstention→abstention） | evidence 为空；rubric 未利用；gold_stale 取 potential_confusion（描述性文本） |
| GateMem | (utility,answer)→single_hop、(privacy,refuse)→abstention、(privacy,answer_redacted)→single_hop、(safety,no_memory)→abstention | asker 身份、leak_targets、as_of_turn_id 截断语义未建模；内置集 140 道 privacy-refuse 题已标记 `ignore=true`（单用户场景跳过，见下节） |

**使用建议**：公开基准做横向对比与回归仪表盘；但厂商公布的 LoCoMo/LongMemEval 分数可比性差（判分器、检索预算、作答模型不同即可造成 ±20 分以上摆动），最终验收仍以你自建的领域黄金集为准。两者结合：公开基准按周跑，黄金集按提交跑。

## FAQ

**Q: compare 的两次 run 有什么前提？**
A: 同一数据集（指纹一致）、同一判分器与判分模型、同一驱动 LLM 与适配器实现，只变被测系统；否则 diff 无法归因。`compare` 会做可比性前置校验，关键维度不一致时给出 WARN/FAIL；要强制拦截用 `--require-comparable`（不可比原因见 CLI 输出与 `compare_report.md` 的「可比性前置校验」小节）。

**Q: heuristic 判分把我的正确答案判错了怎么办？**
A: 启发式是 token 包含匹配，表述差异大时会误判。先跑 `--judge llm` 复核；长期方案是优化黄金集答案表述或改用 LLM 判分（注意对基线重跑保持同尺比较）。

**Q: 我的系统拿不到 token / llm_calls 统计？**
A: 填 0 即可，报告中对应项视为 n/a，不影响 accuracy 等核心指标。

**Q: 如何把门禁接入 CI？**
A: `compare` 子命令在 verdict=FAIL 时以退出码 1 退出，直接放进流水线即可。

**Q: 支持并发评测吗？**
A: 当前 Runner 为顺序执行（记忆系统有状态，顺序最安全）。题目级并发可用 `--jobs N`（仅 query+judge 阶段并发，ingest 仍串行）。数据量大时建议按 dialogue 拆分数据集并行跑多个进程，再合并 results.jsonl。

## 办公 Agent 专用黄金集（office_dataset.jsonl）

针对通用办公助手类 Agent 内置的领域数据集：20 个对话、86 题，覆盖二十类办公场景——项目截止日变更、会议决议追溯、写作风格偏好、日程与联系人变动、敏感信息拒答（红线题）、跨周待办跟踪、自动化流程偏好、长跨度谈判时间线、供应商合同、招聘面试、事故复盘、客户投诉升级、年度预算、流程偏好回滚、差旅报销规则、同名项目干扰、凭证保密、远程办公制度、发布计划、OKR 协作。

```bash
python -m membench.cli run --adapter my-system --dataset membench/data/office_dataset.jsonl --out runs/
```

办公场景的指标优先级建议：**stale_rate（陈旧召回）是一票否决项**——过期的截止日和联系人会直接造成办公事故；其次是 abstention_accuracy（涉密信息拒答）与 preference 题准确率（决定产出物是否"像你写的"）。建议将此数据集作为最小验收集，再按你的实际业务对话扩充到 50–100 题。
