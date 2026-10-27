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
│   ├── paths.py           # 产物根判定 output_root（MEMBENCH_OUTPUT_DIR > skill 的 output/ > 工程根 output/）
│   ├── layout.py          # 任务目录 / 实验目录命名与落位（eval_* / compare_* / matrix_*、分片账本的公开 helper）
│   ├── runtime.py         # 实验 / 任务级运行时数据（commands.jsonl + logs/ + env.json，argv 脱敏）
│   ├── runner.py          # Runner: 执行评测，直接写实验目录的 runs/<dataset>/…
│   ├── snapshot.py        # 状态快照与回放：逐对话 manifest / 池源解析、跑前校验、保真比对
│   ├── agent_loop.py      # AgentLoopAdapter: 复用 ingest/query 的 tool-call 循环
│   ├── doctor.py          # check_target: HTTP bridge 必需三端点契约自检（纯标准库）
│   ├── fuse.py            # fuse: 分片产物融合为完整实验目录（就地 --task-dir 或跨目录，指标重算 + 门禁）
│   ├── compare.py         # compare_runs: 可比性前置校验 + diff + 回归门禁（支持就地 --task-dir）
│   ├── matrix.py          # compare_matrix: 多 run 矩阵对比（platform × commit）
│   ├── evaluate.py        # evaluate: 多数据集宏平均融合 + 办公域 stale 门禁
│   ├── evaluate_compare.py # evaluate_compare: 两组 × 多数据集融合 A/B 对比
│   ├── ratings.py         # 综合评级阈值表 METRIC_THRESHOLDS
│   ├── userdata.py        # 用户测试集管理: install / status / delete / resolve
│   ├── report.py          # report.md + 图表（matplotlib Agg，可选）+ 报告导出
│   └── cli.py             # CLI: task init|status / run / evaluate / evaluate-compare /
│                          #      fuse / compare / matrix / doctor / adapter / user-data /
│                          #      list-datasets / dataset info|fetch
├── tests/                 # pytest（含 mock-weak vs mock-strong 端到端）
└── examples/
    ├── quickstart.py      # 最小接入示例
    ├── optimize_loop.md   # 优化迭代闭环教程
    └── http_bridge_node/  # 非 Python（Node/Electron）系统 HTTP bridge 参考模板
```

## 快速开始

`run --dataset` 既接受数据集 key（`office` / `sample` / `locomo` 等，按需下载的集先
`python -m membench.cli dataset fetch <key>`），也接受 JSONL 文件路径；不传时依次回退到
已安装的用户测试集、内置 office 数据集。

```bash
# 1. 用内置强 mock 适配器跑内置示例数据集（产物落 <output 根>/eval_mock-strong_default_<ts>/）
python -m membench.cli run --adapter mock-strong \
    --dataset membench/data/sample_dataset.jsonl --judge heuristic

# 2. 再跑弱 mock（模拟“优化前”）
python -m membench.cli run --adapter mock-weak \
    --dataset membench/data/sample_dataset.jsonl

# 3. 对比两次评测：可比性前置校验 + 门禁结论
#    --baseline/--candidate 传实验目录或其中的 runs/<dataset> 数据集目录
python -m membench.cli compare \
    --baseline output/eval_mock-weak_default_<ts>/runs/sample_dataset \
    --candidate output/eval_mock-strong_default_<ts>/runs/sample_dataset \
    --require-comparable --tag weak-vs-strong

# 4. 改了「读取侧」（检索/排序/回答）代码：用快照回放，免重跑 ingest
#    快照默认导出（适配器声明了快照能力时），不必额外加开关——见「状态快照与回放」
python -m membench.cli run --adapter <你的适配器> --dataset <同一数据集>
python -m membench.cli run --adapter <你的适配器> --dataset <同一数据集> \
    --replay output/eval_<名>_<平台>_<ts> \
    --replay-verify-source output/eval_<名>_<平台>_<ts>/runs/<dataset>   # 回放 + 逐题保真校验

# 4b. A/B 两臂回放同一源：用回放任务，状态按内容去重只存一份（见「回放任务的快照池」）
TASK=$(python -m membench.cli task init --name replay --platform wx --arms 2 --replay)
python -m membench.cli run --adapter <适配器A> --dataset <同一数据集> \
    --task-dir "$TASK/A" --replay output/eval_<名>_<平台>_<ts>
python -m membench.cli run --adapter <适配器B> --dataset <同一数据集> \
    --task-dir "$TASK/B" --replay output/eval_<名>_<平台>_<ts>
python -m membench.cli task status "$TASK"   # 桶数 / 各臂复用新写 / 两臂是否逐对话一致

# 5. 多分片评测：先建任务目录，各分片写进同一棵树，再就地融合（见「大数据集：冒烟与分片」）
TASK=$(python -m membench.cli task init --name beam --platform wx --arms 1)   # stdout 只有这一行路径
python -m membench.cli run --adapter <你的适配器> --dataset <数据集> \
    --task-dir "$TASK" --shards 4 --shard-index 0 --shard-by dialogue
# ... 各分片（--shard-index 1..3）同样写进 "$TASK"；再补齐数据集级与任务级产物：
python -m membench.cli fuse --task-dir "$TASK"
python -m membench.cli task status "$TASK"      # 只读进度：哪片跑完、还缺什么、下一步跑什么

# 查看内置数据集
python -m membench.cli list-datasets
```

> 第 3 步的两次 mock 评测没填 `--driver-llm / --adapter-version`，`--require-comparable` 会按「关键维度缺失、无法核验」判 FAIL（这是 strict 模式的既定口径，不是示例坏了）；想看到 PASS 就先在这些维度上补全元数据再跑（见「可复现性元数据」）。

## 全量评估（多数据集融合总分）

`evaluate` 对所有数据集全量测试，跨数据集**宏平均**融合出综合评级：

```bash
# 默认全量 = 办公域槽位（用户集>内置office）+ locomo + longmemeval_oracle
#            + gatemem_office + LongMemEval-S + BEAM-1M
python -m membench.cli evaluate --adapter <你的adapter> --judge llm --report-out <实验目录父目录>

python -m membench.cli evaluate --adapter <a> --no-lme-s    # 剔除 LongMemEval-S（~268MB、ingest 成本高）
python -m membench.cli evaluate --adapter <a> --no-beam     # 不跑 BEAM
python -m membench.cli evaluate --adapter <a> --beam-10m    # BEAM 换 10M 全集终验（与 --no-beam 互斥）
python -m membench.cli evaluate --adapter <a> --quick       # 跳过 BEAM 与 LongMemEval-S（分钟级回归）
# 显式数据集清单（可重复 --dataset，key 与 JSONL 路径可混用；与预设开关互斥）
python -m membench.cli evaluate --adapter <a> \
    --dataset office --dataset locomo --dataset ~/data/my_set.jsonl

# 汇总模式：复用已跑完的产物（按数据集挨个跑完之后统一出报告）；
# 目标清单里的缺口自动用 --adapter 补跑（纯复用、无缺口时 --adapter/--judge 可省略）。
# DIR 指向 runs/<数据集目录> 最稳；传实验目录时其下必须只有一个数据集目录（见下）
python -m membench.cli evaluate --adapter <a> \
    --dataset office --dataset locomo \
    --from-runs office=output/eval_<名>_<平台>_<ts>/runs/office_dataset \
    --from-runs locomo=output/eval_<另一个>_<平台>_<ts>/runs/locomo
```

结果落 `--report-out` 父目录下的 `eval_<adapter>_<platform>_<ts>/`：每个数据集一棵 `runs/<dataset>/`（数据集级报告在 `report/`，逐对话产物在 `<对话目录>/`），任务级报告在 `reports/`（含 `evaluate_summary.json`）。也可用 `--task-dir DIR` 写进已存在的任务目录（不新建目录，见「产物布局」）。

**汇总模式（`--from-runs KEY=DIR`，可重复）**：目标数据集里已有产物的直接复用、缺失的补跑，最后统一出融合/评级/门禁报告；复用的数据集产物会并入本次实验目录的 `runs/<dataset>/`，整个实验目录可搬移。`DIR` 是**实验目录**（其下恰有一个数据集目录时自动选它，多数据集实验目录请直接指向数据集目录）或**数据集目录**（`<实验目录>/runs/<dataset>`，目录名 = 数据集文件名去扩展名，如 `office_dataset`、`locomo`、`beam_1m`）。**被测对象与评测口径（`adapter / adapter_version / driver_llm / judge / judge_model / 检索口径`）同一侧必须一致**——已有产物之间混用、或补跑参数与已有产物不一致，都在**开跑之前**直接拒绝（对比场景同理，每侧固定一个被测对象，差异变量只应发生在两侧之间），避免跑完才发现汇总不可归因。仅跨产物的 `platform` 差异保留：在控制台与报告顶部**显著标注**。全新评测缺 `--driver-llm / --adapter-version` 时启动即提示——严格可比性（`compare --require-comparable`）会判这些关键维度「无法核验」，且无法事后补记。

说明：默认全量**包含** LongMemEval-S（每题约 50 个干扰会话 / ~115k token 的检索压力
变体，与 oracle 的分差反映检索/精确率损失）；未下载时默认全量会报错提示先
`dataset fetch longmemeval_s`，不需要则用 `--no-lme-s` 剔除。显式 `--dataset`
清单中**第一个数据集承担办公域槽位**（stale 门禁作用域；无 gold_stale 标注时门禁自动
SKIP）；未显式给出时默认清单的办公域槽位规则不变。

## A/B 两组多数据集融合对比

`evaluate-compare` 一次跑两套 adapter × 多数据集，直接产出**逐数据集 diff + 跨数据集融合 diff + 门禁 + 对比图**，落 `<实验目录父目录>/compare_<name>_<platform>_<ts>/` 自包含结构（`A/`、`B/` 各为完整评测产物——含自身 `runs/` 与 `reports/`；对比结论 `reports/comparison.md` 与两张对比图 `reports/fig_per_dataset_diff.png`、`reports/fig_fused_diff.png` 落实验目录的 `reports/`）：

```bash
python -m membench.cli evaluate-compare \
    --baseline-adapter <adapter_A> --candidate-adapter <adapter_B> \
    --judge llm --report-out <实验目录父目录> --tag "优化前vs优化后"

# 同样支持 --no-lme-s / --no-beam / --beam-10m / --quick 与 --dataset 显式清单；
# 两侧可分别指定 driver_llm / adapter_version
python -m membench.cli evaluate-compare \
    --baseline-adapter A --candidate-adapter B \
    --driver-llm-a gpt-4o --driver-llm-b gpt-4o-mini \
    --adapter-version-a v1 --adapter-version-b v2 \
    --quick

# 汇总模式：两侧逐数据集复用已跑完的产物、缺口自动补跑（KEY=DIR 可重复；
# DIR 为 runs/<数据集目录>，或"其下只有一个数据集目录"的实验目录；
# 某侧全部已跑完时可省略该侧 adapter；被测对象与评测口径同一侧一致——
# 混用或补跑参数与已有产物不一致 → 开跑之前直接拒绝）
python -m membench.cli evaluate-compare \
    --baseline-run office=output/eval_<A>_<平台>_<ts>/runs/office_dataset \
    --candidate-run office=output/eval_<B>_<平台>_<ts>/runs/office_dataset \
    --baseline-run locomo=output/eval_<A2>_<平台>_<ts>/runs/locomo \
    --candidate-run locomo=output/eval_<B2>_<平台>_<ts>/runs/locomo \
    --report-out <实验目录父目录> --tag "优化前vs优化后"
```

门禁口径与 `compare` 一致：accuracy 下降 > `--acc-threshold`（默认 0.02）→ FAIL；stale_rate 上升 > `--stale-threshold`（默认 0.02）→ FAIL；latency_p95 相对上升 > `--latency-warn`（默认 30%）→ WARN。报告默认显示**逐数据集 diff 表** + **逐数据集题型 diff** + **两组融合 diff 表** + **per-dataset 门禁明细**（单数据集 spike 不被融合平均抹平）+ **未定义指标标注**（避免把"数据集未标注 evidence"误读为"召回率为 0"）+ **「两侧可比性（逐数据集）」小节**（复用单数据集 `compare` 的维度表：两侧 `dataset_sha256 / judge / judge_model / driver_llm / platform` 不一致记「关键」、两侧都未标注记「关键缺失」、检索命中口径等次要维度记「次要」；两侧 `adapter / adapter_version` 与 `commit` 是被测变量，差异属预期不计入）；比率型指标以百分比呈现（如 `81.5%`），差值为百分点（如 `+2.3pt`）。指标"未计算"统一记 `null`，与真实 0 区分；诊断指标（检索召回率@5 / 过度拒答率）在 diff 与对照表中可见但不参与评级。旧版 run 的 capabilities 未落账时报告会按数据集/逐题记录自动推断并在头部标注来源。可比性问题在**两侧都复用的数据集对上会于开跑前**先打印到控制台——不必等跑完才发现两侧口径对不上。

融合口径（重要，读总分前必读）：

- **宏平均**：六项质量指标（accuracy/token_f1/bleu1/evidence_recall/abstention_accuracy/stale_rate）先在各数据集内算分，再跨集等权平均；**指标只在具备对应标注能力的数据集上参与**——无 gold_stale 的集（locomo/longmemeval）不参与 stale_rate，无 evidence 标注的集（longmemeval/beam）不参与 evidence_recall，无 abstention 题的集不参与 abstention_accuracy。绝不用"记 0"稀释。此外**诊断指标**（检索召回率@5 需 has_evidence；过度拒答率全数据集参与）同样跨集融合展示，但不参与综合评级与门禁。
- **门禁一票否决独立于总分**：办公域槽位 stale_rate > 0.03 → 总评"不达标"，退出码 1（可接 CI），无论综合评级多高。
- **按需下载的公开集必须先 fetch**：`locomo` / `longmemeval_oracle` / `gatemem_office` / `longmemeval_s` / `beam_1m`（或 `beam_10m`）都没有内置文件，`evaluate` 从 `~/.membench/datasets/` 解析可用路径，缺失时报错提示先 `python -m membench.cli dataset fetch <key>`（`--no-lme-s` 不检查 LongMemEval-S；`--no-beam` / `--quick` 不检查 BEAM 槽位）。judge=heuristic 跑 BEAM 会警告（rubric 型题目规则判分失真，建议 llm/client）。
- **产物**：`--report-out DIR`（DIR 是实验目录的**父目录**，缺省 `output_root()`）落 `DIR/eval_<adapter>_<platform>_<ts>/`：各数据集产物在 `runs/<dataset>/`（数据集级报告 `report/{report.md,metrics.json,results.jsonl,fig_*.png}`，逐对话目录 `<对话>/` 内含 `report/`、`runtime/`、`memory_snapshot/`），任务级报告在 `reports/`（`report.md` + `metrics.json` + `evaluate_summary.json` + `fig_dataset_overview.png`）。

run 产物（`runs/<dataset>/report/`）：`results.jsonl`（逐题原始记录）、`metrics.json`、`report.md`、`fig_category_accuracy.png`、`fig_latency_hist.png`；逐对话产物落 `runs/<dataset>/<对话目录>/report/`（纯文字，不出图）。
run 报告含 **「Token 消耗（全流程）」** 小节，取自 `metrics.json` 的 `token_usage` 块（被测系统上报口径的累计值：`ingest_input_tokens` / `query_input_tokens` / `query_output_tokens` / `ingest_llm_calls` / `query_llm_calls` + `n_dialogues` / `n_questions`）；`evaluate` 报告给出**分数据集消耗与合计**（`evaluate_summary.json` 的 `token_usage` 同步落账，含 `datasets_reported` 标明合计覆盖范围）。ingest 按每次 reset 后的真实写入累计——题目级分片下同一对话被多个分片各自 ingest 也如实各记一次；分片融合时逐片相加，任一分片未落账则整体不落账。适配器不填 token 时记 0 并显式提示"未上报"，不读成真实零消耗；这不是指标，不参与评级与门禁。
report.md 中比率型指标以 **"评价（百分比）"** 呈现（如 `优秀（92.3%）`，四级：优秀/良好/中等/较差），开头附 **综合评级**（六项质量指标连续点数均值，1.0~4.0，档内随数值线性变化，阈值 3.5/2.5/1.5）；效率成本指标不参与评级（依赖部署环境）；**诊断指标**（检索召回率@5、过度拒答率）仅展示与归因，不参与评级——前者是前 5 条检索的证据覆盖率，后者是非拒答题的误拒比例，与 `evidence_recall` / `abstention_accuracy` 配合分诊（检索 vs 生成、漏拒 vs 误拒）。评级规则见 `membench/ratings.py` 的 `METRIC_THRESHOLDS` 表，可按团队标准调整。
compare 产物（`compare_<name>_<platform>_<ts>/`，见「产物布局」）：`A/`、`B/` 各为完整评测产物（自带 `runs/` 与 `reports/`），对比结论 `reports/comparison.md` + `reports/fig_metric_diff.png`（质量指标高亮明显变化 + 效率成本）+ `reports/fig_category_diff.png`（分题型准确率对比）。两张图的图例写明 **A（基线）· 短名 / B（候选）· 短名**（短名取 `adapter@commit`，run_id 见报告头部），B 条的绿/红只表示"明显优化/劣化"，另有独立图例项说明——别把高亮色当身份色。

使用 LLM 判分（可选）：

```bash
python -m membench.cli run --adapter mock-strong \
    --dataset membench/data/sample_dataset.jsonl \
    --judge llm --llm-base-url https://api.example.com/v1 \
    --llm-api-key $API_KEY --llm-model gpt-4o-mini \
    --llm-timeout 120   # 可选：判分请求超时秒数（默认 90s，适配推理模型）
```

题目级并发（可选）：`run` / `evaluate` / `evaluate-compare` 均支持 `--jobs N`
（默认 1）。ingest 阶段始终串行（对话时序与记忆状态敏感），query / judge /
证据匹配（含 `--evidence-match llm` 的逐题 chat 调用）在该并发下执行；
`results.jsonl` 的记录顺序与 `--jobs 1` 完全一致。`--jobs > 1` 要求你的
`adapter.query` 与 judge 线程安全（内置 mock adapter 与 LLMJudge 均线程安全）。
使用 `--evidence-match llm` 评测大规模数据集时建议配 `--jobs 8~16`。

## 大数据集：冒烟与分片

冒烟（快速验证链路，可选）：`run --limit N` 每个数据集只取前 N 道题。
run_id 带 `_limit{N}` 后缀、实验目录名带 `_l<N>` 后缀、metrics.json 记入
`"limit": N`，冒烟结果不会被误当全量结果对比。

数据集级分片（BEAM / LongMemEval 这类大集，可选）。分片产物**只写各片负责的对话目录**
（`runs/<dataset>/<对话>/report/`）与一份**分片账本**（`runs/<dataset>/shards.<i>.json`
逐片条目 + 物化视图 `runs/<dataset>/shards.json`）——不写数据集级 `report/` 与任务级
`reports/`，所以分片产物不会"看起来像一份完整实验"；数据集级与任务级产物一律由
`fuse` 补齐：

```bash
# 先下载（落 ~/.membench/datasets/beam_1m.jsonl）：
#   python -m membench.cli dataset fetch beam_1m

# 1) 建一个任务目录（骨架：arms=1 -> runs/ + reports/；arms=2 -> A/ B/ + reports/）。
#    回放任务加 --replay，额外建任务级内容寻址快照池 snapshots/（见「回放任务的快照池」）。
#    stdout 只有任务目录的绝对路径，可直接 TASK=$(...) 接走；--report-out 是它的父目录
TASK=$(python -m membench.cli task init --name beam --platform wx --arms 1)

# 2) 各分片写进**同一棵**任务目录：整段对话归片 d % 4，分片间不重复 ingest；
#    可多台机器/多进程并行跑（每片独占自己的记忆实例，见下「并行分片的实例隔离」）。
#    共享任务目录只允许 --shard-by dialogue：题目级分片会让多片写同一个对话目录并互相覆盖，
#    这种写法在开跑前就被拒绝；同一分片重复跑也直接拒绝（账本里查得到谁跑过）
python -m membench.cli run --adapter my-sys --dataset ~/.membench/datasets/beam_1m.jsonl \
    --task-dir "$TASK" --shards 4 --shard-index 0 --shard-by dialogue
...
python -m membench.cli run --adapter my-sys --dataset ~/.membench/datasets/beam_1m.jsonl \
    --task-dir "$TASK" --shards 4 --shard-index 3 --shard-by dialogue

# 3) 只读进度视图：骨架 / 各数据集的账本进度（几片跑完、哪片还在跑、题数与对话数）/
#    数据集级产物齐备与否 / 快照目录数与体积 / 下一步该跑什么
#    回放任务（task init --replay）改为显示任务级池：桶数、总体积、各臂复用/新写、
#    两臂是否逐对话一致（如「A 臂与 B 臂快照 7/7 逐对话一致（共用 7 个桶）」）
python -m membench.cli task status "$TASK"

# 4) 就地融合：校验分片齐全、口径一致、题目无重复，补齐 runs/<dataset>/report/ 与 reports/；
#    指标重新计算，stale 门禁对融合结果同样生效，FAIL 时退出码 1。
#    （跨目录融合的写法见下；两种写法都不新建目录）
python -m membench.cli fuse --task-dir "$TASK"
```

跨目录融合（各片本就落在各自实验目录，或要指定输出位置时）：

```bash
# --runs 每项 = 分片的实验目录，或其中的 runs/<dataset> 数据集目录；
# --out = 融合产物的完整实验目录（产物落其 runs/<dataset>/ 与 reports/）
python -m membench.cli fuse \
    --runs output/eval_my-sys_<平台>_<ts0>_s0/runs/beam_1m,output/eval_my-sys_<平台>_<ts1>_s1/runs/beam_1m \
    --out output/beam_1m_full
```

`--shards` 与 `--limit` 可组合（先分片再截断），用于分片链路的廉价验证。

**分片账本（`runs/<dataset>/shards.<i>.json` + `runs/<dataset>/shards.json`）**：同一次
评测各分片的进度与归属。**每个分片一个条目文件** `shards.<i>.json`（单文件原子写，并行
分片只写自己那一片，不会互相覆盖，也不需要锁），`shards.json` 由这些条目物化成视图
（人直接翻文件时看它；`fuse` / `task status` 走 `layout.read_shard_ledger` 的合并结果，
只有旧产物没有逐片条目时才读 `shards.json` 本身）。头部记口径（`schema` / `dataset` /
`shards` / `shard_by` / `limit` / 数据集指纹），`entries.s<i>` 记该片的 `status`
（`running` / `done`）、负责的对话、实测题数、`run_id` 与起止时间。账本口径或数据集
指纹与本次不一致（换了 `--shards` / `--shard-by` / `--limit` / 数据集却写进同一目录）
会被拒绝：同一个任务目录里只能放同一次评测的分片。

**分片 / 冒烟怎么进目录名**：分片用跨目录写法时（不带 `--task-dir`），实验目录名末尾追加
`_s<分片序号>`、`--limit N` 追加 `_l<N>`（如 `eval_my-sys_wx_20261024-193000_s1_l50`）；
不分片且无 limit 时目录名保持 `eval_<name>_<platform>_<ts>` 原样。`run_id`（`metrics.json`
里的身份字符串）用的是另一套更完整的词表 `_shardIofN` / `_dshardIofN` / `_limitN`——含分片
总数与分片维度，两者都只由 `membench/layout.py` 生成。**已知取舍**：`--shards max` 落在单
分片数据集上时实际只跑 1 片，但实际分片数要解析数据集才知道，CLI 建目录时已按"传了
`--shards`"记下 `_s0`——目录名宁可多带标记，也不能让分片产物与不分片产物撞进同一个目录名
（`run_id` 里则不出分片标记，因为那里用的是结算后的有效分片数）。

**分片 / 冒烟怎么进融合产物**：`fuse` 校验各分片的 `limit` 一致（全量分片与 `--limit` 冒烟分片混融会得到"部分全量"的假完整结果，直接拒绝），并把 `limit` 透传到融合产物的 `metrics["limit"]`；融合产物的 `run_id` 带 `_fused`、`_shard<N>`/`_dshard<N>` 与 `_limitN` 标记——冒烟融合产物与全量融合产物因此不会被当成同一身份，`compare --require-comparable` 会按题目数（关键维度）判它们不可比。

**分片产物里的报告怎么读**：分片只落对话级报告，它们**顶部有一行醒目标注**
「分片 i/N（按 dialogue|question）——仅覆盖部分题目」，题目数写成
`14（分片 i/N，总量见融合报告）`——那是**本片**的题数，不是整场结论；
fuse 的融合产物（`shard_index=None`）不带这个标注，它覆盖全部题目。

**并行分片的实例隔离**：分片是进程级并行——每个分片必须独占一个记忆实例（HTTP 桥 → 每分片独立桥进程 / `BRIDGE_PORT` / `BRIDGE_DATA_DIR`；Python 适配器 → 每分片独立进程）。runner 对每个 dialogue 调 `reset()`，共享同一实例并行会互相清库（跨片串扰）；只有一个实例时请串行跑分片，`fuse` 结果不受影响。

**分片数上限与 `--shards max`**：`--shard-by dialogue` 上限为数据集对话数、`--shard-by question` 上限为题目数。传入超过上限的整数**按上限执行**（不报错，输出中会说明实际分片数）；`--shards max` 直接按上限取最大分片；分片索引越过有效分片数的进程**安全跳过**（不产生空 run——空分片纯浪费进程/内存与桥实例）。

**分片 A/B 官方通路**：先 `task init --name <name>` 建一个**双臂任务目录**（骨架 `A/` `B/` `reports/`）→ 两侧分别 `run --task-dir <任务目录>/A`（或 `B`）`--shards N --shard-by dialogue` 并行写进同一棵树 → 两侧各自 `fuse --task-dir <任务目录>/A|B` 就地补齐 → `compare --task-dir <任务目录>` 就地出对比报告（读 `A/` 与 `B/`，`comparison.md` 落任务目录的 `reports/`，不再新建 `compare_*`、也不复制两臂）。也支持"各片各落实验目录 + 跨目录 `fuse`"的老写法，或用 `evaluate-compare --baseline-run <key>=<A 融合实验目录>` 汇总多数据集。融合产物与普通 run 同等可校验：可比性元数据（adapter_version / judge_model / driver_llm / 数据集指纹 / 检索口径）全部透传，可直接用 `compare --require-comparable` 复验。

**分片维度怎么选（`--shard-by`，重要）**：

- `--shard-by question`（默认）按全局题号 `i % N == shard_index` 分片；不含入选题的对话整段跳过（ingest 也省掉）。适合"少量超长对话承载大量题"、能整块跳过对话的场景。
- `--shard-by dialogue` 整段选取 `d % N == shard_index` 的对话、保留其全部题目，**分片之间 ingest 互不重复**。"几乎每个对话都有题"的 ingest 密集型数据集（内置 office：20 对话 / 86 题；BEAM-1M：35 对话 × 20 题）必须用它——题目级分片会让每个分片仍对全部对话 ingest，成本 ×N，分片反而更慢。
- 多个分片写进**同一个任务目录**（`--task-dir`）时**只允许 `--shard-by dialogue`**：题目级分片下同一对话会被多片写进同一个对话目录、互相覆盖，所以在开跑前直接拒绝；题目级分片请各片各落一个实验目录（`--report-out`，目录名带 `_s<i>`）再跨目录 `fuse`。
- 两种维度的分片都能用 `fuse` 融合回完整产物，但不许混维融合（fuse 会明确报错）。

## 可复现性元数据

每个数据集目录的 `report/metrics.json` 除了六维指标，还记一组元数据，回答"这次分数是用什么跑出来的"：

- `schema_version`：元数据 schema 版本（当前 `"0.2"`）；
- `run_id`：本次评测的身份字符串（形如 `<adapter>_<platform>[_<commit8>]_<dataset>[_shardIofN|_dshardIofN][_limitN]_<%Y%m%d_%H%M%S>`，只作身份标识，不再是目录名）；
- `adapter` / `platform` / `commit` / `dataset` / `dataset_path` / `dataset_source`；
- `driver_llm`：驱动被测系统的 LLM 标识（`--driver-llm`，自由文本）；
- `judge` / `judge_model`：判分器名与判分模型（`judge_model` 缺省时自动取 judge 对象的 `.model`）；
- `adapter_version`：适配器实现版本 / commit（`--adapter-version`）；
- `dataset_fingerprint`：`{"sha256": 内容 sha256 前 16 位, "bytes": 字节数, "name": 文件名}`，`compare` 的可比性校验据此机检"数据集是否变了"；
- `adapter_meta`：`--adapter-meta "model=deepseek-v3;temp=0"` 自由文本，原样记入；
- `adapter_describe`：adapter 定义了可选 `describe() -> dict` 时自动调用记入（原样进 metrics，非空时报告渲染「适配器自描述」小节；抛错 / 返回非 dict / 不可 JSON 序列化时警告一次并记 null，不中断评测）；
- `experiment_dir` / `dataset_dir`：本次实验目录与数据集产物目录（读取方据此定位，不必猜层级）；
- `run_mode`：`full` / `replay-restore` / `replay-rebuild`（见「状态快照与回放」）；
- `state_snapshot`：快照**默认导出**（适配器声明了快照能力时；`--no-snapshot` 关闭）或快照回放时记入；普通 run 的 `root` 指向**数据集目录**（即 `--replay` 可传的目录）；回放任务的 `root` 是快照源（数据集目录或回放任务路径），另给 `snapshots_dir`（本 run 的任务级池）与 `n_buckets` / `n_reused` / `n_new`（对话级给 `bucket` / `reused`）。适配器不支持快照时该块记 `role="unsupported"`（如实说明"没导"，读完可区分"这次没导"与"导不了"）。

```bash
python -m membench.cli run --adapter my-system --dataset membench/data/office_dataset.jsonl \
    --driver-llm deepseek-v3 --adapter-version $(git rev-parse --short HEAD) \
    --judge llm --llm-model gpt-4o-mini --report-out output/   # DIR = 实验目录的父目录
```

建议至少把驱动模型、判分模型、适配器版本、数据集指纹落全——`compare --require-comparable` 靠这些维度做前置校验，字段缺失会记入"无法核验"。

## A/B 可比性前置校验

`compare` 只**测量差异**，不**判定**两组是否只差一个变量。为了让"驱动/判分模型、适配器、数据集变了却照给 PASS"不再发生，`compare` 在算 diff 之前先做**可比性前置校验**：逐维度比对两个 run 的 metrics.json 元数据，结论作为门禁列表的**首位**（`comparability`）。

维度分三级：

| 级别 | 维度 | 不一致的后果 |
|---|---|---|
| 关键 | `adapter`、`adapter_version`、`dataset_sha256`（取 `dataset_fingerprint.sha256`）、`judge`、`judge_model`、`driver_llm`、`platform`、`run_mode`（全量 / 快照回放）、`n_questions`（分数算在哪个题集上：`--limit` 冒烟 vs 全量即走这里） | 非 strict 下门禁降级 WARN；`--require-comparable`（strict）下直接 FAIL |
| 次要 | `limit`、`shards`、`shard_index`、`adapter_meta` | 降级 WARN，不 FAIL |
| 提示 | `commit`、`dataset_path`、`dataset` | 仅记录（`same=false`），不影响结论 |

- 关键维度**任一值缺失**（如旧 run 没有 `driver_llm`）记入 `unverifiable`：非 strict 下不判不可比，strict 下与不一致同等处理（判 FAIL）——没填就核验不了，strict 模式不允许"没填当通过"。`n_questions` 取自 `overall.n_questions`（`aggregate` 的落账位置，顶层没有该字段）。
- `commit` 不同是 A/B 的被测变量本身，**不算**不可比；`run_mode` 缺省按 `full` 归一（本特性之前的所有 run 都是全量跑），故历史 run 不会因缺该字段被误判。
- **`--limit` 冒烟产物不得与全量产物对比**：`fuse` 把各分片一致的 `limit` 透传到融合产物，`run_id` 另带 `_fused` / `_shard<N>` / `_limitN` 标记；两侧题目数不同（关键维度）时 strict 直接判 FAIL——冒烟分部署到全量门槛上会被拦下。

门禁语义：

- **PASS**：关键维度全部一致（且无次要不一致）；
- **WARN**：存在关键维度不一致（非 strict）或次要维度不一致——结果可疑但可继续；
- **FAIL**：strict 下存在关键维度不一致或无法核验。

不可比原因在两处可见：CLI 输出逐条 `[membench] 不可比原因: [关键] <维度>: <说明>`；`compare_<name>_<platform>_<ts>/reports/comparison.md` 的「可比性前置校验」小节给出逐维度对照表（A / B / 是否一致）与「不可比原因」列表。

**多数据集对比（`evaluate-compare`）同样逐数据集核对**：报告头部有一行汇总，正文附「两侧可比性（逐数据集）」小节（只列不一致与缺失项，级别同上表——「关键缺失」表示两侧都未标注、核验不了）。两点与单数据集 `compare` 不同：两侧 `adapter / adapter_version` 跨侧不同是本实验的设计变量，不进清单；两侧都已复用的数据集对在**跑前**就把问题清单打到控制台，跑完再用最终产物（含本轮补跑的数据集）复核一遍。

```bash
python -m membench.cli compare \
    --baseline output/eval_<A>_<平台>_<ts>/runs/<dataset> \
    --candidate output/eval_<B>_<平台>_<ts>/runs/<dataset> \
    --require-comparable --tag my-opt
```

`--tag` 决定实验目录名的 `<name>` 段（缺省 `<基线 adapter>_vs_<候选 adapter>`；`platform` 段缺省取两臂平台，两臂不同则 `p1-p2`）。`--baseline` / `--candidate` 传**实验目录**或 **`runs/<dataset>` 数据集目录**都行。`--task-dir DIR` 是另一种用法：就地拿该任务目录的 `A/`（基线）与 `B/`（候选）出对比报告，不新建目录、不复制两臂——A/B 两臂本来就写在同一个任务目录里时（多分片 A/B 通路的常态）用这个最省事。退出码：verdict=FAIL → 1，否则 0。

## 状态快照与回放（改读取侧免重跑 ingest）

改了**检索/排序/回答**这类读取侧代码后，不必重新把数据集写进记忆系统——用上一次
评测导出的「逐对话状态快照」直接回放即可（省掉 ingest 的算力与 LLM 抽取成本，也让
A/B 摆脱"写入漂移"噪声）。

**快照默认启用**：`run` / `evaluate` / `evaluate-compare` 默认就在每个对话
**ingest 完成后、第一道 query 之前**把它唯一的一份状态导出到
`<实验目录>/runs/<dataset>/<对话目录>/memory_snapshot/`（即对话级报告目录
`report/` 的兄弟目录），框架另写一份 `manifest.json` 记录身份与条目（`sha256` /
字节数 / session / 消息 / 题数）。跑完 ingest 就能直接翻这些记忆文件——对话级
`report.md` 的「快照与回放」小节给出 `../memory_snapshot/` 的**可点击链接**与
条目数、体积，不必自己去拼路径。

```bash
# 1) 全量跑一次：快照默认导出（适配器需实现 snapshot()；HTTP 桥同款端点见接入章节）
#    落各对话的 runs/<dataset>/<对话目录>/memory_snapshot/（含框架写的 manifest.json）
python -m membench.cli run --adapter <你的适配器> --dataset <数据集> --judge heuristic
#    不需要快照时显式关掉（省磁盘/时间；关掉后改读取侧只能重跑 ingest）
python -m membench.cli run --adapter <你的适配器> --dataset <数据集> --no-snapshot

# 2) 改了读取侧代码后回放（跳过 reset + ingest，只跑 query）+ 保真校验地板
#    --replay 传导过快照的实验目录（或其中的 runs/<数据集目录>）；
#    --replay-verify-source 传源产物（实验目录或其中 runs/<数据集目录>）：
#    按 <数据集目录>/report/results.jsonl 逐题比对
python -m membench.cli run --adapter <你的适配器> --dataset <同一个数据集> \
    --replay output/eval_<名>_<平台>_<ts> \
    --replay-verify-source output/eval_<名>_<平台>_<ts>/runs/<数据集目录> --judge heuristic
```

- **快照时机**：每个对话 ingest 完成后、第一道 query 之前（runner 每对话 `reset()`，
  评测本身是"一个对话一个独立记忆生命周期"，所以按对话切片天然等价）；
  **清单每对话一份**，落 `<对话目录>/memory_snapshot/manifest.json`——自包含
  （身份 + 数据集指纹 + 能力声明 + 该对话条目 sha256 / 字节数 / 输入指纹与
  `complete` 标记），单个对话目录即可搬到别处；整场齐备性由跑前扫描各对话清单统一校验；
- **适配器不支持快照时**：只打印一次警告并**跳过**，评测照常跑完（不中断）；
  `metrics.json` 的 `state_snapshot.role` 记 `unsupported`，报告里渲染成
  「不支持（适配器未声明快照能力）」并说明怎么补能力——不会渲染成 `n/a` / `vNone`
  这类看起来像"产物质损"的空值，也不给打不开的快照链接。想彻底安静可用 `--no-snapshot`；
- **谁产出快照**：被测系统自己（适配器可选接口 `snapshot_capabilities()` /
  `snapshot()` / `restore()` / `rebuild()`）——框架只给目录与 tag、算哈希、做校验，
  不解析快照内容（HTTP 桥对应 `POST /snapshot-capabilities|/snapshot|/restore|/rebuild`，
  另有可选自描述端点 `POST /describe`）；
- **体积**：快照随对话数与记忆规模线性增长（逐对话各一份），单份超过 100 MB 会打印
  一次告警（提示可用 `--no-snapshot` 关掉）；`task status` 会列出快照目录数与合计体积；
- **三种模式**（`metrics.json` 的 `run_mode`）：`full` / `replay-restore`（格式兼容，
  直接恢复）/ `replay-rebuild`（格式不兼容或需重建派生结构，走系统的 `rebuild`）；
  `--replay-mode auto|restore|rebuild` 控制协商，**不兼容且无 rebuild 会在开跑前拒绝**；
- **跑前校验**：清单缺失/损坏、数据集指纹不符、本 run 需要的对话没快照，一律拒绝，
  不产生半成品产物；
- **保真校验**：`--replay-verify-source <源产物：实验目录或数据集目录>` 逐题比对答案 / 检索 / 判分，
  写进报告与 `metrics.json` 的 `replay_verify`——这就是**回放误差地板**，读取侧 A/B
  的差值必须大于它才有意义；未改代码时答案应 100% 一致；
- **口径（重要）**：回放固定写入侧，结论只覆盖**读取/检索/回答层**，不得当"完整评测"
  汇报；`run_mode` 是 compare 的**关键维度**，回放分与全量分会被判不可比；融合
  （`fuse`）也拒绝混融两种模式的分片；回放的 token 账里 ingest 记 0 并标
  `ingest_skipped`（"没跑"≠"零成本"）；
- **该重跑的场景**：改动落在**写入侧**（抽取提示词、更新/去重决策、写入成本）或
  **逻辑 schema**（字段/粒度变了且系统没有 `rebuild`）时，快照回放测不到——必须
  重新 ingest（至少挑几个对话做子集）。

### 回放任务的快照池

回放本身也导出快照（在 `restore()` / `rebuild()` 完成后、第一道 query 之前），但落点
是**任务级内容寻址池**而不是对话目录：`task init --replay` 在任务根建 `snapshots/`，
每个对话的状态先导到临时目录、按内容 sha256（沿用清单口径，排除 `manifest.json`）
前 12 位定桶：

```
<任务>/
├── runs/ 或 A/ B/                       # 与普通任务相同（单臂 runs/；双臂 A/ B/）
└── snapshots/
    ├── <sha256 前 12 位>/               # 桶（含适配器导出的内容 + 框架写的 manifest.json）
    ├── .tmp-<uuid>/                     # 导出中的临时目录（成功即整体移入桶；不是桶）
    ├── entries/<臂>/<数据集>/<对话>.json  # 逐对话条目（真源；空臂键目录名为 _）
    └── index.json                       # 派生视图：桶 / 逐臂逐对话映射 / 各臂来源与计数
```

- **条目一对话一文件**：条目自足（`digest` / `state_sha256` / `bytes` / `n_files` /
  `dataset` / `dialogue_id` / `arm` / `source` / `source_manifest_sha256` / `reused` /
  `run_id` / `written_at`），单文件临时写 + `os.replace` 原子落位；对话级分片的各进程
  只写自己负责的对话 ⇒ **并行回放没有共享可变状态要抢，不需要锁**。同一对话重复写
  （同片重跑）按"覆盖自己的条目"处理，但 digest 必须一致，否则报错（状态变了）；
- **内容去重**：A/B 两臂回放同一源时，两臂恢复出的状态相同 → 只产生一份桶，条目里
  `A/ds/d1` 与 `B/ds/d1` 指向同一 digest；状态不同（回放不保真）才各存一份——
  **差异本身就是回放不保真的证据**；
- **同源不重复（并发安全）**：桶落位只靠一次 `os.replace` 的成败判"本次创建 / 别人已
  建好"——成功即新写；失败（目标已存在）则读桶内清单的 `entry.sha256`，与本次一致就
  删临时目录、记 `reused=true`；不一致（前缀撞车 / 桶被损坏）直接报错。不预判目录
  存在性，因此两个分片进程同时建同一个桶也只会有一个真创建；
- **索引是派生视图**：`index.json` 的真源是 `entries/` 树 + 各桶 `manifest.json`，
  由 `layout.materialize_snapshot_index` 重算（`fuse --task-dir` 收尾、`task status`
  发现缺失/过期时都会刷新）。并行回放的各进程**不写**它，所以不会丢条目；`--replay`
  读源也优先读条目树，只有旧产物（没有 `entries/`）才回退读索引；
- **并行分片回放**：同一臂可以多进程并行，必须 `--shard-by dialogue`：

  ```bash
  TASK=$(python -m membench.cli task init --name replay --arms 2 --replay)
  for i in 0 1; do
    python -m membench.cli run --adapter <adapter> --dataset <数据集> \
        --task-dir "$TASK/A" --replay <源> \
        --shards 2 --shard-index "$i" --shard-by dialogue &
  done
  wait
  python -m membench.cli fuse --task-dir "$TASK/A"   # 收尾：物化快照池索引
  python -m membench.cli task status "$TASK"
  ```

- **`--no-snapshot`**：回放也不写桶、不写条目 / 索引（`metrics.state_snapshot` 里没有
  `snapshots_dir` / `n_buckets`）；
- **`task status`**：回放任务会打印池的桶数与总体积、各臂 `n_dialogues` / `n_reused` /
  `n_new` 与来源（从条目 + 桶重算，不信任可能过期的 `index.json`），并给出两臂一致性
  的一句话（如 `数据集 ds: A 臂与 B 臂快照 7/7 逐对话一致（共用 7 个桶）`；有差异则
  列出 `内容不同 / 仅 A / 仅 B` 的对话数）；
- **链式回放**：`--replay` 接受回放任务本身——传臂目录（`<任务>/A`）按该臂取桶；传
  任务根时池里只有一个臂才自动用它，多个臂会报错并要求改传具体臂目录（两臂状态
  可能不同，替你选一个臂就是替你选结论）。所以
  `run --task-dir <task2>/A --replay <task1>/A` 可直接跑；
- **报告**：回放任务的对话级报告把「本对话快照桶」链接到
  `snapshots/<digest>/` 并标注「复用 / 新写」，数据集级报告给出任务级池目录与本 run
  的桶数（复用 / 新写）。

逐对话条目的结构（`snapshots/entries/A/ds/d1.json`，`sort_keys` 落盘）：

```json
{
  "schema": 1,
  "digest": "5f54ecf2363e",
  "state_sha256": "5f54…（完整 64 位）",
  "bytes": 71, "n_files": 1,
  "dataset": "ds", "dialogue_id": "d1", "arm": "A",
  "source": "<--replay 的源目录>",
  "source_manifest_sha256": "<源清单合并 sha256>",
  "reused": false,
  "run_id": "<本次回放的 run_id>",
  "written_at": "2026-10-27T15:12:13"
}
```

物化后的 `snapshots/index.json`（结构对旧读者完全兼容）：

```json
{
  "schema": 1,
  "buckets": {
    "5f54ecf2363e": {"dataset": "ds", "dialogue_id": "d1", "bytes": 71,
                     "n_files": 1, "state_sha256": "5f54…（完整 64 位）"}
  },
  "entries": {"A|ds|d1": "5f54ecf2363e", "B|ds|d1": "5f54ecf2363e"},
  "arms": {
    "A": {"source": "<--replay 的源目录>", "source_manifest_sha256": "<源清单合并 sha256>",
           "n_dialogues": 1, "n_reused": 0, "n_new": 1},
    "B": {"source": "<同一个源>", "source_manifest_sha256": "<同上>",
           "n_dialogues": 1, "n_reused": 1, "n_new": 0}
  }
}
```

`arms` 的 `n_dialogues` 是该臂的条目数（跨数据集累计），`n_reused` / `n_new` 是该臂
当前条目的 `reused` 统计；`source` 取该臂**最近一次**写入的条目。`buckets` 里的
`dataset` / `dialogue_id` 是**首次写入该桶**的对话（桶按内容寻址，之后可能被其它对话
或另一个臂复用）；`n_files` 不含框架写的 `manifest.json`。索引只在 `fuse` / `task status`
收尾时物化，不参与回放写入路径。

## 判分健壮性说明

- LLM judge 输出按三级解析：整体 JSON → 抠取首个 JSON 对象 → 回退启发式判分并打 `llm-fallback` 标记。
- 使用 LLM 判分时，metrics.json 顶层记录 `judge_fallback: {count, ratio}`；报告头部展示回退比例，**ratio > 5% 时报告输出警示行**（判分质量可能退化，建议抽查复核）。
- stale 红线判定为确定性规则：旧值命中且新值未命中才判 stale（回答中引用旧值作为变更上下文、同时给出新值的，不判陈旧）。

## 用户测试集管理

`--dataset` 现在是可选参数。不传时按优先级自动选择测试集：

1. 显式 `--dataset <key|路径>`（文件 → `user-specified`；`office` → 用户集 / 内置 office；
   其它注册表 key → 内置文件或按需下载文件，来源 `fetch`）；
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

- `run_id` 形如 `<adapter>_<platform>[_<commit8>]_<dataset>[_shardIofN|_dshardIofN][_limitN]_<%Y%m%d_%H%M%S>`（缺省 platform 为 `default`）；它只作身份字符串落 `metrics.json`，**不再是目录名**；
- `metrics.json` 元数据记 `platform` / `commit` / `dataset_source` 字段；
- `report.md` 开头固定有一张「评测配置」表：测试集路径、测试集来源、平台、commit、判分器。

跨版本（commit）对比时，把两次 run 的 `--driver-llm` / `--adapter-version` 也填上，`compare` 的可比性前置校验才能区分"结果变化来自被测系统"还是"来自换了驱动模型"。

## 矩阵对比（多 run / platform × commit）

`matrix` 子命令对多个带标签的 run 做矩阵对比：

```bash
# --runs 每项形如 label=<产物目录>：右值可以是实验目录或 runs/<dataset> 数据集目录
python -m membench.cli matrix \
  --runs "wx@a1b2c3=output/eval_wx_a1b2c3_<ts>/runs/<dataset>" \
         "fs@a1b2c3=output/eval_fs_a1b2c3_<ts>/runs/<dataset>" \
         "wx@d4e5f6=output/eval_wx_d4e5f6_<ts>" \
         "fs@d4e5f6=output/eval_fs_d4e5f6_<ts>" \
  [--out output/matrix_from_scratch] [--report-out output/]
```

- `--out DIR` 直接指定**完整实验目录**（优先；产物落其 `reports/` 与 `runs/<label>/`）；`--report-out DIR` 是实验目录的**父目录**（给出时落 `DIR/matrix_<ts>/`）；都不给时落 `output_root()/matrix_<ts>/`。
- 退出码：任一两两组合的门禁判 FAIL → 1，否则 0（与 `compare` 同口径，可直接接 CI）。

label 推荐 `platform@commit` 格式（也接受任意字符串）。实验目录自包含（各组产物的数据集树副本落 `runs/<label>/`，内部同 eval 的 `runs/<dataset>/…` 形态），产出 `reports/report.md` +
`reports/matrix_result.json` + `reports/fig_matrix_accuracy.png`：

1. 若标签恰好覆盖 2 平台 × 2 commit，对 accuracy / stale_rate / evidence_recall /
   abstention_accuracy / latency_p95_ms 各生成一张 platform × commit 透视表；
2. 所有 label 的逐指标汇总表（行=指标，列=label）；
3. 所有两两组合的门禁判定（复用 compare 的 PASS/WARN/FAIL 规则）；
4. accuracy 分组条形图（x=commit、hue=platform，或反之取合理者）。

## 产物布局（一次实验一个自包含子目录）

一次评测 = 一个**任务目录**：单臂任务就是 `eval_<name>_<platform>_<ts>`（自身即完整评测产物根），双臂 A/B 任务是 `compare_<name>_<platform>_<ts>`（`A/` 与 `B/` 是两个臂，**各为一个完整的单臂产物根**）。报告与其对应的逐题数据就近归属在同一个任务目录里，不再与全局 run 混放。任务目录的**父目录**由 `membench.paths.output_root()` 统一判定：`MEMBENCH_OUTPUT_DIR` > skill 包的 `output/` > 工程根 `output/`；目录名与层级的唯一来源是 `membench/layout.py`。

```bash
# 任务目录可以"预定"（可选）：run / evaluate 各自跑会各建一个 eval_*；要让**同一次
# 评测的多片**或 **A/B 两臂**写进同一棵树，先建任务目录再传 --task-dir：
#   --arms 1 → runs/ + reports/（单臂）；--arms 2 → A/ B/ + reports/（A/B 双臂）
#   stdout 只有任务目录的绝对路径（说明走 stderr），可直接 TASK=$(...) 接走
python -m membench.cli task init --name beam --platform wx --arms 1
python -m membench.cli task status <任务目录>   # 只读进度：骨架 / 分片账本 / 快照 / 下一步该跑什么

# 单次评测（run / evaluate --report-out DIR）：DIR 是任务目录的父目录（缺省 output_root()）；
# --task-dir DIR：写进**已存在**的任务目录（不新建）；fuse --out DIR 直接给完整任务目录
DIR/eval_<adapter>_<platform>_<ts>[_s<i>][_l<limit>]/   # ts = YYYYMMDD-HHMMSS；分片/冒烟后缀见上
├── runtime/                             # 实验级运行时数据：commands.jsonl + logs/ + env.json（见下）
├── runs/<dataset>/                      # 每个数据集一棵子树
│   ├── report/{report.md,metrics.json,results.jsonl,fig_category_accuracy.png,fig_latency_hist.png}
│   ├── shards.<i>.json, shards.json     # 分片账本：逐片条目 + 物化视图（分片评测才有，见「冒烟与分片」）
│   └── <对话目录>/                       # 对话 id 归一成安全目录名（非法字符换 `_`，空值回退 dialogue）
│       ├── runtime/                     # **对话级**运行时数据（桥的 reset 前备份落 runtime/reset_backup/）
│       ├── memory_snapshot/             # 默认导出的对话快照（含框架写的 manifest.json；--no-snapshot 关）
│       └── report/{report.md,metrics.json,results.jsonl}   # 对话级，纯文字不出图
└── reports/                             # 任务级报告：report.md + metrics.json + 报告引用的图
                                         # run → 数据集级那两图；evaluate → fig_dataset_overview.png
                                         # 且另有 reports/evaluate_summary.json

# A/B 对比任务目录（compare / evaluate-compare --report-out DIR：同为任务目录的父目录）
DIR/compare_<name>_<platform>_<ts>/
├── A/  B/                               # 各为一个完整 eval 实验形态（含自身 runs/ 与 reports/）
└── reports/comparison.md                # 结论；另有两张对比图：
                                         # compare → fig_metric_diff.png / fig_category_diff.png
                                         # evaluate-compare → fig_per_dataset_diff.png / fig_fused_diff.png
# 可选：optimization_report.md 由你的优化流程写入 reports/

# 回放任务（task init --replay）：任务根多一个内容寻址快照池
DIR/{eval_<name>_<platform>_<ts> | compare_<name>_<platform>_<ts>}/
├── runs/ 或 A/ B/                       # 与普通任务相同
└── snapshots/
    ├── <sha256 前 12 位>/                # 桶：适配器导出的状态 + 框架写的 manifest.json（记来源）
    ├── entries/<臂>/<数据集>/<对话>.json  # 逐对话条目（真源；并行分片各写各的）
    └── index.json                       # 派生视图：桶 / 臂|数据集|对话 -> 桶 / 各臂来源与计数

# 矩阵对比（matrix：--report-out DIR 为父目录，--out 直接指定完整实验目录，--out 优先）
DIR/matrix_<ts>/
├── reports/{report.md,matrix_result.json,fig_matrix_accuracy.png}
└── runs/<label>/                        # 各 label 的数据集树副本（内部同 eval 的 runs/<dataset>/…）
```

**实验级 `runtime/`（与对话级 runtime 区分）**：每次会写产物的命令在它自己的顶层输出容器（`eval_*`、`compare_*`、`matrix_*`、`fuse` 的目标目录、`task init` 的新任务目录）下建 `runtime/`；`run --task-dir <任务>/A` 这类命令的容器就是臂目录，runtime 落 `A/`（`evaluate-compare` 一条命令写两臂时，容器是 `compare_*` 根，runtime 记在那里）：

- `runtime/logs/<YYYYmmdd-HHMMSS>-<command>-<pid>.log`：该次命令 stdout+stderr 全量（终端照常显示）；
- `runtime/commands.jsonl`：一行一次调用（`argv` / `cwd` / 起止时间 / 耗时 / `exit_code` / `python` / `platform` / `pid` / `log` / `env`；密钥参数已脱敏；`O_APPEND` 追加，多分片并发不丢条目）；
- `runtime/env.json`：一次性环境快照（python / 平台 / membench 版本与 commit / `MEMBENCH_*`）。

它与 `<实验目录>/runs/<dataset>/<对话目录>/runtime/` 不是一回事：后者是**该对话**的运行时数据（适配器 reset 备份等），前者记录的是**评测命令本身**。只读命令（`task status` / `doctor` / `dataset info` / `list-datasets` / `adapter list`）不写；以库方式直接调用 `Runner` / `evaluate` 也不写——只有 CLI 会话才落。`task status` 会展示最近命令、非零退出条数与最近日志路径；`verify_artifacts.py` 对缺 `runtime/commands.jsonl` 的任务给警告级提示。

长跑：`skill/src/scripts/run_task.py ... --background` 派生到后台（stdout 只打印任务目录一行），用 `skill/src/scripts/watch_task.py <任务目录> --until-done` 定时查看进度；用法以 skill 的 `SKILL.md` 为准。

**`--task-dir`（与 `--report-out` 互斥，二选一）**：把产物写进**已存在的任务目录**（`task init` 建的目录，或它的某个臂目录 `A/` / `B/`），而不是新建 `eval_*` / `compare_*`——同一次评测的多个分片因此能写进同一棵树：

- `run --task-dir DIR` / `evaluate --task-dir DIR`：产物直接落 DIR；共享目录下**只允许 `--shard-by dialogue`**，同一分片重复跑直接拒绝（账本查得到谁跑过）；
- `evaluate-compare --task-dir DIR`：两臂分别写 `DIR/A`、`DIR/B`，对比报告落 `DIR/reports/`；
- `compare --task-dir DIR`：就地读 `DIR/A`（基线）与 `DIR/B`（候选），`comparison.md` 落 `DIR/reports/`，不新建 `compare_*`、也不复制两臂；
- `fuse --task-dir DIR`：就地补齐数据集级与任务级产物（各片本来就写在这一棵树里），与 `--runs/--out` 互斥。

- 不传 `--report-out` / `--task-dir` / `--out` 时一律落 `output_root()`：`run` / `evaluate` 落 `output_root()/eval_<adapter>_<platform>_<ts>/`，`compare` / `evaluate-compare` 落 `output_root()/compare_<name>_<platform>_<ts>/`，`matrix` 落 `output_root()/matrix_<ts>/`。
- **分片账本** `runs/<dataset>/shards.<i>.json`（逐片条目）+ `runs/<dataset>/shards.json`（物化视图）：多片共用一个任务目录（或跨目录各落一棵）时那棵树"跑到哪了"的记录；口径（`shards` / `shard_by` / `limit` / 数据集指纹）与本次不一致会被拒绝，`fuse` 与 `task status` 都经 `layout.read_shard_ledger` 读合并结果（旧产物没有逐片条目时才读 `shards.json`）。分片只写各片负责的对话目录 + 账本，数据集级与任务级产物由 `fuse` 补齐（见「大数据集：冒烟与分片」）。
- **对话级报告**除本对话指标外，还给该对话记忆快照的可点击相对链接（`../memory_snapshot/`）与清单条目数 / 体积——跑完 ingest 就能翻记忆文件；分片产物的对话级报告顶部另有「分片 i/N——仅覆盖部分题目」标注。
- 命名与落位 helper 是**公开 API**（不必自己重写约定）：`membench.layout` 的 `task_subdir` / `init_task_dir` / `arm_dir` / `eval_subdir` / `compare_subdir` / `matrix_subdir` / `experiment_tags` / `run_id` / `run_id_tags` / `experiment_path` / `runs_root` / `dataset_dir` / `dataset_report_dir` / `dialogue_dir` / `dialogue_report_dir` / `dialogue_runtime_dir` / `dialogue_snapshot_dir` / `shard_ledger_path` / `read_shard_ledger` / `task_report_dir` / `resolve_dataset_dir`，以及回放任务快照池的 `snapshot_pool_root` / `snapshot_arm_key` / `snapshot_entry_path` / `snapshot_bucket_dir` / `snapshot_index_path` / `read_snapshot_entry` / `write_snapshot_entry` / `build_snapshot_index` / `materialize_snapshot_index` / `read_snapshot_index` / `write_snapshot_index`；产物根 `membench.paths.output_root()`。运行时数据的读取走 `membench.runtime` 的 `runtime_dir` / `commands_path` / `read_commands` / `summarize`，不要在调用方重解析 `commands.jsonl`。
- **读取方的「产物目录」入参**（`compare --baseline/--candidate`、`evaluate --from-runs`、`evaluate-compare --baseline-run/--candidate-run`、`matrix --runs` 右值、`run --replay`、`run --replay-verify-source`、`fuse --runs`）统一经 `layout.resolve_dataset_dir` 解析：传**实验目录**（其下恰有一个数据集目录时自动定位；有多个则要求指向具体数据集目录）或**数据集目录**（`<实验目录>/runs/<dataset>`）都行；指标固定读 `<数据集目录>/report/metrics.json`。`run --replay` 另接受**回放任务**（带 `snapshots/` 的任务根或臂目录），按逐对话条目取桶而不是走数据集目录（见「回放任务的快照池」）。未融合的分片产物（数据集目录下只有账本与对话目录）会明确要求先 `fuse`——"没融合"与"没产物"是两回事。`--from-runs` 的 `KEY` 还会与实际产物身份核对：`office` ↔ `office_dataset`（或用户测试集的 `user_dataset`）这类"key 与目录名不同"的映射由数据集注册表反推，不会拿 key 与目录名直接比字符串；key 认错时直接报错，不把别的数据集的分数记到该 key 名下。
- Runner API 使用者传 `Runner(..., out_dir=实验目录)`：Runner 直接写 `<实验目录>/runs/<dataset>/…`（数据集级报告 + 逐对话报告 + `runtime/`），任务级报告落 `<实验目录>/reports/`，并把 `experiment_dir` / `dataset_dir` 写回 `metrics.json`。
- 旧版产物（平铺的 run 目录、缺 `report/` 层级）**不再兼容**：读取时统一报错「产物结构为旧版（缺少 report/ 层级），本版本不再兼容，请重新跑评测」。目录里**根本没有**数据集产物（空目录、跑到一半中断）另有独立报错文案——那是"没有产物"而不是"版本旧"，别去改代码兼容，换个目录或重跑即可。

## 如何接入自己的记忆系统

> 详细接入指导（必需三挂点与可选接口的写法、**真实性红线与降级口径流程**、验收判据、接入 checklist）见 skill 包内 `references/adapter_guide.md`。

四条路径，按推荐程度排序：

### 路径一：继承 BaseMemoryAdapter（推荐，最灵活）

必需接口是 `name` 属性 + `reset / ingest / query` 三个方法：

```python
from membench import layout
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

# out_dir = 实验目录（Runner 直接写它的 runs/<dataset>/…，任务级报告落它的 reports/）
exp_dir = layout.experiment_path(None, layout.eval_subdir("my-system"))
Runner(MyAdapter(), "membench/data/sample_dataset.jsonl", out_dir=exp_dir).run()
```

可选能力分两组，都不是接入必需：

- **快照组**（需要「快照回放」、改读取侧免重跑 ingest 时实现，见「状态快照与回放」）：
  `snapshot_capabilities()`（声明 `supported/restore/rebuild/format_version/compatible_from`）、
  `snapshot(target_dir, tag)`、`restore(source_dir, tag)`、`rebuild(source_dir, tag)`——
  **声明了就必须实现**；`restore` 只在快照格式兼容时用，`rebuild` 拿快照里的源数据重建
  派生结构（跨格式版本也可用）。框架只传目录与 tag，导出什么由你决定。
- **自描述**：`describe()` 返回 dict（如嵌入模型、检索方式、索引类型、版本），原样写进
  `metrics.adapter_describe`，并在报告「适配器自描述」小节展示——**不参与评分，也不是
  可比性维度**。默认返回 `{}`（报告里整节不出现）；抛异常只警告一次，评测照常完成。

完整可运行示例见 `examples/quickstart.py`。token / llm_calls 填不了就填 0（报告中视为 n/a /「未上报」）。`reset()` 里若要实现"清空前快照"这类回溯能力，可读框架给的可选提示环境变量 `MEMBENCH_RUN_DIR`（**本次对话目录**，即 `<实验目录>/runs/<dataset>/<对话目录>`，备份建议落它的 `runtime/reset_backup/`）/ `MEMBENCH_DIALOGUE_ID` / `MEMBENCH_DIALOGUE_INDEX`（框架不读也不写你的存储）。

`query` 返回的 `retrieved` 决定 `evidence_recall` / `recall@5` 是否成立，口径 = **注入即口径**：填实际拼进回答 prompt 的**记忆条目原文**，按注入优先级排序，多轮检索先合并去重再返回，报注入的那一份全量（框架不合并、不去重、不截断；空列表 = 静默按 0 召回计）。

### 路径二：实现 REST 契约，用 HttpAdapter

如果你的记忆系统已有 HTTP 服务，实现三个必需端点即可，无需写 Python：

- `POST /reset`：清空全部记忆状态。请求体带**可选**上下文 `{"run_dir", "dialogue_id", "dialogue_index"}`（`run_dir` = 本次**对话目录**，备份建议落它的 `runtime/reset_backup/`）——供服务端在清空前自行快照/归档；不认识就忽略，不影响契约。
- `POST /ingest`：body `{"messages": [{"role", "content", "timestamp"}]}`
- `POST /query`：body `{"question": "..."}`，返回 `{"answer": str, "retrieved": [str], ...}`

可选端点五个（缺失即视为不支持，不报错、不中断评测；`HttpAdapter` 会转发目录与 tag，
内容由桥自己决定）：

- `POST /snapshot-capabilities`：快照能力声明（`supported/restore/rebuild/format_version/compatible_from`）；
- `POST /snapshot` / `POST /restore` / `POST /rebuild`：导出状态 / 同版本恢复 / 跨版本重建；
- `POST /describe`：返回 dict 自描述（老桥没有该端点时 `HttpAdapter.describe()` 返回 `{}`）。

参考模板 `examples/http_bridge_node` 五个都已实现（文件级快照，带标记文件防误恢复；
`/restore` 只接受当前格式版本，`/rebuild` 接受 `compatible_from` 列出的旧版本并重建派生结构）。

然后在 Python 侧用内置 `HttpAdapter` 指向你的服务地址，照常跑 `Runner`。服务地址经通用构造参数 `--adapter-arg` 传入，命令行一步到位，无需再包一层适配器：

```bash
python -m membench.cli run --adapter http \
    --adapter-arg base_url=http://127.0.0.1:8787 \
    --adapter-arg timeout=10 \
    --dataset membench/data/office_dataset.jsonl --judge heuristic --report-out output/
# HttpAdapter 接受的构造参数：base_url（必填）、api_key、timeout；
# 必填参数缺失 / key 写错都会在开跑前报错并列出可接受的参数名
```

Python API 侧同理：`HttpAdapter(base_url="http://127.0.0.1:8787", timeout=10.0)`。

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
from membench import layout
from membench.agent_loop import AgentLoopAdapter
from membench.runner import Runner

adapter = AgentLoopAdapter(
    llm_client="my_agent.llm:LLMClient",   # module:ClassName（延迟导入、无参实例化），与 --judge client 同约定
    ingest_tools=[{"name": "write_memory", "schema": {...}, "handler": write_memory}],
    query_tools=[{"name": "retrieve_memory", "schema": {...}, "handler": retrieve}],  # 缺省复用 ingest_tools
    system_prompt="你是一个带长期记忆的办公助手。",
)
Runner(adapter, "membench/data/office_dataset.jsonl", judge="heuristic",
       driver_llm="deepseek-v3",
       out_dir=layout.experiment_path(None, layout.eval_subdir("agent-loop"))).run()
```

- 确切签名：`run_tool_loop(client, messages, tools, max_rounds=8, system_prompt=None) -> dict`；`AgentLoopAdapter(llm_client, ingest_tools, query_tools=None, system_prompt=None, name="agent-loop", max_rounds=8, retrieved_from=None, ingest_prompt=None, query_prompt=None)`。
- `llm_client` 三选一：带 `chat_with_tools(messages, tools)` 的对象（优先，走原生 tool-call）> 带 `chat` / `complete` 的对象或可调用对象（走纯文本 JSON 约定）> `"module:ClassName"` 类路径字符串。`tools` 元素为 `{"name", "schema", "handler"}`；handler 收解析后的 `arguments` dict，返回 str 或 dict（handler 抛异常会被折叠成观察文本回填，**不中断**循环）。
- membench 负责循环、`IngestStats` / `QueryResult` 统计与 `retrieved` 回填。回填由 `retrieved_from(question, loop_result) -> list[str]` 决定，缺省 `default_retrieved_from` 识别工具名含 `retrieve` / `search` / `recall` 的调用，把**多轮检索的结果按调用顺序合并、去重**后返回（不丢后续轮次）——这决定 `evidence_recall` 等证据指标是否有意义：**检索工具名要对得上关键词、结果要是被检索到的记忆原文**（JSON 列表或一行一条），否则请自传 `retrieved_from`。
- `AgentLoopAdapter.describe()` 返回驱动 LLM 标识与工具名列表，自动落入 run 元数据 `adapter_describe`。

### 路径四：导出对话日志，用 TranscriptAdapter

最低成本路径：把你系统线上真实的「记忆写入/检索」日志导出为离线 transcript（对话 + 每题实际检索到的记忆条目），用 `TranscriptAdapter` 回放注入。适合**不改系统、先量现状**的场景，也适合复现线上 badcase。

### 适配器构造参数（--adapter-arg，通用机制）

需要构造参数的适配器（`http` 的 `base_url`、`transcript` 的 `transcript_path`、`mock-weak` 的 `top_k` 等）一律用通用参数传入，不给单个适配器开专用开关：

```bash
python -m membench.cli run --adapter http \
    --adapter-arg base_url=http://127.0.0.1:8787 --adapter-arg timeout=10 ...
python -m membench.cli evaluate --adapter transcript \
    --adapter-arg transcript_path=/path/to/transcript.jsonl ...
# A/B 两侧各自一组：evaluate-compare 用 --adapter-arg-a / --adapter-arg-b
python -m membench.cli evaluate-compare --baseline-adapter http --candidate-adapter http \
    --adapter-arg-a base_url=http://127.0.0.1:8787 --adapter-arg-b base_url=http://127.0.0.1:8788 ...
```

- `KEY` 必须是该适配器构造函数的形参名（注册表项是工厂函数时按工厂签名判）：不认识 → 报错并**列出该适配器可接受的参数名**；缺 `=` 或值为空 → 报错说明 `KEY=VALUE` 形式；必填参数没给 → 报错提示用 `--adapter-arg` 传（如 `--adapter http` 缺 `base_url`）。
- 值按形参注解做最小转换：`int` / `float` 注解转数字，其余保持字符串（值里的 `=` 原样保留，可放 URL query）；同一个 `KEY` 给多次时后者生效。

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
| 检索质量 | evidence_recall | 非 abstention 题中，证据命中（任一检索条目支持该证据）的比例 | ≥ 0.90 |
| 检索质量 | recall_at_5（诊断） | 非拒答题中，前 5 条检索的证据条目覆盖率（严格口径；仅 evidence 标注数据集参与） | 诊断项，不评级 |
| 效率成本 | latency_p50 / p95 | query 延迟分位数（毫秒） | 按系统预算自定 |
| 效率成本 | avg_input_tokens / avg_llm_calls | 每题平均注入 token 与 LLM 调用数 | 越低越好 |
| 记忆动态 | stale_rate | knowledge_update 题中命中过期事实且未包含更新值的比例 | ≤ 0.03 |

> 注：stale 的"包含更新值"接受两种口径——gold 片段逐字覆盖，或 gold 内容 token 召回 ≥ 0.8（容忍同义改写，如"下周一下午"→"周一下午"）。旧值仅作为变更上下文被引用、且新值已正确给出时不计为 stale。
| 拒答 | abstention_accuracy | abstention 题中正确拒答的比例 | ≥ 0.80 |
| 可扩展性 | （随数据规模观察上述指标的退化曲线） | 用更大数据集多跑几轮对比 | — |

**检索命中口径**（`--evidence-match`，默认 `substring`）：`substring` 为逐字子串（要求适配器返回接近原文的检索片段）；`overlap` 用内容 token 包含度（`--evidence-match-threshold`，默认 0.6）容错标点/包裹/轻微改写；`llm` 复用 `--llm-*` 配置逐题做语义判定，覆盖摘要/改写式记忆存储（失败自动回退 substring 并记入 `retrieval_match_fallbacks`）。口径落账到 `metrics.json` 的 `retrieval_match` 字段；**不同口径的跑分不可直接对比**（`compare` 可比性校验会给出 WARN），换口径后 A/B 两侧都要重跑。报告中若出现"两项检索指标全为 0"的警示，通常是 evidence 原文不在检索结果里（如摘要式存储），请换口径或让适配器回填原文片段后再比对。

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

> `compare` 只测量差异，不判定两组是否只差一个变量——必须自己固定模型 / harness / 数据集，只切优化开关，并用 `--require-comparable` 让 compare 帮你拦下"变量不一致"的对比。

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

**时间戳统一（ISO 8601，精度不变）**：转换器把各原始数据集的时间戳归一成 ISO 8601——LoCoMo `1:56 pm on 8 May, 2023` → `2023-05-08T13:56`、LongMemEval `2023/04/10 (Mon) 17:50` → `2023-04-10T17:50`、BEAM `March-01-2024` → `2024-03-01`、GateMem 与内置集本来就是 ISO（空格分隔会规范成 `T`）。**粒度保持原样**：只到日的不补 `00:00`、只到分的不补秒、原始无时区就不造时区。粒度差异依然存在（LoCoMo / LongMemEval 会话级、GateMem 逐轮、BEAM 每 batch 一条只到日的锚点≈0.6% 消息）；不认识的形态在 `fetch` 阶段报错停下，不静默留混格式。归一实现在 `membench/data/timestamps.py`，解析表按实测形态定（含 LoCoMo 12 小时制边界与 LongMemEval 星期一致性校验）。

```bash
python -m membench.cli dataset fetch locomo            # 逐个下载；幂等（产物已存在且可解析即直接返回）
python -m membench.cli dataset fetch longmemeval_oracle
python -m membench.cli dataset fetch beam_1m
# 全部可下载 key：locomo / longmemeval_oracle / longmemeval_s / gatemem_office / beam_1m / beam_10m
```

`dataset fetch` 幂等：产物已存在且能被 `load_dataset` 解析时直接返回；未知 key → 退出码 2（并列出可下载的 key）；下载失败 → 打印消息、退出码 1。CLI 未暴露 `--force`，要强制重下请删掉 `~/.membench/datasets/<key>.jsonl` 与同名 raw 缓存目录后重跑（或调 `membench.data.fetch.fetch_dataset(key, force=True)`）。`evaluate` 全量评估会从下载目录解析这些集，缺失时报错提示先 `dataset fetch <key>`。

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

# 评测（LoCoMo 为英文长对话，建议先用 --limit 5 冒烟）
python -m membench.cli run --adapter mock-strong --dataset ~/.membench/datasets/locomo.jsonl --limit 5
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
A: 同一数据集（指纹一致）、同一判分器与判分模型、同一驱动 LLM 与适配器实现，只变被测系统；否则 diff 无法归因。`compare` 会做可比性前置校验，关键维度不一致时给出 WARN/FAIL；要强制拦截用 `--require-comparable`（不可比原因见 CLI 输出与 `compare_<name>_<platform>_<ts>/reports/comparison.md` 的「可比性前置校验」小节）。

**Q: heuristic 判分把我的正确答案判错了怎么办？**
A: 启发式是 token 包含匹配，表述差异大时会误判。先跑 `--judge llm` 复核；长期方案是优化黄金集答案表述或改用 LLM 判分（注意对基线重跑保持同尺比较）。

**Q: 我的系统拿不到 token / llm_calls 统计？**
A: 填 0 即可，报告中对应项视为 n/a /「未上报」，不影响 accuracy 等核心指标。注意这与"真实消耗为 0"是两回事：`token_usage` 块全为 0 时报告会显式标注"适配器未上报该字段"。适配器（或 HTTP 桥的 `/ingest`、`/query` 响应）能填多少填多少——填了才有"这次评测烧了多少 token"的账。

**Q: 每个对话都 `reset()`，怎么回溯"清空前记忆里存了什么"？**
A: 框架不读不写你的记忆存储（黑盒契约），但会把 reset 上下文（本次对话目录 / 对话 id 与序号）作为可选提示给出：Python 适配器读环境变量 `MEMBENCH_RUN_DIR`（= `<实验目录>/runs/<dataset>/<对话目录>`）/ `MEMBENCH_DIALOGUE_ID` / `MEMBENCH_DIALOGUE_INDEX`；HTTP 桥收 `POST /reset` 请求体里的 `run_dir` / `dialogue_id` / `dialogue_index`。要不要在清空前快照、落到哪里，由接入侧决定——约定是落该对话目录的 `runtime/reset_backup/`（参考模板的实现：`BRIDGE_RESET_BACKUP=run` 时即如此，见 `examples/http_bridge_node/README.md`）。逐题 `retrieved` 已在数据集级 `report/results.jsonl` 里，快照只用于查内部存储状态；reset 每对话一次，长跑请谨慎常开。

**Q: 如何把门禁接入 CI？**
A: `compare` 子命令在 verdict=FAIL 时以退出码 1 退出，直接放进流水线即可。

**Q: 支持并发评测吗？**
A: 当前 Runner 按题目级并发执行（记忆系统有状态，ingest 顺序最安全）。并发用 `--jobs N`（query / judge / 证据匹配并发，ingest 仍串行）。数据量大时建议按 dialogue 拆分数据集并行跑多个进程，再合并 results.jsonl。

## 办公 Agent 专用黄金集（office_dataset.jsonl）

针对通用办公助手类 Agent 内置的领域数据集：20 个对话、86 题，覆盖二十类办公场景——项目截止日变更、会议决议追溯、写作风格偏好、日程与联系人变动、敏感信息拒答（红线题）、跨周待办跟踪、自动化流程偏好、长跨度谈判时间线、供应商合同、招聘面试、事故复盘、客户投诉升级、年度预算、流程偏好回滚、差旅报销规则、同名项目干扰、凭证保密、远程办公制度、发布计划、OKR 协作。

```bash
python -m membench.cli run --adapter my-system --dataset membench/data/office_dataset.jsonl
```

办公场景的指标优先级建议：**stale_rate（陈旧召回）是一票否决项**——过期的截止日和联系人会直接造成办公事故；其次是 abstention_accuracy（涉密信息拒答）与 preference 题准确率（决定产出物是否"像你写的"）。建议将此数据集作为最小验收集，再按你的实际业务对话扩充到 50–100 题。
