# membench — Agent 记忆系统评测工程

membench 是一个面向 **agent 记忆系统** 的评测框架（Python 3.10+，纯标准库优先，仅依赖 matplotlib 出图；HTTP 适配器另需 requests）。它回答两个问题：

1. **run**：我的记忆系统在一套多会话对话数据集上表现如何？（六维指标 + 中文报告 + 图表）
2. **compare**：这次优化相比基线是进步还是回归？（逐指标 diff + PASS/FAIL 门禁）

设计要点：

- **六维指标**：正确性、检索质量、效率成本、记忆动态（知识更新/陈旧召回）、拒答、可扩展性。
- **判分双轨**：启发式判分（默认，零成本、确定性）+ LLM-as-Judge（可选，OpenAI 兼容接口，失败自动回退启发式）。
- **适配器插件化**：被测系统实现 3 个方法即可接入；内置 mock / transcript / http 参考适配器。

## 目录结构

```
├── membench/
│   ├── adapters/          # BaseMemoryAdapter + mock / transcript / http 适配器
│   ├── data/              # 数据集 schema、JSONL 加载器、sample_dataset.jsonl
│   ├── judges/            # HeuristicJudge / LLMJudge
│   ├── metrics/           # token_f1 / bleu1 / recall@k + 六维汇总
│   ├── runner.py          # Runner: 执行评测, 写 run 目录
│   ├── fuse.py            # fuse: 分片 run 合并为完整 run（指标重算 + 门禁）
│   ├── compare.py         # compare_runs: 两个 run 的 diff + 回归门禁
│   ├── matrix.py          # compare_matrix: 多 run 矩阵对比（platform × commit）
│   ├── userdata.py        # 用户测试集管理: install / status / delete / resolve
│   ├── report.py          # report.md + 图表（matplotlib Agg）+ copy_report 导出
│   └── cli.py             # CLI: run / fuse / compare / matrix / user-data / list-datasets / dataset info|fetch
├── tests/                 # pytest（含 mock-weak vs mock-strong 端到端）
└── examples/
    ├── quickstart.py      # 最小接入示例
    └── optimize_loop.md   # 优化迭代闭环教程
```

## 快速开始

```bash
# 1. 用内置强 mock 适配器跑内置示例数据集
python -m membench.cli run --adapter mock-strong \
    --dataset membench/data/sample_dataset.jsonl \
    --judge heuristic --out runs/

# 2. 再跑弱 mock（模拟“优化前”）
python -m membench.cli run --adapter mock-weak \
    --dataset membench/data/sample_dataset.jsonl --out runs/

# 3. 对比两个 run，输出门禁结论
python -m membench.cli compare --baseline runs/<run_weak> --candidate runs/<run_strong>

# 查看内置数据集
python -m membench.cli list-datasets
```

## 全量评估（多数据集融合总分）

`evaluate` 对所有数据集全量测试，跨数据集**宏平均**融合出综合评级：

```bash
# 默认全量 = 办公域槽位（用户集>内置office）+ locomo + longmemeval_oracle + gatemem_office + BEAM-1M
python -m membench.cli evaluate --adapter <你的adapter> --judge llm --report-out <报告目录>

python -m membench.cli evaluate --adapter <a> --quick      # 跳过 BEAM，分钟级日常回归
python -m membench.cli evaluate --adapter <a> --beam-10m   # 用 BEAM-10M 替换 1M，全集终验
python -m membench.cli evaluate --adapter <a> --include-lme-s  # 追加 LongMemEval-S（大体积，与 oracle 分差反映检索/精确率损失）
```

融合口径（重要，读总分前必读）：

- **宏平均**：六项质量指标（accuracy/token_f1/bleu1/evidence_recall/abstention_accuracy/stale_rate）先在各数据集内算分，再跨集等权平均；**指标只在具备对应标注能力的数据集上参与**——无 gold_stale 的集（locomo/longmemeval）不参与 stale_rate，无 evidence 标注的集（longmemeval/beam）不参与 evidence_recall，无 abstention 题的集不参与 abstention_accuracy。绝不用"记 0"稀释。
- **门禁一票否决独立于总分**：办公域槽位 stale_rate > 0.03 → 总评"不达标"，退出码 1（可接 CI），无论综合评级多高。
- BEAM 槽位文件缺失时报错提示先 `dataset fetch`（--quick 不检查）；judge=heuristic 跑 BEAM 会警告（rubric 型题目规则判分失真，建议 llm/client）。
- 产物：`runs/evaluate_<时间戳>/evaluate_report.md`（门禁结论 + 综合评级 + 融合指标表含参与集 + 分数据集明细 + 各 run 链接）。

run 产物（`runs/<run_id>/`）：`results.jsonl`（逐题原始记录）、`metrics.json`、`report.md`、`fig_category_accuracy.png`、`fig_latency_hist.png`。
report.md 中质量指标以 **"评价（评分）"** 呈现（如 `优秀（0.923）`，四级：优秀/良好/中等/较差），开头附 **综合评级**（六项质量指标评级点数均值，阈值 3.5/2.5/1.5）；效率成本指标不参与评级（依赖部署环境）。评级规则见 `membench/ratings.py` 的 `METRIC_THRESHOLDS` 表，可按团队标准调整。
compare 产物（`runs/compare_<a>_vs_<b>/`）：`compare_report.md`、`fig_metric_diff.png`。

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
# 分 4 片跑（确定性分片：题目 i 归片 i % 4；可在多台机器上并行）
python -m membench.cli run --adapter my-sys --dataset data/beam_1m.jsonl --shards 4 --shard-index 0
...
python -m membench.cli run --adapter my-sys --dataset data/beam_1m.jsonl --shards 4 --shard-index 3

# 合并为完整 run（校验分片齐全、adapter/dataset 一致、题目无重复；
# 指标重新计算，stale 门禁对融合结果同样生效，FAIL 时退出码 1）
python -m membench.cli fuse --runs runs/..._shard0of4,runs/..._shard1of4,... --out runs/beam_1m_full
```

`--shards` 与 `--limit` 可组合（先分片再截断），用于分片链路的廉价验证。

## 可复现性元数据

metrics.json 的 metadata 除 adapter 名外还可记两类信息，避免换模型后分数漂移无账可查：

- `--adapter-meta "model=deepseek-v3;temp=0"`：自由文本，原样记入 `metadata.adapter_meta`；
- adapter 定义可选方法 `describe() -> dict` 时，Runner 自动调用并记入 `metadata.adapter_describe`（调用失败记 null，不中断评测）。

建议 ingest/query 的实际驱动模型、温度、检索参数至少以其中一种方式落账。

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

## 矩阵对比（多 run / platform × commit）

`matrix` 子命令对多个带标签的 run 做矩阵对比：

```bash
python -m membench.cli matrix \
  --runs "wx@a1b2c3=runs/dir1" "fs@a1b2c3=runs/dir2" \
         "wx@d4e5f6=runs/dir3" "fs@d4e5f6=runs/dir4" \
  [--out runs/matrix_xxx] [--report-out DIR]
```

label 推荐 `platform@commit` 格式（也接受任意字符串）。产出 `matrix_report.md` +
`matrix_result.json` + `fig_matrix_accuracy.png`：

1. 若标签恰好覆盖 2 平台 × 2 commit，对 accuracy / stale_rate / evidence_recall /
   abstention_accuracy / latency_p95_ms 各生成一张 platform × commit 透视表；
2. 所有 label 的逐指标汇总表（行=指标，列=label）；
3. 所有两两组合的门禁判定（复用 compare 的 PASS/WARN/FAIL 规则）；
4. accuracy 分组条形图（x=commit、hue=platform，或反之取合理者）。

## 报告输出到指定目录（--report-out）

`run` / `compare` / `matrix` 均支持 `--report-out DIR`：在正常产物之外，把最终
markdown 报告复制为带时间戳的文件名，并连带复制报告引用的 png（文件名加同一
时间戳前缀，md 内引用同步修正）：

```
DIR/eval_report_{adapter}_{platform}_{yymmdd_hhmmss}.md        # run
DIR/compare_report_{labelA}_vs_{labelB}_{yymmdd_hhmmss}.md     # compare
DIR/matrix_report_{yymmdd_hhmmss}.md                           # matrix
```

适合集中归档每日 / 每次 CI 的评测报告。

## 如何接入自己的记忆系统

三条路径，按推荐程度排序：

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

### 路径三：导出对话日志，用 TranscriptAdapter

最低成本路径：把你系统线上真实的「记忆写入/检索」日志导出为离线 transcript（对话 + 每题实际检索到的记忆条目），用 `TranscriptAdapter` 回放注入。适合**不改系统、先量现状**的场景，也适合复现线上 badcase。

### 适配器持久化（adapter install）

业务适配器（如 `qwenwork-ohos`）写好一次后即可落盘，之后任何 session 无需重新生成或设环境变量即可使用：

```bash
# 落盘到持久化目录（默认 ~/.membench/adapters/，可用 MEMBENCH_ADAPTERS_DIR 覆盖）
python -m membench.cli adapter install ./qwenwork_ohos.py [--name qwenwork-ohos]

# 查看内置 + 持久化适配器（表格含注册名/类名/安装时间/来源路径）
python -m membench.cli adapter list

# 删除
python -m membench.cli adapter remove qwenwork_ohos
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
| 拒答 | abstention_accuracy | abstention 题中正确拒答的比例 | ≥ 0.80 |
| 可扩展性 | （随数据规模观察上述指标的退化曲线） | 用更大数据集多跑几轮对比 | — |

## 优化迭代工作流

```
run 基线 → 看 report.md 失败样例找短板 → 单变量改动 →
run 候选 → compare 看门禁（PASS/FAIL/WARN）→ 固化或回滚
```

门禁规则（可用 `--acc-threshold` 等参数覆盖）：

- accuracy 下降 > 0.02 → **FAIL**
- stale_rate 上升 > 0.02 → **FAIL**
- latency_p95 上升 > 30% → **WARN**
- 其余指标变动仅报告

完整教程（含如何扩充黄金集、何时切换 LLM 判分）见 `examples/optimize_loop.md`。

## 接入公开基准（LoCoMo / LongMemEval / BEAM / GateMem）

membench 内置四个公开记忆基准的格式转换器。查看全部支持的数据集（来源/特点/基准线）：

```bash
python -m membench.cli dataset info            # 列举全部 8 个数据集及简介
python -m membench.cli dataset info locomo     # 单个数据集详细介绍（含基准线：如何解读分数）
```

内置 JSONL（开箱即用）：`data/office_dataset.jsonl`、`data/sample_dataset.jsonl`、`data/locomo.jsonl`、`data/longmemeval_oracle.jsonl`、`data/longmemeval_s.jsonl`、`data/gatemem_office.jsonl`。
BEAM 体积过大不内置，按需下载转换：

```bash
python -m membench.cli dataset fetch beam_1m    # 35 chats × ~1M token，约 700 题
python -m membench.cli dataset fetch beam_10m   # 10 chats × ~10M token，约 200 题（全集终验）
```

手动转换（自选数据源时）：

```bash
# LoCoMo（Snap Research，10 个超长对话，1986 题，5 类题型含对抗/拒答题）
curl -fsSL -o /tmp/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
python -m membench.data.convert_locomo /tmp/locomo10.json -o data/locomo.jsonl

# LongMemEval（ICLR 2025，500 题，5 类能力；oracle 版 ~5MB 适合冒烟，
# s 版 ~250MB 是公开榜单标准，m 版 ~2GB）
curl -fsSL -o /tmp/longmemeval_oracle.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
python -m membench.data.convert_longmemeval /tmp/longmemeval_oracle.json -o data/lme_oracle.jsonl

# BEAM（arXiv:2510.27246）与 GateMem（办公三维）：见 dataset fetch 与转换器 docstring
python -m membench.data.convert_beam <chats根目录> --scale 1m -o data/beam_1m.jsonl
python -m membench.data.convert_gatemem episodes.jsonl checkpoints.jsonl -o data/gatemem_office.jsonl

# 评测（LoCoMo 为英文长对话，建议先用 --max-dialogues 1 冒烟）
python -m membench.cli run --adapter mock-strong --dataset data/locomo.jsonl --out runs/
```

题型映射与已知限制：

| 基准 | 题型映射 | 已知限制 |
|---|---|---|
| LoCoMo | cat1→multi_hop, cat2→temporal, cat3→open_ended, cat4→single_hop, cat5→abstention（gold 为 `<ABSTAIN>` 哨兵） | evidence 自动从 dia_id 解析为原文；约 6% 标注质量争议是数据集本身已知问题 |
| LongMemEval | user/assistant→single_hop, multi-session→multi_hop, temporal-reasoning→temporal, knowledge-update→knowledge_update, preference→preference | oracle 版无 per-turn 证据（recall 恒 0）；**S 版 evidence 已从 answer_session_ids 解析为 session 级原文**；无 gold_stale（stale_rate 恒 0）；cleaned 版无拒答题 |
| BEAM | 10 类题型→membench 7 类（information_extraction→single_hop、multi_session_reasoning→multi_hop、event_ordering/temporal_reasoning→temporal、knowledge_update/contradiction_resolution→knowledge_update、instruction_following/preference_following→preference、summarization→open_ended、abstention→abstention） | evidence 为空；rubric 未利用；gold_stale 取 potential_confusion（描述性文本） |
| GateMem | (utility,answer)→single_hop、(privacy,refuse)→abstention、(privacy,answer_redacted)→single_hop、(safety,no_memory)→abstention | asker 身份、leak_targets、as_of_turn_id 截断语义未建模 |

**使用建议**：公开基准做横向对比与回归仪表盘；但厂商公布的 LoCoMo/LongMemEval 分数可比性差（判分器、检索预算、作答模型不同即可造成 ±20 分以上摆动），最终验收仍以你自建的领域黄金集为准。两者结合：公开基准按周跑，黄金集按提交跑。

## FAQ

**Q: compare 的两次 run 有什么前提？**
A: 同一数据集、同一判分器、同一题型分布，只变被测系统。否则 diff 无法归因。

**Q: heuristic 判分把我的正确答案判错了怎么办？**
A: 启发式是 token 包含匹配，表述差异大时会误判。先跑 `--judge llm` 复核；长期方案是优化黄金集答案表述或改用 LLM 判分（注意对基线重跑保持同尺比较）。

**Q: 我的系统拿不到 token / llm_calls 统计？**
A: 填 0 即可，报告中对应项视为 n/a，不影响 accuracy 等核心指标。

**Q: 如何把门禁接入 CI？**
A: `compare` 子命令在 verdict=FAIL 时以退出码 1 退出，直接放进流水线即可。

**Q: 支持并发评测吗？**
A: 当前 Runner 为顺序执行（记忆系统有状态，顺序最安全）。数据量大时建议按 dialogue 拆分数据集并行跑多个进程，再合并 results.jsonl。

## 办公 Agent 专用黄金集（office_dataset.jsonl）

针对 WorkBuddy 类通用办公助手内置的领域数据集：20 个对话、86 题，覆盖二十类办公场景——项目截止日变更、会议决议追溯、写作风格偏好、日程与联系人变动、敏感信息拒答（红线题）、跨周待办跟踪、自动化流程偏好、长跨度谈判时间线、供应商合同、招聘面试、事故复盘、客户投诉升级、年度预算、流程偏好回滚、差旅报销规则、同名项目干扰、凭证保密、远程办公制度、发布计划、OKR 协作。

```bash
python -m membench.cli run --adapter my-system --dataset membench/data/office_dataset.jsonl --out runs/
```

办公场景的指标优先级建议：**stale_rate（陈旧召回）是一票否决项**——过期的截止日和联系人会直接造成办公事故；其次是 abstention_accuracy（涉密信息拒答）与 preference 题准确率（决定产出物是否"像你写的"）。建议将此数据集作为最小验收集，再按你的实际业务对话扩充到 50–100 题。
