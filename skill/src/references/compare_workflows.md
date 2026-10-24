# Commit 对比与平台对比工作流详解

## 前置假设

- 被测 Agent 是 git 仓库，两个 commit 都能独立构建/运行；
- adapter 已实现，且能以参数切换平台配置（多平台场景）。

## 可比性前置校验（A/B 前必做）

**`compare` 只测量差异，不判定两组是否只差一个变量。** "驱动/判分模型、适配器实现、数据集"任一变了，PASS/FAIL 都不可归因——这是本波修复的核心问题。因此对比前必须显式校验关键维度。

| 关键维度 | 记录字段（`metrics.json`） | 不一致的后果 |
|---|---|---|
| 驱动（被测 agent 的底座）模型 | `driver_llm` | 分数差异混入"换了模型"的差异 |
| 判分模型 | `judge_model` | 判分尺度不同，diff 是噪声 |
| 适配器实现 / 版本 | `adapter_version` | 接入层改动被误当成记忆能力改动 |
| 数据集 | `dataset_fingerprint`（`sha256` / `bytes` / `name`） | 题目不同，逐指标 diff 无意义 |
| 平台 | run 的 `platform`（`--platform`） | A/B 分组错配 |
| 题目数 | `overall.n_questions` | 分数算在**不同题集**上（典型：`--limit` 冒烟 vs 全量、分片融合只覆盖一部分），diff 不可归因 |

- `run` 用 `--driver-llm` / `--adapter-version` 显式落账这些字段。
- `compare --require-comparable` 做**前置校验**（strict）：任一关键维度**不一致**或**无法核验**（元数据缺失）→ 可比性门禁 **FAIL**，整体 verdict=FAIL、退出码 1（即拒绝当 A/B 用）；报告含「可比性前置校验」小节与「不可比原因」清单。
- 不加 `--require-comparable` 时：关键维度不一致 → 可比性门禁 WARN（verdict 至少 WARN，不会静默 PASS）；关键维度无法核验 → 只在 CLI 输出与报告里提示，不影响 verdict。
- `compare --tag <标签>` 给本次对比命名，决定产物目录 `compare_<name>_<platform>_<ts>/` 的 `<name>` 段（缺省 `<基线 adapter>_vs_<候选 adapter>`；`<platform>` 取两臂一致的 platform，不一致则 `p1-p2`、缺省 `default`）。

**不可比时的表现**：报告开头给出"不可比原因"清单（如 `dataset_fingerprint.sha256 不同`、`driver_llm 不同：gpt-4o vs claude-...`），并明确拒绝当 A/B 使用；此时应回到"同尺比较"——固定模型、判分器与数据集后重跑。

**消融纪律**：优化对比必须由优化者自己保证**单变量**——固定模型与 harness、只切优化开关。`compare` 不做这个判断，别指望它替你保证。

## 工作流 3：两个 commit 的 A/B 对比

```bash
AGENT_REPO=/path/to/your/agent

# 1) 两个 commit 各建一个 worktree（互不干扰）
git -C $AGENT_REPO worktree add /tmp/agent_old <SHA_A>
git -C $AGENT_REPO worktree add /tmp/agent_new <SHA_B>

# 2) 各自构建/安装依赖（按项目实际命令，如 pip install -e . / pnpm build）
(cd /tmp/agent_old && <build命令>)
(cd /tmp/agent_new && <build命令>)

# 3) 同一测试集、同一判分器、同一驱动模型，分别跑（落账元数据）
cd <skill目录> && export MEMBENCH_USER_DATA_DIR=$PWD/user_dataset
PYTHONPATH=/tmp/agent_old:<membench路径> python -m membench.cli run \
  --adapter <adapter模块路径> --commit <SHA_A> --judge llm \
  --driver-llm <驱动模型> --adapter-version <版本> --report-out output/
PYTHONPATH=/tmp/agent_new:<membench路径> python -m membench.cli run \
  --adapter <adapter模块路径> --commit <SHA_B> --judge llm \
  --driver-llm <驱动模型> --adapter-version <版本> --report-out output/

# 4) 对比（baseline=旧 commit；产物落 <report-out>/compare_<name>_<platform>_<ts>/）
python -m membench.cli compare \
  --baseline <旧 commit 的实验目录或 runs/<dataset> 数据集目录> \
  --candidate <新 commit 的实验目录或 runs/<dataset> 数据集目录> \
  --require-comparable --tag <标签> --report-out output/

# 5) 清理
git -C $AGENT_REPO worktree remove /tmp/agent_old
git -C $AGENT_REPO worktree remove /tmp/agent_new
```

门禁规则（默认，CLI 可调）：accuracy 降 >0.02 → FAIL；stale_rate 升 >0.02 → FAIL；p95 延迟升 >30% → WARN。

**产物目录（一次实验一个自包含子目录）**：

```
<report-out>/eval_<adapter>_<platform>_<ts>/
├── runs/<dataset>/
│   ├── report/{report.md, metrics.json, results.jsonl, fig_*.png}   # 数据集级聚合
│   └── <对话目录>/{runtime/, memory_snapshot/, report/}              # 对话级；report/ 不出图
└── reports/                                                          # 任务级（单数据集 = 数据集级内容）
<report-out>/compare_<name>_<platform>_<ts>/
├── A/  B/                                    # 各一个完整 eval 实验形态（基线 / 候选）
└── reports/{comparison.md, fig_metric_diff.png, fig_category_diff.png}
<report-out>/matrix_<ts>/
├── reports/{report.md, matrix_result.json, fig_matrix_accuracy.png}
└── runs/<label>/                             # 各 label 的数据集树副本，内部同 runs/<dataset>/
```

- `<ts>` = `YYYYMMDD-HHMMSS`；`<platform>` 缺省 `default`，两臂平台不同时 compare 目录用 `p1-p2`。
- `runtime/` 每个对话都建（接入侧的运行时数据，如桥的 reset 前备份落 `runtime/reset_backup/`）；
  `memory_snapshot/` 只在 `run --snapshot` 时落（含 `manifest.json`）；对话目录名 = 对话 id
  归一化（非法字符换 `_`，无 id 时 `dialogue`）。
- A/B 副本是源实验目录或数据集目录的整理结果：源是实验目录时整树复制并保留其内部相对位置；
  源是数据集目录时在 `A/runs/<dataset>/` 下重建，缺任务级 `reports/` 时由 compare 从数据集级报告补齐。

## 工作流 5：平台对比

同一 commit 下，仅切换平台（adapter 构造参数或 Agent 侧配置），其余条件锁死：

```bash
python -m membench.cli run --adapter <a> --platform wecom  --commit <SHA> ... --report-out output/
python -m membench.cli run --adapter <a> --platform feishu --commit <SHA> ... --report-out output/
python -m membench.cli compare \
  --baseline <wecom 实验目录或 runs/<dataset> 数据集目录> \
  --candidate <feishu 实验目录或 runs/<dataset> 数据集目录> \
  --require-comparable --tag wecom-vs-feishu --report-out output/
```

解读重点：by_category diff。例如某平台 temporal 显著更低，常见原因是该平台的消息时间戳没传进记忆写入路径。

## 工作流 6：平台 × commit 矩阵

固定顺序跑 4 个组合，标签统一为 `平台@commit`：

```bash
for p in wecom feishu; do for c in $SHA_A $SHA_B; do
  PYTHONPATH=/tmp/agent_$c:<membench> python -m membench.cli run \
    --adapter <a> --platform $p --commit $c --judge llm ... --report-out output/
done; done

python -m membench.cli matrix \
  --runs "wecom@$SHA_A=<r1：实验目录或 runs/<dataset> 数据集目录>" \
         "feishu@$SHA_A=<r2：同上>" \
         "wecom@$SHA_B=<r3：同上>" \
         "feishu@$SHA_B=<r4：同上>" \
  --report-out output/
```

矩阵报告含：每个核心指标的 平台×commit 透视表、6 组两两门禁、分组条形图；产物落 `matrix_<ts>/`——`reports/{report.md,matrix_result.json,fig_matrix_accuracy.png}`，各 label 的数据集树副本落 `runs/<label>/`（label 目录名按 slug 归一，`platform@commit` → `platform-commit`）。`--out DIR` 可直接指定完整实验目录（优先于 `--report-out`）。
**解读顺序**：先看同平台跨 commit（回归），再看同 commit 跨平台（平台差），最后看对角线（交互效应——某平台在某 commit 独差，通常是平台适配代码与新改动的冲突）。

## 公平性检查清单

- [ ] 四个 run 用**同一测试集**（注意：若存在用户测试集，默认会被优先选用——矩阵对比时显式 `--dataset` 锁定）
- [ ] 同一判分器与判分模型（`judge_model` 一致）
- [ ] 同一 LLM 底座与温度（`driver_llm` 一致，建议 temperature=0）
- [ ] 同一适配器实现版本（`adapter_version` 一致）
- [ ] 数据集指纹一致（`dataset_fingerprint.sha256` 一致）
- [ ] 混淆时先核对 `runs/<dataset>/report/metrics.json` 的元数据（`run_id` / `platform` / `commit`）——`run_id` 只记身份，目录名是 `eval_<adapter>_<platform>_<ts>`，不含 commit
