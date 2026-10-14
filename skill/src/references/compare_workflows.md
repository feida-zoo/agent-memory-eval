# Commit 对比与平台对比工作流详解

## 前置假设

- 被测 Agent 是 git 仓库，两个 commit 都能独立构建/运行；
- adapter 已实现，且能以参数切换平台配置（多平台场景）。

## 可比性前置校验（A/B 前必做）

**`compare` 只测量差异，不判定两臂是否只差一个变量。** "驱动/判分模型、适配器实现、数据集"任一变了，PASS/FAIL 都不可归因——这是本波修复的核心问题。因此对比前必须显式校验关键维度。

| 关键维度 | 记录字段（`metrics.json`） | 不一致的后果 |
|---|---|---|
| 驱动（被测 agent 的底座）模型 | `driver_llm` | 分数差异混入"换了模型"的差异 |
| 判分模型 | `judge_model` | 判分尺度不同，diff 是噪声 |
| 适配器实现 / 版本 | `adapter_version` | 接入层改动被误当成记忆能力改动 |
| 数据集 | `dataset_fingerprint`（`sha256` / `bytes` / `name`） | 题目不同，逐指标 diff 无意义 |
| 平台 | run 的 `platform`（`--platform`） | A/B 分组错配 |

- `run` 用 `--driver-llm` / `--adapter-version` 显式落账这些字段。
- `compare --require-comparable` 做**前置校验**（strict）：任一关键维度**不一致**或**无法核验**（元数据缺失）→ 可比性门禁 **FAIL**，整体 verdict=FAIL、退出码 1（即拒绝当 A/B 用）；报告含「可比性前置校验」小节与「不可比原因」清单。
- 不加 `--require-comparable` 时：关键维度不一致 → 可比性门禁 WARN（verdict 至少 WARN，不会静默 PASS）；关键维度无法核验 → 只在 CLI 输出与报告里提示，不影响 verdict。
- `compare --tag <标签>` 给本次对比命名，决定产物目录 `ab_<tag>_<ts>/`。

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

# 4) 对比（baseline=旧 commit；先做可比性前置校验）
python -m membench.cli compare --baseline runs/<runA> --candidate runs/<runB> \
  --require-comparable --tag <标签> --report-out output/

# 5) 清理
git -C $AGENT_REPO worktree remove /tmp/agent_old
git -C $AGENT_REPO worktree remove /tmp/agent_new
```

门禁规则（默认，CLI 可调）：accuracy 降 >0.02 → FAIL；stale_rate 升 >0.02 → FAIL；p95 延迟升 >30% → WARN。

**产物目录（本波落地）**：

```
<report-out>/ab_<tag>_<ts>/
├── A/{report.md, metrics.json, fig_*.png, runs/<run_id>/}   # 基线（旧 commit）
├── B/{report.md, metrics.json, fig_*.png, runs/<run_id>/}   # 候选（新 commit）
├── compare_report.md / fig_metric_diff.png                   # 对比报告与逐指标 diff 图
└── optimization_report.md（可选，优化闭环写入）
```

## 工作流 5：平台对比

同一 commit 下，仅切换平台（adapter 构造参数或 Agent 侧配置），其余条件锁死：

```bash
python -m membench.cli run --adapter <a> --platform wecom  --commit <SHA> ...
python -m membench.cli run --adapter <a> --platform feishu --commit <SHA> ...
python -m membench.cli compare --baseline runs/<wecom run> --candidate runs/<feishu run> \
  --require-comparable --tag wecom-vs-feishu --report-out output/
```

解读重点：by_category diff。例如某平台 temporal 显著更低，常见原因是该平台的消息时间戳没传进记忆写入路径。

## 工作流 6：平台 × commit 矩阵

固定顺序跑 4 个组合，标签统一为 `平台@commit`：

```bash
for p in wecom feishu; do for c in $SHA_A $SHA_B; do
  PYTHONPATH=/tmp/agent_$c:<membench> python -m membench.cli run \
    --adapter <a> --platform $p --commit $c --judge llm ...
done; done

python -m membench.cli matrix \
  --runs "wecom@$SHA_A=runs/<r1>" "feishu@$SHA_A=runs/<r2>" \
         "wecom@$SHA_B=runs/<r3>" "feishu@$SHA_B=runs/<r4>" \
  --report-out output/
```

矩阵报告含：每个核心指标的 平台×commit 透视表、6 组两两门禁、分组条形图；产物落 `matrix_<ts>/`（各臂 run 副本在 `runs/<label>/`）。
**解读顺序**：先看同平台跨 commit（回归），再看同 commit 跨平台（平台差），最后看对角线（交互效应——某平台在某 commit 独差，通常是平台适配代码与新改动的冲突）。

## 公平性检查清单

- [ ] 四个 run 用**同一测试集**（注意：若存在用户测试集，默认会被优先选用——矩阵对比时显式 `--dataset` 锁定）
- [ ] 同一判分器与判分模型（`judge_model` 一致）
- [ ] 同一 LLM 底座与温度（`driver_llm` 一致，建议 temperature=0）
- [ ] 同一适配器实现版本（`adapter_version` 一致）
- [ ] 数据集指纹一致（`dataset_fingerprint.sha256` 一致）
- [ ] runs 目录里 run_id 含 platform 与 commit，混淆时先核对 metrics.json 元数据
