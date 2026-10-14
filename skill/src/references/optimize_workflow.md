# 记忆系统优化闭环操作手册

本文是「工作流 4：记忆系统优化闭环」的详细操作手册。目标是让每一次优化都有**可量化、可回归、可归因**的证据，而不是"改完感觉好像好了"。

闭环六阶段：

```
诊断 → 优化建议 → 优化 plan → subagent 执行 → AB 对比 → 优化报告
```

配套材料：入门教程 `membench/examples/optimize_loop.md`；对比与可比性见 `references/compare_workflows.md`。

> **两个强制检查点**（贯穿全程，不得跳过）
> ① 任何优化/改动完成后，**先询问用户**是否执行 A/B 对比，不得径自开跑；
> ② 启动 A/B 前，**先与用户确认数据集范围**（单集/多集/全量）、A/B 分组（基线 vs 候选）与平台/标签。

---

## 阶段 1：诊断——问题出在哪

**输入**：基线 run 的 `report.md`（必要时加 `metrics.json` / `results.jsonl`）。

**动作**：按顺序读四类信息，形成"短板维度 + 证据"的结论。

1. **综合评级与六维指标**：看哪一维拖后腿（正确性 / 检索 / 记忆动态 / 拒答 / 效率成本 / 稳定性）。
2. **by_category（分题型）**：定位到具体题型，如 `knowledge_update` 差、`temporal` 差、`abstention` 差。
3. **失败样例 Top10**：逐条看是"真错"还是"判 False"（判分噪声），并把错误模式归类。
4. **门禁**：`stale_rate` 是否越过 0.03、`abstention_accuracy` 是否 < 0.8 —— 红线优先于总分。

**从失败样例推到短板维度**（参照 `optimize_loop.md` 的"现象→短板维度"表）：

| 现象 | 短板维度 | 常见根因 |
|---|---|---|
| `by_category` 中 knowledge_update 差、`stale_rate` 高 | 记忆动态 | 写入端"追加"而非"同主题 UPDATE" |
| `abstention` 题乱答 | 拒答 | 检索置信度阈值 / 拒答策略缺失 |
| `evidence_recall` 低 | 检索质量 | 检索器弱（词重叠）、未向量/混合检索 |
| `multi_hop` 差 | 正确性/检索 | 缺多跳检索或推理链 |
| `latency_p95` 高 | 效率成本 | 索引缺失、缓存缺失、LLM 调用过多 |

**产物**：诊断结论（问题维度 + 定位到题型的证据 + 优先修复面）。

**判读标准**：结论必须落到"哪个维度、哪些题型、哪些失败样例支撑"，而不是笼统的"分数不高"。若失败样例里大量是"答案其实对但判 False"，先考虑换 `--judge llm`（见阶段 5）再诊断。

---

## 阶段 2：优化建议——可执行方向清单

**输入**：阶段 1 的诊断结论。

**动作**：针对短板维度给出**可执行**的优化方向清单，按**预期收益**排序；每条写清"改什么、为什么、预期影响哪个指标"。

- 例：`[高] 写入端同主题事实改为 UPDATE 语义 → 预期 stale_rate ↓、knowledge_update accuracy ↑`。
- 优先覆盖门禁红线（`stale_rate` / `abstention_accuracy`）。
- 从 `membench/examples/optimize_loop.md` 的"何时改判分器"判断是否应先把判分切到 LLM-as-Judge。

**产物**：建议清单（方向 + 预期收益 + 目标指标）。

**判读标准**：清单里每条都应能对应到阶段 1 的某个证据；避免"什么都优化一遍"。

---

## 阶段 3：优化 plan——拆成单变量、定验收

**输入**：阶段 2 的建议清单。

**动作**：把建议拆成**单变量**改动，排优先级，界定验收指标：

- **单变量**：每次只改一处（检索策略 / 写入策略 / 拒答阈值……），否则无法归因。
- **优先级**：先动门禁红线，再动总分。
- **验收指标**：写清"改完看哪个指标、期望阈值是多少"，例如 `stale_rate ≤ 0.03 且 accuracy 不降 > 0.02`。
- **数据集与判分器锁定**：plan 里写明本次 AB 用哪个/哪些数据集、哪个判分器——两次 run 之间保持不变。

**产物**：优化 plan（一人一步：改动 + 验收指标 + 数据集/判分器口径）。

**检查点**：进入执行前，若数据集范围或 A/B 分组尚不明确，先跟用户确认（检查点②的前置）。

---

## 阶段 4：subagent 执行——落地单个改动

**输入**：阶段 3 的 plan 中的一条改动。

**动作**：用 subagent 落地这一条改动（只改一处），改完自测跑通：

```bash
cd <skill目录>/membench
python -m membench.cli run --adapter <adapter> --dataset <同一数据集> --judge <同一判分器> --out runs/
```

- 基线 run 请提前备份（如 `cp -r runs/<run_a> runs/baseline_v1`），防止被覆盖。
- 改动过程中如顺手改了别处，**回退**——多变量会让 AB 结论失效。

**产物**：代码改动 + 候选 run。

**⛔ 检查点①**：改动完成后，**先询问用户是否执行 A/B 对比**，不得径自开跑。

---

## 阶段 5：AB 对比——消融 + 可比性前置校验

**输入**：基线 run（baseline）与候选 run（candidate）。

**动作**：

1. **⛔ 检查点②**：先与用户确认**数据集范围**（单集/多集/全量）、**A/B 分组**（基线 vs 候选）与**平台/标签**。
2. **保证消融**：固定驱动模型与 harness，只切优化开关；两次 run 用**同一数据集、同一判分器**。
3. **可比性前置校验 + 对比**：

```bash
python -m membench.cli compare --baseline runs/<run_a> --candidate runs/<run_b> \
  --require-comparable --tag <标签> --report-out output/
```

- `--require-comparable`：关键维度（`driver_llm` / `judge_model` / `adapter_version` / `dataset_fingerprint` 等）不一致或无法核验 → 可比性门禁 **FAIL**（整体 verdict=FAIL、退出码 1），即拒绝当 A/B 用；不加该参数时不一致只降级为 WARN。详见 `references/compare_workflows.md`。
- `--tag <标签>`：决定产物目录 `ab_<tag>_<ts>/`。
- **`compare` 只测差异，不判定两臂是否只差一个变量**——消融靠你自己保证。

**产物**：`ab_<tag>_<ts>/`（A/B 各自报告与 run、`compare_report.md`、`fig_metric_diff.png`）。

**判读标准**：

- 先看门禁结论（accuracy / stale_rate 是否 FAIL），再看逐指标 diff 与 `by_category` diff 定位提升/回退面。
- `FAIL` → 回滚或修复；`PASS` → 固化，进入下一轮或收尾。
- 判分器本身升级过（heuristic → llm）时，基线要按新判分器重跑再 compare（同尺比较）。

---

## 阶段 6：优化报告——方案 + 可归因效果

**输入**：诊断结论、plan、AB 产物。

**动作**：写 `optimization_report.md`，落 `ab_<tag>_<ts>/` 根目录，内容至少包含：

- **方案**：改了什么（单变量）、为什么、对应哪个短板维度；
- **可比性声明**：`driver_llm` / `judge_model` / `adapter_version` / `dataset_fingerprint`（证明同尺）；
- **效果**：门禁结论 + 关键指标 diff（含 `by_category`），明确"涨的是哪个指标、有没有引入回退"；
- **结论与后续**：是否固化、下一轮优化方向。

**产物**：`ab_<tag>_<ts>/optimization_report.md`。

**判读标准**：报告要能让人仅凭它复现结论——方案、口径、数字三者齐全；避免只写"效果很好"。

---

## 常见坑

- **多变量同改**：accuracy 涨了却不知道是哪一处生效。
- **数据集变了但没重跑基线**：diff 无意义（用 `--require-comparable` 兜底）。
- **只看 accuracy**：stale_rate / abstention / latency 的退化同样致命，交给门禁。
- **样本太少就下结论**：24 题里 1 题 ≈ 4 个百分点的 accuracy 波动，扩充黄金集再定胜负。
- **拿 compare 当"消融证明"**：compare 只测差异；消融是你（优化者）的责任。
