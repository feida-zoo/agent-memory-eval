# 优化迭代闭环教程（membench optimize loop）

本文给出用 membench 做记忆系统优化的完整闭环：**建立基线 → 单变量改动 → compare 门禁 → 迭代**。目标是让每一次优化都有可量化、可回归的证据。

---

## 0. 总览

```
优化循环：
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ run 基线      │ ──▶ │ 单变量改动    │ ──▶ │ run 候选      │ ──▶ │ compare 门禁  │
  │ (baseline)   │     │ (只改一处)    │     │ (candidate)  │     │ PASS/FAIL    │
  └──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
        ▲                                                               │
        └────────────────────── FAIL 则回滚 / PASS 则固化 ◀─────────────┘
```

核心纪律：

1. **一次只改一个变量**（检索策略、写入策略、拒答阈值……），否则无法归因。
2. **数据集与判分器在两次 run 之间保持不变**，只变被测系统。
3. **compare 的门禁结论说了算**，不要只看 accuracy 单一指标。

---

## 1. 建立基线

先用当前系统跑一版基线，保存好 run 目录：

```bash
python -m membench.cli run --adapter my-system \
    --dataset membench/data/sample_dataset.jsonl \
    --judge heuristic --out runs/
# 输出: runs/my-system_sample_dataset_20250917_120000/
```

基线 run 目录里：

- `results.jsonl`：逐题原始记录（judge 判定、检索命中、延迟、token）。
- `metrics.json`：六维汇总指标。
- `report.md`：含失败样例 Top10 —— **这是排优化优先级的依据**。

> 建议：把基线 run 目录改名或备份（如 `cp -r runs/<run_a> runs/baseline_v1`），防止后续被覆盖或混淆。

**读报告找短板**：

| 现象 | 短板维度 | 优先优化方向 |
|---|---|---|
| by_category 中 knowledge_update 差、stale_rate 高 | 记忆动态 | 写入端：同主题事实 UPDATE 而非追加 |
| abstention 题乱答 | 拒答 | 检索置信度阈值 + 拒答策略 |
| evidence_recall 低 | 检索质量 | 检索器（词重叠 → 向量/混合检索） |
| multi_hop 差 | 正确性/检索 | 多跳检索或推理链 |
| latency_p95 高 | 效率成本 | 索引、缓存、减少 LLM 调用 |

## 2. 单变量改动

选定一个假设，例如「旧事实不覆盖导致 stale」，只改写入逻辑：

```python
# 改动前：ingest 追加
self.store.append(fact)
# 改动后：同 subject 覆盖（UPDATE 语义）
self.store[subject_of(fact)] = fact
```

其他一切（检索、生成、数据集、判分器）保持不动。

## 3. 跑候选并对比

```bash
python -m membench.cli run --adapter my-system --dataset <同一数据集> --out runs/
python -m membench.cli compare --baseline runs/<run_a> --candidate runs/<run_b>
# 需要自定义门禁时:
#   --acc-threshold 0.02 --stale-threshold 0.02 --latency-warn 0.30
```

门禁规则（SPEC 3.6）：

- accuracy 下降 > 0.02 → **FAIL**（回归，必须回滚或修复）
- stale_rate 上升 > 0.02 → **FAIL**
- latency_p95 上升 > 30% → **WARN**（精度换延迟，人工决策是否接受）
- 其余指标（token_f1、bleu1、evidence_recall、token 成本等）只报告、不门禁

`compare_report.md` 里有逐指标 diff 表和 `fig_metric_diff.png` 对比图。CI 中可直接用退出码：`FAIL` 时进程退出码为 1。

## 4. 何时改判分器

默认 `heuristic` 判分（token 包含匹配 + 拒答检测）零成本、可复现，适合快速迭代。但以下情况应切换到 LLM-as-Judge：

- 答案表述多样、与 gold 不字面重合（开放题、偏好题），启发式误判增多；
- 失败样例 Top10 里出现大量「答案其实对，但判 False」的样例；
- 优化进入收尾阶段，需要更接近人评的结论做验收。

```bash
python -m membench.cli run --adapter my-system --dataset <数据集> \
    --judge llm --llm-base-url https://api.example.com/v1 \
    --llm-api-key $KEY --llm-model gpt-4o-mini
```

注意：

- LLMJudge 调用失败会回退启发式并在记录里标记 `judge=llm-fallback`，报告中可统计回退比例。
- **对比时两次 run 必须用同一判分器**，否则 diff 混入判分噪声。
- 判分器本身升级（heuristic → llm）时，应对基线重跑一次再 compare，即「同尺比较」。

## 5. 扩充黄金集（自建数据集）

内置 `sample_dataset.jsonl` 只够冒烟。真实优化需要覆盖你业务分布的黄金集。格式：JSONL，**每行一个 dialogue**：

```json
{
  "dialogue_id": "d001",
  "sessions": [
    [{"role": "user", "content": "我养了一只狗，叫旺财。", "timestamp": "2025-01-01T10:00:00"},
     {"role": "assistant", "content": "旺财真可爱！", "timestamp": "2025-01-01T10:00:05"}]
  ],
  "questions": [
    {"qid": "d001_q1", "question": "用户的狗叫什么名字？", "gold_answer": "旺财",
     "category": "single_hop", "evidence": ["叫旺财"]}
  ]
}
```

字段要求：

- `sessions`：list of session，每个 session 是消息 list，消息含 `role`/`content`/`timestamp`（timestamp 可空）。
- `questions[].category` ∈ `single_hop | multi_hop | temporal | knowledge_update | abstention | preference | open_ended`。
- `evidence`：对话中支持答案的**原文子串**列表，用于检索召回判定。
- `gold_stale`：仅 knowledge_update 题必填，值为过期答案。

各类题型构造建议：

| category | 构造方法 | 示例 |
|---|---|---|
| single_hop | 单条消息里的事实直接提问 | 「我住在北京」→ 用户住在哪里？ |
| multi_hop | 答案需组合 2+ 条不同 session 的信息 | A 说公司名、B 说公司地址 → 用户公司在哪？ |
| temporal | 同一主题多个时间点，问「最近/现在」 | 1 月住朝阳、3 月搬海淀 → 用户现在住哪？ |
| knowledge_update | 对话中明确更正事实；gold_answer=新值、gold_stale=旧值 | 「狗叫旺财」→「改名来福了」→ 狗叫什么？ |
| abstention | 问对话中从未出现的信息；gold_answer 固定 `"<ABSTAIN>"` | 对话没提学历 → 用户什么学历？期望拒答 |
| preference | 喜好/习惯类 | 「我不吃香菜」→ 点外卖要注意什么？ |
| open_ended | 开放式问题，gold 给要点 | 适合配合 LLM judge 使用 |

质量要点：

1. **evidence 必须是对话原文的子串**（可验证检索），不要复述。
2. knowledge_update 题建议 ≥4 道、abstention ≥3 道，否则 stale_rate / abstention_accuracy 统计噪声大。
3. 每个 dialogue 2–3 个 session、每 session 4–8 条消息，模拟真实多会话间隔。
4. 混入口语化、错别字、指代（「它」「那边」），避免黄金集过于干净而高估系统。

## 6. 常见坑

- **数据集变了但没重跑基线**：compare 假设两次 run 同数据集同判分器，否则 diff 无意义。
- **多变量同改**：accuracy 涨了却不知道是哪一处生效。
- **只看 accuracy**：stale_rate、abstention、latency 的退化同样致命，交给门禁。
- **样本太少就下结论**：24 题里 1 题 ≈ 4 个百分点的 accuracy 波动，扩充黄金集再定胜负。
