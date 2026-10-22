# Adapter 接入指南（把被测 Agent 的记忆系统接上 membench）

本文是「agent 接入 membench」的完整指导：选路径 → 实现三个挂点 → 真实性与口径红线
→ 持久化与自检 → 冒烟与验收。速查卡见 `SKILL.md` 工作流 0；最小可运行示例见
`membench/examples/quickstart.py`；非 Python 系统另见 `references/http_bridge_guide.md`。

membench 与被测系统之间**只有一个协议**：`BaseMemoryAdapter`
（`membench/adapters/base.py`）——实现 `reset / ingest / query` 三个方法即可，
其余（数据集、判分、指标、报告、门禁）全部由 harness 负责。

一次 run 的数据流：

```
数据集(dialogue/questions) → 每个 dialogue 前 reset() → 逐 session ingest(messages)
→ 逐题 query(question) → judge 判分 → 六维指标 + 诊断指标 → metrics.json / report.md
```

## 1. 选接入路径

| 你的被测系统形态 | 路径 | 入口 |
|---|---|---|
| Python 库/进程，记忆写入与检索能在 Python 侧调用 | **① 实现 `BaseMemoryAdapter`**（推荐） | 本文第 2 节 |
| 独立服务（Node/Electron/其他进程），不便嵌 Python | **② HTTP 三端点 + 内置 `HttpAdapter`** | 第 6 节 + `references/http_bridge_guide.md` |
| 记忆能力本身就是"LLM + 工具调用"循环（ingest/query 同构） | **③ `AgentLoopAdapter` 复用循环** | 第 2.5 节 |
| 不改系统、先量现状，或复现线上 badcase | **④ `TranscriptAdapter` 回放真实日志** | 第 2.6 节 |

## 2. 三个挂点怎么找、怎么写

`references/adapter_guide.md` 的定位方法一句话版：**在被测 Agent 源码里找"记忆清空"、
"记忆写入路径"、"记忆读取路径 + 回答生成"三处入口**，包成三个方法。

### 2.1 `reset()`——每个 dialogue 开始前调用

- 必须**真清空全部记忆状态**：向量库 collection、事实表、情景缓冲、图谱，一个都不能漏；
- 只清会话历史不清记忆库 → 跨 dialogue 串扰（典型症状：single_hop 异常高、
  abstention 异常低）；
- **用独立测试实例/库/命名空间，绝不碰生产数据**。

### 2.2 `ingest(messages)`——写入一段会话（逐 session 调用）

- 入参 `messages: [{"role": "user"|"assistant", "content": str, "timestamp": str|None}]`
  ——**忠实回放**，不要增删消息；
- 走**真实写入路径**（抽取/总结/入库/索引），不得用测试专用旁路（见第 3 节）；
- **timestamp 必须传给记忆系统**：办公集与公开集里有大量"昨天/下周"类时间锚题，
  时间戳丢失会让 temporal 类题系统性失败；
- 返回 `IngestStats(input_tokens, llm_calls, latency_ms)`：写入过程触发的 LLM 调用
  （如抽取、总结）必须计入 `llm_calls`，否则效率成本指标失真；填不了的字段填 0
  （报告中显示为 n/a）。

### 2.3 `query(question)`——逐题独立调用，返回答案与检索明细

- 粒度：**逐题独立、不带当轮对话上下文**；不要自行补数据集之外的上下文（跨系统不可比）；
- 走**真实读取路径 + 回答生成**：检索、重排、注入、生成，与线上同一套；
- `retrieved` 决定 `evidence_recall` / `recall@5` 是否成立，口径 = **注入即口径**（四条规则）：
  1. **内容**：只填**实际拼进回答 prompt 的记忆条目原文**——不是摘要、不是条目 ID、不是给你自己的查询语句、更不是答案本身；
  2. **顺序**：按**注入优先级**排列（最相关的在前）——`recall@5` 只看前 5 条，顺序即优先级；
  3. **合并与去重**：多次检索（先搜 → 改写 query → 再搜）的结果**先合并、再去重**再返回（重复条目按首次出现保留）；框架不合并、不去重，重复项会白占前 5 窗口，丢后续轮次会让 `evidence_recall` 系统性偏低；
  4. **条数**：报**注入的那一份全量**——不要只报 top-k（`evidence_recall` 扫全部返回条目，报少了会虚低），也不要报未经注入裁剪的内部原始 top-N（报多了会虚高）。
- 落地写法：在**构造 prompt 的那一处**把选中的列表直接返回，别事后另算一遍（另算必然与注入内容漂移）；
- 填不出检索细节时至少返回注入的记忆文本；
  - 摘要/改写式存储：回填原文片段，或用 `--evidence-match overlap|llm` 换命中口径
    （口径会记入 `metrics.json`，不同口径的跑分不可直接对比）；
  - 若报告出现"两项检索指标全为 0"的警示，先排查证据原文是否根本不在检索结果里，
    再怀疑检索能力；
- 返回 `QueryResult(answer, retrieved, latency_ms, input_tokens, output_tokens, llm_calls)`：
  查询过程触发的全部 LLM 调用与 token 都要计入；填不了填 0。

### 2.4 时间锚：temporal 题的正确性从哪来

- 消息时间戳（`messages[].timestamp`）是**唯一时间锚**：逐条原样传给 ingest；被测系统
  解释"昨天 / 上周 / 上季度 / 上次会议"时必须相对**消息时间线**，而不是机器时钟、
  也不是记忆的写入时刻；
- **格式已统一为 ISO 8601，精度保持原样**（转换器归一，`membench/data/timestamps.py`）：
  只到日的仍是 `YYYY-MM-DD`（不补 00:00）、只到分的仍是 `YYYY-MM-DDTHH:MM`（不补秒）、
  原到秒的保留到秒、原始没有时区就不造时区。转换前的原始形态（供对照）：
  LoCoMo `1:56 pm on 8 May, 2023`、LongMemEval `2023/04/10 (Mon) 17:50`、
  BEAM `March-01-2024`、GateMem/内置集本来就是 ISO。**粒度不同依旧存在**：
  LoCoMo / LongMemEval 是**会话级**（同一 session 内所有消息共用一个时间）、
  GateMem **逐轮**、BEAM **极稀疏**（每个 batch 首条 user 消息一个只到日的锚点，
  约 0.6% 消息带时间戳）。所以解析要按"可能缺时间、可能只有日期"来写，
  Python 3.10 下 `datetime.fromisoformat` 不认缺秒的串，稳妥写法：

  ```python
  from datetime import datetime
  def parse_ts(v: str) -> datetime:
      for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
          try:
              return datetime.strptime(v, fmt)
          except ValueError:
              continue
      raise ValueError(f"非 ISO 时间戳：{v!r}")
  ```
- 不认识的形态在 `dataset fetch` 阶段就会**报错停下**（不静默留混格式）：混进去之后
  只会在个别题上无声失效，比 fetch 期报错难查得多；用户自建集请直接写 ISO 8601；
- membench **不会**把数据集时间平移到"现在"，分数与墙钟无关（stale / abstention
  红线也是规则判定）——数据集是 2026-09 的对话、评测跑在哪一天，都不影响正确性；
  参考：office 集的 gold 会把锚点写明（如 `9月11日（9月12日说的"昨天"）`）；
- **验证方法（5 分钟）**：同一对话构造两份时间戳版本（整体平移 3 年），各跑一遍，
  答案应完全一致；若随评测日期或写入时刻变化 → 系统缺陷（用墙钟解释相对时间），
  不要用"数据集时间对不上"去解释或掩盖；
- 系统"写入时刻 = 事件时间"是个常见坑：若系统改不动，只能在**数据集构造阶段**
  整体平移时间戳（同 delta、保持相对结构），且因 `dataset_fingerprint` 变化
  **必须重跑基线**；membench 无内置平移工具，绝不要评测期偷偷平移；
- 消息缺 `timestamp` 的数据集：temporal 题无法锚定，属数据集缺陷，构造时补齐
  （见 `references/user_dataset_guide.md`）。

### 2.5 `AgentLoopAdapter`（路径③速览）

若 ingest/query 是同一套 tool-call 骨架，用
`membench.agent_loop.AgentLoopAdapter` 只给 `{llm_client, tools, system_prompt}`：

- `llm_client` 三选一：带 `chat_with_tools(messages, tools)` 的对象（原生 tool-call，
  优先）> 带 `chat`/`complete` 的对象或可调用对象（文本 JSON 约定）> `"module:ClassName"`
  类路径字符串（与 `--judge client` 同约定）；
- `tools` 元素 `{"name", "schema", "handler"}`；handler 收解析后的 arguments，
  返回 str/dict；handler 抛异常会折叠成观察文本回填、不中断循环；
- `retrieved` 回填由 `retrieved_from(question, loop_result)` 决定，缺省把工具名含
  `retrieve` / `search` / `recall` 的**多轮调用结果按调用顺序合并、去重**——**检索工具名要对得上关键词、结果要是
  被检索到的记忆原文**，否则请自传 `retrieved_from`；
- 循环内每一轮 LLM 调用都是被测系统的真实成本，membench 会自动统计。

### 2.6 `TranscriptAdapter`（路径④速览）

把线上真实「写入/检索」日志导出为离线 transcript（对话 + 每题实际检索到的记忆条目），
回放注入。适合先量现状、复现 badcase；**它固定了写入侧**，属降级口径：按第 3 节
由用户拍板、元数据声明、A/B 同口径，结论限定为"读取/回答层评测"。

## 3. 真实性红线（评测有效性的前提，必读）

**原则：adapter 内调用的应当是线上同一套代码/服务，只有环境不同（独立测试库/命名空间）。
这是"完整评测"分数成立的唯一前提。** 三条红线（违反其一，即不得以"完整评测"名义汇报分数）：

1. **不得用测试专用旁路代替真实写入路径**——例如跳过抽取/索引逻辑直接 insert 原文；
2. **不得用格式仿真的手写文件冒充真实写入**（如绕过记忆 MCP/库，按记忆格式自写文件）。
   写入侧能力与成本一行没跑，分数不可归因到该记忆系统；且格式"像"≠路径"真"，
   系统可能以降级路径（读时重建索引等）消费你的文件，读取行为也不是常态；
3. **不得绕过真实检索直接问底座模型**——分数虚高，`evidence_recall` 会暴露。

### 走不了真实写入路径时：降级流程（不是放行通道）

如果真实写入路径在评测环境不可用或不现实，**不要伪装**，按降级处理：

1. **停止并告知用户**：完整评测需要真实写入路径，当前不可行；列出可选口径
   （a. 先跑"读取/回答层评测"；b. 修环境后重跑；c. 放弃本次评测）；
2. **由用户拍板选择口径**。执行者（尤其 agent）不得自行降级，更不得把旁路结果
   当完整评测汇报——本节不提供任何"做了某件事就能放行"的自助路径；
3. 用户选择读取层评测后：在适配器元数据（`--adapter-meta` / `describe()`）与结论中
   明确"写入路径未覆盖、ingest 成本未计"；A/B 两侧同口径（一侧真实一侧旁路不是 A/B）。
4. **可选的等价性排查（小样本、非放行条件）**：仅在"真实写入能小样本跑、但无法全量
   跑"这类场景下，为口径措辞提供证据——真实写入 → 真实检索 vs 旁路写入 → 同一真实
   检索，对比检索结果与 `recall@5`。结论一致只能用于"该写入层是薄存储、存储差异
   对读取影响有限"的说明；**即便一致，也不得把旁路结果称作完整评测**，不一致则连
   读取层结论都需加注存储差异。

## 4. 元数据、统计与并发

- `name` 带版本（如 `my-agent-v1`），便于 compare 区分；
- 评测时传 `--driver-llm`（驱动被测系统的 LLM 标识）与 `--adapter-version`
  ——可比性前置校验的关键维度，缺失会记"无法核验"；
- **统计诚实**：token / `llm_calls` 填不了可以填 0（报告标 n/a），但填得出来的
  （尤其写入侧抽取、查询侧生成的调用）**必须计**，否则效率成本与"token 正收益"
  类结论失真；
- 并发：`--jobs>1` 时 query / judge / 证据匹配按题目并发执行，**要求 `query` 线程安全**
  （ingest 始终串行，不受影响）；适配器内部有共享可变状态（缓存、连接池）时先排查。

## 5. 持久化与加载

```bash
cd <skill目录>/membench
python -m membench.cli adapter install /path/to/your_adapter.py --name <适配器名>
# 落盘 ~/.membench/adapters/（可用 MEMBENCH_ADAPTERS_DIR 覆盖）
python -m membench.cli adapter list      # 查看
python -m membench.cli adapter remove <名>  # 删除
```

- 临时/CI 场景用环境变量 `MEMBENCH_ADAPTER_MODULES=module.path`（同名时优先于持久化）；
- 也可以在 Python 侧直接传实例（`Runner(MyAdapter(), ...)`），不经注册表。

## 6. HTTP 路径：契约与自检

在 Agent 侧实现三个端点（细节与模板见 `references/http_bridge_guide.md`、
`membench/examples/http_bridge_node/`）：

- `POST /reset` → 2xx；
- `POST /ingest` body `{"messages": [{"role","content","timestamp"}]}` → 2xx +
  可选 `{"input_tokens": int, "llm_calls": int, "latency_ms": float}`；
- `POST /query` body `{"question": str}` → 2xx + JSON + 字符串 `answer`；
  `retrieved` 若出现须为字符串列表。

正式评测前先自检（连通性 + 契约校验，避免把"接错"误判成"记忆能力差"）：

```bash
python -m membench.cli doctor --target http://127.0.0.1:8787 [--timeout 10.0] [--api-key K]
```

## 7. 冒烟与验收

```bash
cd <skill目录>/membench
# 1) 冒烟：迷你集跑通链路
python -m membench.cli run --adapter <你的adapter> --dataset membench/data/sample_dataset.jsonl --judge heuristic
# 2) 业务验收：办公黄金集
python -m membench.cli run --adapter <你的adapter> --dataset membench/data/office_dataset.jsonl --judge heuristic
```

**验收判据**：

- pipeline 跑通，报告各指标可解释；
- `stale_rate` 与 `abstention_accuracy` 的**缺陷暴露合理**（这两项是红线类指标，
  规则判定、不依赖 LLM 判分）；
- **若全部满分，先回头查接错**（query 没走记忆、reset 没清库、判分口径等），
  而不是庆祝。

## 8. 判分器三选一

| 方式 | 命令 | 适用 |
|---|---|---|
| `--judge heuristic` | 零配置 | 接入冒烟；stale / abstention 红线始终规则判定 |
| `--judge llm` | `--llm-base-url / --llm-api-key / --llm-model` | **正式评估首选**：第三方模型，避免自评偏差 |
| `--judge client` | `--client-llm "my_agent.llm:LLMClient"` | 复用 agent 自己的 LLM 快速验证；报告需注明同源偏差 |

`--client-llm` 接受 `module:ClassName`（无参实例化）或返回字符串的函数；类需带
`.chat(prompt)->str` 或 `.complete(prompt)->str`；名称不同就继承
`membench.judges.client.ClientLLMJudge` 重写 `_call()`。

## 9. 接入完成定义（Checklist）

- [ ] 三个方法走的是**线上同款**写入/读取路径（完整评测的唯一前提）；若降级为读取层评测：已由用户确认口径、元数据（`--adapter-meta`）声明、A/B 同口径；
- [ ] `reset` 真清空（独立测试实例，不碰生产）；
- [ ] timestamp 已落库；`retrieved` 填的是注入 prompt 的记忆原文；
- [ ] 写入/查询触发的 LLM 调用与 token 已计入统计；
- [ ] 适配套件已持久化（`adapter install`）或配置 `MEMBENCH_ADAPTER_MODULES`；
- [ ] HTTP 路径已过 `doctor`；
- [ ] sample 冒烟通过，office 验收跑通且缺陷暴露合理；
- [ ] 需要 `--jobs>1` 时，`query` 已确认线程安全；
- [ ] **评测启动前已与用户确认召回口径**（`--evidence-match`：`substring` / `overlap` / `llm`；对比两侧同口径）——摘要/改写式存储切勿用默认 `substring` 跑长任务，确需字面口径则先回填原文片段。

## 10. 常见接错（自查清单）

- [ ] `query` 里把问题直接丢给底座模型而没走记忆检索 → 分数虚高，证据召回字段暴露；
- [ ] 手写文件仿记忆格式、绕过真实写入管线 → 违反红线，分数不可归因；走不通真实路径时按第 3 节降级流程：停下、告知用户、由用户拍板，不得自行放行；
- [ ] `reset` 只清了会话没清记忆库 → 跨 dialogue 串扰（single_hop 异常高、abstention 异常低）；
- [ ] `ingest` 里同步调了生成模型做总结但没计入 `llm_calls` → 成本指标失真；
- [ ] timestamp 没传给记忆系统 → temporal 题系统性失败；
- [ ] 系统用机器时钟/写入时刻解释"昨天/上周"（而不是消息时间戳）→ temporal 题随评测日期漂移；用"整体平移 3 年"的对照验证（见第 2.4 节）；
- [ ] 检索工具名不含 `retrieve/search/recall` 关键词且未自传 `retrieved_from`
  （AgentLoopAdapter）→ 证据回填空，检索指标全 0。
