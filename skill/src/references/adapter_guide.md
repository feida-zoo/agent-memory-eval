# Adapter 接入指南（把被测 Agent 的记忆系统接上 membench）

本文是「agent 接入 membench」的完整指导：选路径 → 实现三个挂点 → 真实性与口径红线
→ 持久化与自检 → 冒烟与验收。命令与流程编排见 `SKILL.md` 的决策表与标准流程；最小可运行示例见
`membench/examples/quickstart.py`；非 Python 系统另见 `references/http_bridge_guide.md`。

membench 与被测系统之间**只有一个协议**：`BaseMemoryAdapter`
（`membench/adapters/base.py`）。契约分三组：

- **必需**：`name` 属性 + `reset / ingest / query` 三个方法——实现这三个就完成了接入；
- **可选·快照组（4）**：`snapshot_capabilities()` / `snapshot()` / `restore()` / `rebuild()`
  ——用于 `--replay` 免重跑 ingest，**声明了就必须实现**；
- **可选·自描述（1）**：`describe()`——返回 dict 写进 `metrics.adapter_describe`，
  **不参与评分与可比性**。

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
| 独立服务（Node/Electron/其他进程），不便嵌 Python | **② HTTP 三必需端点 + 五个可选端点 + 内置 `HttpAdapter`** | 第 6 节 + `references/http_bridge_guide.md` |
| 记忆能力本身就是"LLM + 工具调用"循环（ingest/query 同构） | **③ `AgentLoopAdapter` 复用循环** | 第 2.5 节 |
| 不改系统、先量现状，或复现线上 badcase | **④ `TranscriptAdapter` 回放真实日志** | 第 2.6 节 |

## 2. 必需三挂点怎么找、怎么写

本节的一句话版：**在被测 Agent 源码里找"记忆清空"、
"记忆写入路径"、"记忆读取路径 + 回答生成"三处入口**，包成三个方法。
快照组与自描述是可选项，见 §2.7 / §2.8。

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

### 2.7 状态快照与回放（可选能力）

改了**读取侧**（检索/排序/回答）代码后，不必重新 ingest 整个数据集：用上一次评测导出的
「逐对话状态快照」直接回放。框架只做编排（目录与 tag、清单、哈希、跑前校验、保真校验），
状态的产出与消费**全部走你的代码**——框架不解析快照内容，也不读你的存储格式。

需要实现**快照组的四个可选方法**（不实现 = 不支持本能力，框架明确提示而不是静默跳过）：

```python
def snapshot_capabilities(self) -> dict:
    return {"supported": True, "restore": True, "rebuild": False,
            "format_version": 1, "compatible_from": [1]}   # 声明了就必须实现对应方法

def snapshot(self, target_dir: str, tag: str) -> dict:   # 导出当前状态（tag=对话 id）
def restore(self, source_dir: str, tag: str) -> None:    # 同版本恢复（只接受当前 format_version）
def rebuild(self, source_dir: str, tag: str) -> None:    # 跨版本 / 重建派生结构（接受 compatible_from）
```

`restore` 只在快照格式与当前代码兼容时使用；`rebuild` 拿快照里的**源数据**，用当前
代码的构建路径重建索引 / 向量 / 摘要 / 缓存，所以存储结构或 schema 变了也能回放，
且不必重跑 LLM 抽取。

- **时机与落位**：runner 在每个对话 ingest 完成后、第一道 query 之前调用 `snapshot()`——回放
  就是"恢复这个状态，然后跑该对话的题"；runner 每对话 `reset()`，所以按对话切片天然等价。
  快照落该对话目录的 `memory_snapshot/`（框架另写一份 `manifest.json` 清单），完整产物布局见
  `membench/README.md`「产物布局」；
- **回放时同样会调用 `snapshot()`**：`restore()` / `rebuild()` 完成后、第一道 query 之前，
  框架把恢复出来的状态再导出一次（这是"这次回放保真吗"的证据）。落点由框架决定：普通 run 是
  对话目录的 `memory_snapshot/`，回放任务则是任务级内容寻址池 `snapshots/<内容哈希前 12 位>/`
  ——状态内容相同的导出（例如 A/B 两组回放同一源）共用同一个桶，你不需要感知或自己拼接路径；
  导出的内容哈希口径不变（排除框架写的 `manifest.json`），因此同一份状态无论落到哪里哈希一致；
- **默认导出**：`run` / `evaluate` / `evaluate-compare` 默认就导出（每个对话只导这一份，
  per-session 导出是没用的中间态），不需要额外加开关，目录也固定不用指定；不需要时用
  `--no-snapshot` 关闭（回放时连池也不写）。跑完 ingest 就能在对话级
  `report.md` 里点开 `../memory_snapshot/` 翻记忆文件（报告另给清单条目数与体积），不必自己拼路径；
- **内容边界**：快照只放**可检索记忆**（回答问题时真正会取用的那份状态），**不要塞原始
  日志 / transcript / 逐轮对话归档**。框架会扫描快照里的文本文件判定 `evidence_in_state`：
  safety 题按「记忆里还能查到被删除的敏感串 = 没真遗忘」判错，把原始日志一起导出等于把
  "日志里出现过"当成"记忆仍记得"，整类 safety 题会被误判成泄露；失败归因也会把日志里的
  字面命中当成"证据已写入"。原始输入 / 调试日志放对话目录的 `runtime/`（框架不扫描它），
  不要放进 `memory_snapshot/`；框架会在检测到 `*trace*` / `*log*` / `query*` 这类文件名时
  **告警并落账 `metrics.state_scan.suspect_files`**（只提示、不排除——若这些文件同时被
  检索覆盖，排除会把真实泄露藏起来）；
- **不实现会怎样**：框架打印一次警告并**跳过**快照，评测照常完成——`metrics.json` 的
  `state_snapshot.role` 记 `unsupported`（报告渲染成「不支持（适配器未声明快照能力）」而不是
  空值），除了不能用 `--replay` 免重跑 ingest 之外没有其它影响；
- **回放命令**：`run --replay <实验目录或 runs/<dataset> 数据集目录>`（也接受回放任务的任务根 /
  组目录）——跳过 reset+ingest 只跑 query，清单齐备性 / 数据集指纹 / 对话覆盖 / 能力 / 模式都在
  开跑前校验，失败即停不留半成品；
- **模式**：`full` / `replay-restore`（格式兼容）/ `replay-rebuild`（格式不兼容或需重建
  派生结构）——不兼容且没声明 `rebuild` 会被**拒绝**，提示必须重新 ingest；
- **口径**：回放固定写入侧，结论只覆盖读取/回答层；`run_mode` 是 compare 的关键维度，
  回放分与全量分判不可比；
- **`replay_verify_source` / `--replay-verify-source`**：回放时与源产物逐题比对，写进
  `metrics["replay_verify"]` 与报告——这就是"回放误差地板"（未改代码时答案应逐题一致，
  不一致说明恢复不保真，先修这个再谈 A/B）。**入参为源评测的实验目录或数据集目录**
  （`<实验目录>` 或 `<实验目录>/runs/<dataset>`，实现按 `<数据集目录>/report/results.jsonl`
  取源明细）；
- **必须重跑的场景**：改动落在写入侧（抽取/更新/去重/写入成本）或逻辑 schema 变更且系统
  没有 `rebuild`；
- **HTTP 桥**：同款端点 `POST /snapshot-capabilities|/snapshot|/restore|/rebuild`（见
  `references/http_bridge_guide.md`）——桥侧要注意"恢复后重建进程内实例"，只换磁盘不重建
  缓存等于没恢复。

### 2.8 `describe()`——自描述（可选，不参与评分）

返回一个 dict，写进 `metrics.json` 的 `adapter_describe`，并在报告「适配器自描述」小节
展示；**不参与评分，也不是 compare 的可比性维度**。适合放 `embedding_model` / `retrieval` /
`top_k` / `index_type` / `features` 这类"事后看懂这跑分背后系统长什么样"的信息（比
`--adapter-meta` 自由文本更结构化）。

```python
def describe(self) -> dict:
    return {"embedding_model": "bge-m3", "retrieval": "hybrid",
            "top_k": 8, "index_type": "hnsw", "features": ["summary_cache"]}
```

默认返回 `{}`（报告里该小节整节不出现）；抛异常只打印一次警告并记 `None`，评测照常完成；
返回非 dict 会被忽略并警告。HTTP 路径对应 `POST /describe`（缺失即视为没有自描述）。

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

把适配器落盘：`python -m membench.cli adapter install /path/to/your_adapter.py --name <适配器名>`（落 `~/.membench/adapters/`，可用 `MEMBENCH_ADAPTERS_DIR` 覆盖；`adapter list` / `adapter remove` 见 `membench/README.md`「适配器持久化」）。

- 临时/CI 场景用环境变量 `MEMBENCH_ADAPTER_MODULES=module.path`（同名时优先于持久化）；
- 也可以在 Python 侧直接传实例（`Runner(MyAdapter(), ...)`），不经注册表。

## 6. HTTP 路径：契约与自检

在 Agent 侧实现**三个必需端点**（细节与模板见 `references/http_bridge_guide.md`、
`membench/examples/http_bridge_node/`）：

- `POST /reset` → 2xx；
- `POST /ingest` body `{"messages": [{"role","content","timestamp"}]}` → 2xx +
  可选 `{"input_tokens": int, "llm_calls": int, "latency_ms": float}`；
- `POST /query` body `{"question": str}` → 2xx + JSON + 字符串 `answer`；
  `retrieved` 若出现须为字符串列表。

**可选端点五个**（缺失即视为不支持，不报错、不中断评测）：`POST /snapshot-capabilities`、
`POST /snapshot`、`POST /restore`、`POST /rebuild`、`POST /describe`——语义与 Python 侧的
快照组 / `describe()` 一一对应（`/restore` 只接受当前格式版本，`/rebuild` 接受
`compatible_from` 列出的旧版本并重建派生结构）。`doctor` 只探测三个必需端点。

正式评测前先自检（连通性 + 契约校验，避免把"接错"误判成"记忆能力差"）：

```bash
python -m membench.cli doctor --target http://127.0.0.1:8787 [--timeout 10.0] [--api-key K]
```

自检通过后用 `--adapter http --adapter-arg base_url=<服务地址>` 直连跑评测（`base_url` 是
`HttpAdapter` 的构造参数，不需要为桥再包一层适配套件；命令形态见 `SKILL.md` 标准流程）。
接受的构造参数：`base_url`（必填）、`api_key`、`timeout`；缺必填参数 / key 写错会在开跑前报错。

## 7. 冒烟与验收

冒烟用 `membench/data/sample_dataset.jsonl`，业务验收用 `membench/data/office_dataset.jsonl`
（命令同 `SKILL.md` 标准流程「评测」，`--dataset` 换成对应路径）。

产物布局见 `membench/README.md`「产物布局」；接入排查最常用的逐题明细（含 `retrieved`）在
`<数据集目录>/report/results.jsonl`。

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

- [ ] **必需三个方法**（`reset` / `ingest` / `query`）走的是**线上同款**写入/读取路径（完整评测的唯一前提）；若降级为读取层评测：已由用户确认口径、元数据（`--adapter-meta`）声明、A/B 同口径；
- [ ] 声明了快照组能力时四个方法（`snapshot_capabilities` / `snapshot` / `restore` / `rebuild`）都已实现，且 `restore` 只接受当前格式版本、`rebuild` 覆盖 `compatible_from` 列出的旧版本；
- [ ] `describe()` 已返回结构化配置（嵌入模型 / 检索方式 / top_k / 索引类型等）；不打算提供时保持默认 `{}` 即可（报告不出现该小节）；
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
