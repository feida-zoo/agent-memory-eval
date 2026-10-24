# 非 Python 系统接入 checklist（HTTP bridge）

适用于被测 Agent 的源码**不是 Python**、且记忆能力是**库内能力、没有独立服务**的情况
（典型：Electron 主进程的 TS 记忆模块、Node/Electron 桌面应用）。
做法是写一个极薄的 **HTTP bridge** 把库内能力暴露成三端点，再用 membench 内置
`HttpAdapter` 跑评测。跑之前用 `doctor.check_target` 自检契约。

要点：**协议只有三个端点**，桥的难点全在"与协议无关的工程问题"（打包、stub 宿主、原生依赖、常量）。
可直接复制的模板见 [`membench/examples/http_bridge_node/`](../../membench/examples/http_bridge_node/)。

## 三端点契约（bridge 必须实现）

| 端点 | 请求体 | 响应（最少） | 说明 |
|---|---|---|---|
| `POST /reset` | `{}` 或 `{"run_dir","dialogue_id","dialogue_index"}`（可选提示；`run_dir` = **该对话的目录** `<实验目录>/runs/<dataset>/<对话目录>`） | 2xx 即可（体可空） | 真清空记忆库（向量 collection / 事实表 / 情景缓冲 / 图谱），测试库与生产库隔离；上下文只用于"清空前快照"等回溯能力，可忽略 |
| `POST /ingest` | `{"messages":[{"role","content","timestamp"}]}` | `{"input_tokens":int,"llm_calls":int,"latency_ms":float}` | 走**记忆写入路径**；拿不到 token/调用数填 `0`（报告标「未上报」） |
| `POST /query` | `{"question":str}` | `{"answer":str,"retrieved":[str],...}` | 走**记忆读取+回答**；`retrieved` 填实际注入 prompt 的**记忆原文**，注入即口径（优先级顺序、多轮检索先合并去重、报全量，见下方常见坑）；`input_tokens`/`output_tokens` 填了才会进「Token 消耗」账 |
| `POST /snapshot-capabilities`（可选） | `{}` | `{"supported":true,"restore":true,"rebuild":false,"format_version":1}` | 声明快照能力；老桥缺该端点 → 框架按不支持处理 |
| `POST /snapshot`（可选） | `{"target_dir","tag"}` | `{"items":int}` | 把当前记忆整体导出到 target_dir（框架传的是该对话的 `memory_snapshot/`；**内容与布局由桥决定**，模板 = `memory/` + 标记文件） |
| `POST /restore`（可选） | `{"source_dir","tag"}` | `{"restored":true}` | 从快照恢复；**只认带标记文件的快照**，并重建进程内实例（只换磁盘不重建缓存等于没恢复） |

- `Content-Type: application/json`；如果服务要求鉴权，客户端用 `api_key` 时发
  `Authorization: Bearer <key>`，bridge 侧做对应校验即可。
- `latency_ms` 可由 bridge 实测，也可留空——`HttpAdapter` 本地用 `perf_counter` 兜底。
- 契约源码以 `membench/membench/adapters/http.py` 为准；自检判据见下节。

## checklist

### 1. 打包（esbuild / vite）

- 把**真实业务源码 entry** 打成单文件：`platform=node`、`format=cjs`、`bundle=true`。
  浏览器向的默认配置（`platform=browser`、`format=esm`）会引入 `window`/`process` 假设，主进程跑不通。
- bridge 自身用 `node:http`，**不引 express**——少一个依赖少一类构建、native 坑。
- 只交付可读模板时不要真的联网装依赖；本机有 `node` 时用 `node --check <file>.mjs` 过语法。

### 2. 宿主运行时 stub

真实源码会 `import { app } from "electron"`（拿数据目录）、`import { t } from "i18n"`（拼文案），
评测环境没有宿主。用打包器 **`alias`** 把它们替成 `stubs/*.mjs`：

- `electron`：至少实现 `app.getPath(name)`（`userData`/`temp`/`logs`/`home`）、`ipcMain.handle/on`（空操作）、
  `app.whenReady()`（返回 resolved Promise）。
- `i18n`：`t(key)` 原样回退，或注入真实文案——**别让文案差异污染记忆能力评分**。
- `app.getPath("userData")` 决定记忆库落盘目录：**必须**指向评测临时目录（模板用 `BRIDGE_DATA_DIR`），
  绝不碰真实用户数据。

若源码依赖**无法从 import 图发现的全局标识符**（宿主注入的单例等），用打包器 `inject`
（CLI `--inject`）在打包时补齐。

### 3. 原生依赖 external

`better-sqlite3` / `sqlite-vec` / `node-llama-cpp` 等带 `.node` 二进制，打进单文件会
`Cannot find module '.../x.node'`。一律 **external**：

- 打包侧：`--external:better-sqlite3 --external:sqlite-vec --external:node-llama-cpp`。
- 运行时：这些模块从 **bridge 自身** 的 `node_modules` 解析，所以 `npm i` 要在 bridge 目录做，
  且**版本与真实产品一致**（ABI/行为差异会导致结果不可比）。

### 4. 构建期常量注入

`__DEV__` / `__PRODUCT__` 之类由构建器注入的标识符不在源码里，缺了会 `ReferenceError`。
打包侧用 `--define:__DEV__=false --define:__PRODUCT__=\"...\"`（字符串值要带引号）。

### 5. 契约字段对齐（最容易接错的地方）

- `/ingest` 的 `messages` 逐条透传，**保留 `timestamp`**——temporal 题依赖时间锚。
- `/query` 必须真的走检索；`retrieved` 填实际注入 prompt 的记忆原文（`evidence_recall` 的判据）。
  把问题直接丢给底座模型会让分数虚高、证据召回暴露。
  **口径 = 注入即口径**：① 按注入优先级顺序排列（`recall@5` 只看前 5 条）；② 多轮检索先合并去重
  （重复条目白占前 5 窗口）；③ 报注入的那一份全量（只报 top-k 会让 `evidence_recall` 虚低，
  报未经注入裁剪的内部 top-N 会虚高）；④ 在构造 prompt 的那一处直接记下列表，别事后另算。
- 拿不到 token / 调用数就填 `0`，不要瞎填——成本指标失真比缺省更糟。
- **快照回放要"真恢复"**：`/restore` 必须把记忆库换回快照那一刻（含内存里的索引/缓存），
  并拒绝不认识的目录（防误传路径把库清空）；回放模式下框架不再调 `/reset`、`/ingest`。
- `reset` 只清会话不清记忆库 → 跨 dialogue 串扰，`single_hop` 异常高、`abstention` 异常低。

### 6. 自检：`doctor.check_target`

```python
from membench.doctor import check_target

result = check_target("http://127.0.0.1:8787", timeout=10.0, api_key=None)
# result == {"ok": bool, "base_url": str,
#            "checks": [{"name","status","detail"}, ...]}   # 顺序 reset/ingest/query
```

- 顺序探测三端点：`/reset`、`/ingest` 2xx 即 PASS（`/ingest` 若返回 JSON dict，额外校验
  `input_tokens`/`llm_calls` 为数值，缺失可接受）；`/query` 须 2xx + JSON dict + 字符串 `answer`，
  `retrieved` 若出现（且非 null）必须是字符串列表。
- **单端点失败不抛异常**：网络错误/超时/非 2xx/非 JSON 都记 `FAIL`，`detail` 可直接定位
  （`HTTP 404` / `响应非 JSON` / `缺少 answer 字段` / `retrieved 必须是字符串列表`）。
- `/reset` 失败**不短路**后续端点（接入方常把 reset 做成可选），故 `SKIP` 当前不产出。
- `ok=True` 表示没有任何 check FAIL；只要有一项 FAIL 就别急着跑评测，先修契约。
- 只用标准库 `urllib`，最小环境也能跑。

### 7. 跑评测（内置 HttpAdapter 直连）

`HttpAdapter` 的 `base_url` 是构造参数，用通用参数 `--adapter-arg` 传（不需要再包一层适配套件）：

```bash
# 建议先 doctor 自检，再跑评测；--report-out 是实验目录的父目录
python -m membench.cli run --adapter http \
  --adapter-arg base_url=http://127.0.0.1:8787 --adapter-arg timeout=10 \
  --dataset membench/data/office_dataset.jsonl --judge heuristic --report-out <实验目录父目录>
# 接受的构造参数：base_url（必填）、api_key、timeout；缺 base_url / key 写错会在开跑前报错
```

并行分片时各分片可共用同一个 `--report-out`——实验目录名末尾带分片序号 `_s<i>`
（`--limit` 另带 `_l<N>`），不会互相覆盖；同片重跑落同名目录仍会被非空守卫拒绝。

### 8. 与模板的对应关系

| checklist 项 | 模板位置 |
|---|---|
| 三端点 + 挂点 | `examples/http_bridge_node/bridge.mjs`（顶部 import 为挂点） |
| 打包 / alias / external / define | `examples/http_bridge_node/build.mjs` |
| 宿主 stub | `examples/http_bridge_node/stubs/electron.mjs`、`stubs/i18n.mjs` |
| 依赖与脚本 | `examples/http_bridge_node/package.json` |
| 从零到跑通 + 常见坑 | `examples/http_bridge_node/README.md` |

## 挂点提示（"记忆是库内能力、无独立服务"的系统）

- 真实系统里，记忆通常是**一类模块**而非进程：写入发生在对话结束的抽取/总结逻辑里，
  读取在回答生成前。bridge 要做的是**直接调这两个函数**，不要另起一份"平行记忆"实现。
- 若系统只在 UI 事件里触发记忆写入，注意 bridge 调用的是**纯函数/服务层**，绕开 UI 依赖
  （stub 掉 `ipcMain`/`BrowserWindow` 即可），否则会因缺窗口上下文而失败。
- 评测用独立数据目录（`BRIDGE_DATA_DIR`）与独立实例：`reset` 必须真清空，且**绝不能碰生产数据**。
- **并行分片 = 每分片一个桥实例**：评测对每个 dialogue 调 `reset()`，两个分片共享一个实例会互相清库（跨片串扰）。N 分片并行就起 N 个桥（各自 `BRIDGE_PORT` + `BRIDGE_DATA_DIR`）；只有一个实例时请串行跑分片。被测系统若自带多命名空间隔离，可自行在桥上按命名空间路由——那是系统能力增强，不是本模板的默认行为。分片评测的输出目录不必各自指定 `--report-out`：实验目录名带 `_s<i>` 后缀，各片（乃至同秒起跑的 N 片）可共用同一个父目录。
- **清空前快照（可选，回溯用）**：框架在每次 `reset` 前把上下文放进请求体（`run_dir` / `dialogue_id` / `dialogue_index`），**不读不写**你的记忆存储。`run_dir` 现在指向**该对话的目录**（`<实验目录>/runs/<dataset>/<对话目录>`，框架已把它的 `runtime/` 建好）：要"清空前留一份"就在桥里自己做，约定落 `runtime/reset_backup/reset_<序号>_<dialogue_id>/`，**默认关闭**。逐题 `retrieved` 已在 `results.jsonl`，快照只用于查"内部存储为什么长这样"（stale / 误改记忆 / 该记没记）；reset 每对话一次，长跑常开等于每对话复制一份记忆目录。
