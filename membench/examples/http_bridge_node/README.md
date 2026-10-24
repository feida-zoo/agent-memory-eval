# http_bridge_node：Node / Electron 系统的 HTTP bridge 参考模板

被测系统的记忆能力是**库内能力**、没有独立服务（典型是 Electron 主进程里的 TS 源码）时，
用本目录把三个动作暴露成 membench 认识的 REST 端点，再用内置 `HttpAdapter` 跑评测。
协议层（三端点契约 + HttpAdapter）已经够用，本目录补的是**与协议无关的工程脚手架**：
单文件打包、宿主运行时 stub、原生依赖处理、自检。

> 只交付可读模板，不预装依赖。`node_modules/` 由你按需 `npm i` 生成（唯一需要的是 esbuild）。

## 目录

| 文件 | 作用 |
|---|---|
| `bridge.mjs` | `node:http` 服务，暴露 `POST /reset\|/ingest\|/query`；顶部「挂点」import 你的记忆模块 |
| `build.mjs` | esbuild 打包脚本：真实源码 → 单文件 CJS，处理 stub / external / define |
| `stubs/electron.mjs` | `electron` 最小 stub（`app.getPath` / `ipcMain` / `BrowserWindow`） |
| `stubs/i18n.mjs` | `i18n` 最小 stub（`t()` 原样回退，可注入真实文案） |
| `package.json` | `scripts.build` / `scripts.start`；`devDependencies` 只列 esbuild |

## 从零到跑通

```bash
cd membench/examples/http_bridge_node

# 1) 接上真实记忆模块
#    - 写好 ./src/memory-entry.ts（默认导出 createMemorySystem({ dataDir })），或
#    - 直接改 bridge.mjs 顶部「挂点」那行 import 指向你已有的入口。
#    createMemorySystem() 需返回 { reset, ingest, query }（见 bridge.mjs 注释）。

# 2) 打包（按需先装 esbuild）
npm i                 # 生成 node_modules，只装 esbuild
npm run build         # = node build.mjs，产物 dist/memory-entry.cjs

# 3) 起服务
npm start             # = node bridge.mjs，默认 http://127.0.0.1:8787
#   可用环境变量覆盖：BRIDGE_HOST / BRIDGE_PORT / BRIDGE_DATA_DIR
#   可选：清空前快照（调试/回溯用）——BRIDGE_RESET_BACKUP=run 时每次
#   /reset 前把记忆目录快照到本次对话目录的 runtime/reset_backup/
#   （目录名带 reset 序号与 dialogue_id，与 results.jsonl 对得上）；
#   设成具体目录则快照到该目录。默认关闭（每个对话调一次 reset，长跑别开）。
#   另有一组快照回放端点（无需环境变量）：POST /snapshot-capabilities|/snapshot|/restore，
#   配合 membench run --snapshot / --replay 使用（改读取侧代码免重跑 ingest）。

# 4) 自检（另开一个终端）
cd <仓库根>/membench
python -m membench.cli doctor --target http://127.0.0.1:8787
#   ok 时退出码 0 且 reset/ingest/query 全 PASS；FAIL 的 detail 直接指出问题。
#   需要结构化结果时也可用 Python API：
#   python -c "import json; from membench.doctor import check_target; print(json.dumps(check_target('http://127.0.0.1:8787'), ensure_ascii=False, indent=2))"

# 5) 用 HttpAdapter 跑评测：内置类的 base_url 是构造参数，先用持久化适配器把桥地址绑上去
cat > /tmp/my-bridge-adapter.py <<'EOF'
from membench.adapters.http import HttpAdapter


def register(registry):
    registry["my-bridge"] = lambda: HttpAdapter("http://127.0.0.1:8787")
EOF
python -m membench.cli adapter install /tmp/my-bridge-adapter.py --name my-bridge

# --report-out 是实验目录的父目录
python -m membench.cli run --adapter my-bridge --judge heuristic \
  --dataset membench/data/office_dataset.jsonl --report-out ../output/
# 产物: ../output/eval_http_default_<ts>/      （实验目录名取适配器的 name 属性 = http）
#   ├── runs/office_dataset/report/{report.md, metrics.json, results.jsonl, fig_*.png}
#   ├── runs/office_dataset/<对话目录>/{report/, runtime/}   # 对话级（report/ 不出图）
#   └── reports/{report.md, metrics.json, fig_*.png}         # 任务级
```

`check_target` 的判据：`/reset` 与 `/ingest` 2xx 即 PASS（`/ingest` 若返回 JSON 再校验
`input_tokens`/`llm_calls` 为数值）；`/query` 必须 2xx + JSON dict + 字符串 `answer`，
`retrieved` 若出现必须是字符串列表。`api_key` 非空时会带 `Authorization: Bearer <key>`。

## 常见坑

- **原生依赖必须 external**：`better-sqlite3` / `sqlite-vec` / `node-llama-cpp` 带 `.node`
  二进制，打进单文件会报 `Cannot find module ... .node`。`build.mjs` 已 `external`；
  运行时要能从 **bridge 自身** 的 `node_modules` 解析它们——即 `npm i` 要在本目录做，
  且版本与真实产品一致（不同版本 ABI/行为可能不同）。
- **宿主运行时必须 stub**：真实源码里 `import { app } from "electron"`、`import { t } from "i18n"`
  在评测环境不存在。用 `alias` 把它们指向 `stubs/`；`app.getPath("userData")` 决定记忆库落盘目录，
  务必让它落在评测用临时目录（`BRIDGE_DATA_DIR`），**绝不指向真实用户数据**。
- **构建期常量要注入**：`__DEV__` / `__PRODUCT__` 这类标识符不在源码里，缺了会
  `ReferenceError: __DEV__ is not defined`。用 `define` 注入；字符串值要写成 `'"..."'`。
- **路径与解析**：`alias` 用的是**打包期绝对路径**（本模板用 `import.meta.url` 推导），
  换成相对路径会在其它 cwd 下失效；`external` 的模块在运行时解析，所以别把它们也 alias 掉。
- **memory-entry 是挂点**：模板默认 `import "./memory-entry.mjs"` 指向一个不存在的占位文件，
  这是故意的——接上真实入口前 `node bridge.mjs` 起不来，`node --check bridge.mjs` 仍能过语法。
- **ingest 的 messages 要带 timestamp**：`temporal` 类题（"昨天/下周"）依赖时间锚，
  丢时间戳会系统性失败。
- **并行分片要起多个实例**：一个桥进程只服务一个分片——评测对每个 dialogue 调
  `reset()`，两片共享一个实例会互相清库（跨片串扰）。N 分片并行 = N 个进程，
  各自 `BRIDGE_PORT` + `BRIDGE_DATA_DIR`；单实例时只能串行跑分片。
- **query 要真的走检索**：别把问题直接丢给底座模型。`retrieved` 填**实际注入 prompt 的
  记忆原文**，否则 `evidence_recall` 会暴露"没走记忆路径"。
- **回溯记忆内部状态用 `BRIDGE_RESET_BACKUP`**：它把"清空"变成"先快照再清空"。框架本身
  不读不写你的记忆存储（黑盒契约），只通过 `/reset` 请求体告诉你本次对话目录、对话 id
  与序号；要不要快照、落到哪里，由桥决定（模板 `run` 模式落到 `<对话目录>/runtime/reset_backup/`）。
  逐题 `retrieved` 已在 `results.jsonl` 里，快照只用来查"内部存储为什么这样"，
  别在所有长跑里常开（每对话一份 = 磁盘翻倍）。
- **快照回放端点**：`/snapshot` 把记忆整体导出到 `target_dir`（模板布局 = `memory/` +
  `.membench-snapshot.json` 标记，框架还会在同目录写清单 `manifest.json`），`/restore` 替换
  `DATA_DIR` 并**重建记忆实例**（真实系统若在内存缓存索引，必须在这里重新创建实例，
  否则恢复的只是磁盘）；`/restore` 只认带标记的快照目录，防止路径传错把库清空。用法：
  `run --snapshot`（每对话 ingest 后导出到该对话的 `memory_snapshot/`）→ 改读取侧代码 →
  `run --replay <实验目录或数据集目录> --replay-verify-source <源数据集目录>`
  （回放跳过 reset + ingest，写入侧固定，结论只覆盖读取/回答层）。
- **别用 express 等额外框架**：三端点用 `node:http` 足够，少一个依赖少一类 native/构建坑。

对应 checklist（打包 / stub / external / define / 契约字段 / doctor 用法）见
`skill/src/references/http_bridge_guide.md`。
