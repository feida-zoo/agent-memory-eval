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

# 4) 自检（另开一个终端）
cd <仓库根>/membench
python -m membench.cli doctor --target http://127.0.0.1:8787
#   ok 时退出码 0 且 reset/ingest/query 全 PASS；FAIL 的 detail 直接指出问题。
#   需要结构化结果时也可用 Python API：
#   python -c "import json; from membench.doctor import check_target; print(json.dumps(check_target('http://127.0.0.1:8787'), ensure_ascii=False, indent=2))"

# 5) 用 HttpAdapter 跑评测（--report-out 是实验目录的父目录）
python -m membench.cli run --adapter http --judge heuristic \
  --dataset membench/data/office_dataset.jsonl --report-out ../output/
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
- **query 要真的走检索**：别把问题直接丢给底座模型。`retrieved` 填**实际注入 prompt 的
  记忆原文**，否则 `evidence_recall` 会暴露"没走记忆路径"。
- **别用 express 等额外框架**：三端点用 `node:http` 足够，少一个依赖少一类 native/构建坑。

对应 checklist（打包 / stub / external / define / 契约字段 / doctor 用法）见
`skill/src/references/http_bridge_guide.md`。
