/**
 * membench HTTP bridge 参考模板（Node，零第三方依赖）。
 *
 * 用途：被测系统的记忆能力是「库内能力」、没有独立服务（典型是 Electron 主进程
 * 里的 TS 源码）时，用本服务把记忆动作暴露成 membench HttpAdapter 认识的 REST 端点。
 *
 * 必需端点（三个）：
 *
 *   POST /reset   -> 清空记忆（可选：清空前把记忆目录快照一份，见 BRIDGE_RESET_BACKUP）
 *   POST /ingest  -> 写入一段对话，返回 {"input_tokens","llm_calls","latency_ms"}
 *   POST /query   -> 查一个问题，返回 {"answer","retrieved",...}
 *
 * 可选端点（五个；不做 = 不支持，框架按「缺失即不支持 / 空」处理）：
 *
 *   POST /snapshot-capabilities -> 快照能力声明（supported/restore/rebuild/format_version）
 *   POST /snapshot              -> 导出当前状态到 target_dir
 *   POST /restore               -> 同格式版本恢复（只接受当前 format_version）
 *   POST /rebuild               -> 跨版本 / 重建派生结构（接受 compatible_from 的版本）
 *   POST /describe              -> 桥与记忆模块自描述（进 metrics.adapter_describe）
 *
 * 契约对齐：membench/adapters/http.py 与 doctor.check_target。
 * 模板默认监听 127.0.0.1:8787，可用 BRIDGE_PORT / BRIDGE_HOST 覆盖。
 */
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import process from "node:process";

// ============================================================================
// 挂点（接入方唯一必须改的地方）
//
// 把下面的入口换成被测系统真实的记忆模块。它需要默认导出（或具名导出）
// createMemorySystem(options)，返回一个对象：
//
//   reset()                       -> Promise<void> | void
//   ingest(messages)              -> Promise<{inputTokens?, llmCalls?}>
//   query(question)               -> Promise<{answer, retrieved?, inputTokens?,
//                                             outputTokens?, llmCalls?}>
//
// 可选：对象上的 describe()（返回 dict）与 version 会被 POST /describe 带进自描述，
// 原样落到 metrics.adapter_describe——不参与评分，能拿到就带上。
//
// messages 的每项形如 { role: "user"|"assistant", content: string, timestamp }。
// 若真实系统是 TS/ESM 源码，先由 build.mjs 用 esbuild 打成单文件；bridge 仍
// import 那个入口路径即可（打包产物是 CJS，Node 的 ESM 也能 import）。
// ============================================================================
import { createMemorySystem } from "./memory-entry.mjs";

const PORT = Number(process.env.BRIDGE_PORT || 8787);
const HOST = process.env.BRIDGE_HOST || "127.0.0.1";
const DATA_DIR = process.env.BRIDGE_DATA_DIR || "./.bridge-data";
// 清空前快照（默认关闭）："" = 不做；"run" = 落到本次对话目录的
// runtime/reset_backup/（对话目录由 membench 经 /reset 请求体的 run_dir 带过来）；
// 其它值 = 固定目录。
// 注意成本：reset 每个对话调一次，开启后等于每对话复制一份记忆目录——调试/复现时开，
// 长跑批量评测建议关。
const RESET_BACKUP = process.env.BRIDGE_RESET_BACKUP || "";

let memory = createMemorySystem({ dataDir: DATA_DIR });

/**
 * 清空前快照（BRIDGE_RESET_BACKUP，默认关闭，人类可读布局）：把当前记忆目录
 * 复制一份，便于评测后回溯"这一刻记忆里有什么"。目录名带 reset 序号与对话 id，
 * 与 results.jsonl 的 dialogue_id 直接对得上；`run` 模式落对话目录的
 * `runtime/reset_backup/`（请求体里 run_dir 就是该对话目录）。
 */
function snapshotBeforeReset(body) {
  if (!RESET_BACKUP) return null;
  const base =
    RESET_BACKUP === "run"
      ? body.run_dir
        ? path.join(String(body.run_dir), "runtime", "reset_backup")
        : null
      : RESET_BACKUP;
  if (!base) return null;
  const tag = body.dialogue_id ? String(body.dialogue_id) : "unknown";
  const dest = path.join(base, `reset_${String(resetSeq).padStart(4, "0")}_${tag}`);
  try {
    copyDir(DATA_DIR, dest);
    return dest;
  } catch (err) {
    console.error(`[bridge] 快照失败（不影响 reset）：${err?.message || err}`);
    return null;
  }
}

// 状态快照格式版本：改快照布局时递增。SNAPSHOT_COMPATIBLE_FROM 是**重建**窗口：
// 列出的旧版本仍可用当前代码读源数据重建（/rebuild）；/restore 只接受当前版本
// （直接放回状态，要求格式完全兼容）。窗口外的版本只能重新导出。当前只支持 [1]。
const SNAPSHOT_FORMAT_VERSION = 1;
const SNAPSHOT_COMPATIBLE_FROM = [1];
const SNAPSHOT_MARKER = ".membench-snapshot.json";
// 桥契约修订号（端点 / 行为变更时递增）与挂点入口文件名（改顶部 import 时同步）
// ——进 /describe 自描述。
const BRIDGE_VERSION = 1;
const MEMORY_MODULE_LABEL = "memory-entry.mjs";

let resetSeq = 0;

/** 带 HTTP 状态码的桥错误：4xx = 请求侧问题（快照不可用 / 版本不支持），5xx = 桥内部故障。 */
class BridgeError extends Error {
  constructor(message, status = 500) {
    super(message);
    this.status = status;
  }
}

/** 读取并解析 JSON 请求体（空体视为 {}）。 */
function readJson(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => {
      const raw = Buffer.concat(chunks).toString("utf8").trim();
      if (!raw) return resolve({});
      try {
        resolve(JSON.parse(raw));
      } catch {
        reject(new Error("请求体不是合法 JSON"));
      }
    });
    req.on("error", reject);
  });
}

function send(res, code, payload) {
  const body = JSON.stringify(payload ?? {});
  res.writeHead(code, { "Content-Type": "application/json; charset=utf-8" });
  res.end(body);
}

/** 取不到就填 0 / 空列表，保证响应字段类型稳定。 */
function num(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

/** 递归复制目录（零依赖实现；目标存在则先清掉）。 */
function copyDir(src, dest) {
  if (!fs.existsSync(src)) return false;
  fs.rmSync(dest, { recursive: true, force: true });
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.cpSync(src, dest, { recursive: true });
  return true;
}

/** 统计目录下的文件数（登记到快照标记里，便于人工核对）。 */
function countFiles(dir) {
  let n = 0;
  if (!fs.existsSync(dir)) return 0;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    n += entry.isDirectory() ? countFiles(full) : 1;
  }
  return n;
}

/**
 * 状态快照：把当前记忆目录整体导出到 target_dir（框架在"每个对话 ingest 完成、
 * query 之前"调用）。布局 = memory/（记忆本体）+ 标记文件（格式版本/来源/条目数）。
 */
function writeSnapshot(targetDir) {
  const abs = path.resolve(String(targetDir));
  fs.rmSync(abs, { recursive: true, force: true });
  fs.mkdirSync(abs, { recursive: true });
  const items = copyDir(DATA_DIR, path.join(abs, "memory"));
  const marker = {
    bridge: "http_bridge_node",
    format_version: SNAPSHOT_FORMAT_VERSION,
    created_at: new Date().toISOString(),
    items: items ? countFiles(path.join(abs, "memory")) : 0,
  };
  fs.writeFileSync(path.join(abs, SNAPSHOT_MARKER), JSON.stringify(marker, null, 2));
  return { path: abs, items: marker.items, format_version: SNAPSHOT_FORMAT_VERSION };
}

/**
 * 读取并校验快照标记：标记文件与 memory/ 目录缺一不可。
 * 返回 {abs, marker}；不合格抛 4xx——宁可拒绝恢复，也不能把记忆库换成空库。
 */
function readSnapshotMarker(sourceDir) {
  const abs = path.resolve(String(sourceDir));
  const markerPath = path.join(abs, SNAPSHOT_MARKER);
  if (!fs.existsSync(markerPath)) {
    throw new BridgeError(
      `快照缺少标记文件 ${SNAPSHOT_MARKER}（拒绝恢复，避免覆盖成空库）：${abs}`,
      400,
    );
  }
  let marker;
  try {
    marker = JSON.parse(fs.readFileSync(markerPath, "utf8"));
  } catch (err) {
    throw new BridgeError(
      `快照标记文件损坏（${markerPath}）：${err?.message || err}`,
      400,
    );
  }
  if (!fs.existsSync(path.join(abs, "memory"))) {
    throw new BridgeError(`快照缺少 memory/ 目录：${abs}`, 400);
  }
  return { abs, marker };
}

/** 标记里读不到版本时按 NaN 处理——不在任何窗口内，直接拒绝。 */
function snapshotVersion(marker) {
  return Number(marker?.format_version);
}

/**
 * 替换整个 DATA_DIR 并**重建记忆实例**（真实系统若在内存里缓存了索引/状态，
 * 必须在这里重新创建实例或触发它重新加载，否则恢复的只是磁盘、跑的还是旧内存）。
 */
function loadMemoryDir(snapshotAbs) {
  fs.rmSync(DATA_DIR, { recursive: true, force: true });
  fs.mkdirSync(path.dirname(path.resolve(DATA_DIR)), { recursive: true });
  fs.cpSync(path.join(snapshotAbs, "memory"), DATA_DIR, { recursive: true });
  memory = createMemorySystem({ dataDir: DATA_DIR });
}

/**
 * 从快照恢复（/restore）：只接受**当前**格式版本——格式不兼容时直接放回状态
 * 没有意义，那是 /rebuild 的职责。替换 DATA_DIR 并重建记忆实例。
 */
function restoreSnapshot(sourceDir) {
  const { abs, marker } = readSnapshotMarker(sourceDir);
  const version = snapshotVersion(marker);
  if (version !== SNAPSHOT_FORMAT_VERSION) {
    throw new BridgeError(
      `快照格式 v${marker?.format_version} 不是当前版本 v${SNAPSHOT_FORMAT_VERSION}，`
        + `/restore 只接受当前版本；跨版本请走 /rebuild：${abs}`,
      409,
    );
  }
  loadMemoryDir(abs);
  return { restored: true, from: abs, format_version: SNAPSHOT_FORMAT_VERSION };
}

/**
 * 从快照重建（/rebuild）：接受 SNAPSHOT_COMPATIBLE_FROM 列出的旧版本。
 *
 * 与 /restore 的差别有两层：
 *   1) 版本窗口：restore 只认当前格式版本；rebuild 还接受窗口内的旧版本，
 *      由当前代码重读旧版**源数据**；
 *   2) 派生结构：rebuild 只把源数据落回数据目录，索引 / 向量 / 摘要缓存由重建
 *      记忆实例时的构建路径重新生成——因此存储结构或 schema 变了也不必重跑
 *      LLM 抽取。
 *
 * 模板的 stub 记忆模块在构造时从 memory/ 加载状态并重建进程内索引，所以 memory/
 * 就是它的源数据。真实系统若把派生结构也落在 DATA_DIR 里，接入时先剔掉这些条目
 * （或改调系统自己的 build 入口）再重建实例——别把快照里的旧派生结构当源数据。
 */
function rebuildSnapshot(sourceDir) {
  const { abs, marker } = readSnapshotMarker(sourceDir);
  const version = snapshotVersion(marker);
  if (!SNAPSHOT_COMPATIBLE_FROM.includes(version)) {
    throw new BridgeError(
      `快照格式 v${marker?.format_version} 不在可重建窗口 `
        + `${JSON.stringify(SNAPSHOT_COMPATIBLE_FROM)} 内，无法重建：${abs}；`
        + `请用当前版本 v${SNAPSHOT_FORMAT_VERSION} 重新导出快照`,
      422,
    );
  }
  loadMemoryDir(abs);
  return {
    ok: true,
    mode: "rebuild",
    format_version: SNAPSHOT_FORMAT_VERSION,
    source_format_version: version,
  };
}

const routes = new Map([
  [
    "/reset",
    async (body) => {
      resetSeq += 1;
      const backup = snapshotBeforeReset(body);
      await memory.reset();
      const payload = { reset_seq: resetSeq };
      if (backup) payload.backup_path = backup;
      return payload;
    },
  ],
  [
    "/snapshot-capabilities",
    async () => ({
      supported: true,
      restore: true,
      rebuild: true,
      format_version: SNAPSHOT_FORMAT_VERSION,
      compatible_from: [...SNAPSHOT_COMPATIBLE_FROM],
    }),
  ],
  [
    "/snapshot",
    async (body) => writeSnapshot(body.target_dir),
  ],
  [
    "/restore",
    async (body) => restoreSnapshot(body.source_dir),
  ],
  [
    "/rebuild",
    async (body) => rebuildSnapshot(body.source_dir),
  ],
  [
    "/describe",
    async () => {
      // 桥自身的静态自描述 + 记忆模块能给出的一切（describe() / version）。
      // 这是可选信息：记忆模块没提供就只回静态字段，取不到不报错。
      const described = {
        bridge: "http_bridge_node",
        bridge_version: BRIDGE_VERSION,
        memory_module: MEMORY_MODULE_LABEL,
        format_version: SNAPSHOT_FORMAT_VERSION,
      };
      if (typeof memory.describe === "function") {
        try {
          const extra = await memory.describe();
          if (extra && typeof extra === "object" && !Array.isArray(extra)) {
            described.memory = extra;
          }
        } catch (err) {
          described.memory_describe_error = String(err?.message || err);
        }
      }
      if (memory.version !== undefined) {
        described.memory_version = memory.version;
      }
      return described;
    },
  ],
  [
    "/ingest",
    async (body) => {
      const messages = Array.isArray(body.messages) ? body.messages : [];
      const startedAt = Date.now();
      const stats = (await memory.ingest(messages)) || {};
      return {
        input_tokens: num(stats.inputTokens ?? stats.input_tokens),
        llm_calls: num(stats.llmCalls ?? stats.llm_calls),
        latency_ms: Date.now() - startedAt,
      };
    },
  ],
  [
    "/query",
    async (body) => {
      const question = String(body.question ?? "");
      const startedAt = Date.now();
      const result = (await memory.query(question)) || {};
      return {
        answer: String(result.answer ?? ""),
        retrieved: Array.isArray(result.retrieved)
          ? result.retrieved.map((item) => String(item))
          : [],
        latency_ms: Date.now() - startedAt,
        input_tokens: num(result.inputTokens ?? result.input_tokens),
        output_tokens: num(result.outputTokens ?? result.output_tokens),
        llm_calls: num(result.llmCalls ?? result.llm_calls),
      };
    },
  ],
]);

const server = http.createServer(async (req, res) => {
  let pathname = "/";
  try {
    pathname = new URL(req.url, `http://${req.headers.host || "localhost"}`).pathname;
  } catch {
    return send(res, 400, { error: "非法请求 URL" });
  }

  if (req.method !== "POST") {
    return send(res, 405, { error: "只支持 POST" });
  }
  const handler = routes.get(pathname);
  if (!handler) {
    return send(res, 404, { error: `未知端点：${pathname}` });
  }

  try {
    const body = await readJson(req);
    send(res, 200, await handler(body));
  } catch (err) {
    // 端点内部异常也返回 JSON 体，方便 doctor / HttpAdapter 的 detail 直接看到原因；
    // 快照不可用 / 版本不在窗口内这类请求侧问题按 BridgeError 的状态码返回 4xx，
    // 不静默降级（框架的计划校验依赖"明确拒绝"而不是"假装成功"）。
    const status = err instanceof BridgeError ? err.status : 500;
    send(res, status, { error: String(err?.message || err) });
  }
});

server.listen(PORT, HOST, () => {
  console.log(`membench http bridge 监听 http://${HOST}:${PORT}（dataDir=${DATA_DIR}）`);
});
