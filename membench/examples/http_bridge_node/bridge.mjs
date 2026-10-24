/**
 * membench HTTP bridge 参考模板（Node，零第三方依赖）。
 *
 * 用途：被测系统的记忆能力是「库内能力」、没有独立服务（典型是 Electron 主进程
 * 里的 TS 源码）时，用本服务把三个动作暴露成 membench HttpAdapter 认识的 REST 端点：
 *
 *   POST /reset   -> 清空记忆（可选：清空前把记忆目录快照一份，见 BRIDGE_RESET_BACKUP）
 *   POST /ingest  -> 写入一段对话，返回 {"input_tokens","llm_calls","latency_ms"}
 *   POST /query   -> 查一个问题，返回 {"answer","retrieved",...}
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

// 状态快照格式版本：改快照布局时递增，并在 capabilities 里声明可兼容的旧版本。
const SNAPSHOT_FORMAT_VERSION = 1;
const SNAPSHOT_MARKER = ".membench-snapshot.json";

let resetSeq = 0;

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
 * 从快照恢复：只认带标记文件的快照（防止误传路径把记忆库清空）。
 * 替换整个 DATA_DIR，并**重建记忆实例**（真实系统若在内存里缓存了索引/状态，
 * 必须在这里重新创建实例或触发它重新加载，否则恢复的只是磁盘、跑的还是旧内存）。
 */
function restoreSnapshot(sourceDir) {
  const abs = path.resolve(String(sourceDir));
  const markerPath = path.join(abs, SNAPSHOT_MARKER);
  if (!fs.existsSync(markerPath)) {
    throw new Error(`快照缺少标记文件 ${SNAPSHOT_MARKER}（拒绝恢复，避免覆盖成空库）：${abs}`);
  }
  const memDir = path.join(abs, "memory");
  if (!fs.existsSync(memDir)) {
    throw new Error(`快照缺少 memory/ 目录：${abs}`);
  }
  fs.rmSync(DATA_DIR, { recursive: true, force: true });
  fs.mkdirSync(path.dirname(path.resolve(DATA_DIR)), { recursive: true });
  fs.cpSync(memDir, DATA_DIR, { recursive: true });
  memory = createMemorySystem({ dataDir: DATA_DIR });
  return { restored: true, from: abs };
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
      rebuild: false,
      format_version: SNAPSHOT_FORMAT_VERSION,
      compatible_from: [SNAPSHOT_FORMAT_VERSION],
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
    // 端点内部异常也返回 JSON 体，方便 doctor 的 detail 直接看到原因。
    send(res, 500, { error: String(err?.message || err) });
  }
});

server.listen(PORT, HOST, () => {
  console.log(`membench http bridge 监听 http://${HOST}:${PORT}（dataDir=${DATA_DIR}）`);
});
