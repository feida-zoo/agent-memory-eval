/**
 * membench HTTP bridge 参考模板（Node，零第三方依赖）。
 *
 * 用途：被测系统的记忆能力是「库内能力」、没有独立服务（典型是 Electron 主进程
 * 里的 TS 源码）时，用本服务把三个动作暴露成 membench HttpAdapter 认识的 REST 端点：
 *
 *   POST /reset   -> 清空记忆
 *   POST /ingest  -> 写入一段对话，返回 {"input_tokens","llm_calls","latency_ms"}
 *   POST /query   -> 查一个问题，返回 {"answer","retrieved",...}
 *
 * 契约对齐：membench/adapters/http.py 与 doctor.check_target。
 * 模板默认监听 127.0.0.1:8787，可用 BRIDGE_PORT / BRIDGE_HOST 覆盖。
 */
import http from "node:http";
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

const memory = createMemorySystem({ dataDir: DATA_DIR });

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

const routes = new Map([
  [
    "/reset",
    async () => {
      await memory.reset();
      return {};
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
    // 端点内部异常也返回 JSON 体，方便 doctor 的 detail 直接看到原因。
    send(res, 500, { error: String(err?.message || err) });
  }
});

server.listen(PORT, HOST, () => {
  console.log(`membench http bridge 监听 http://${HOST}:${PORT}（dataDir=${DATA_DIR}）`);
});
