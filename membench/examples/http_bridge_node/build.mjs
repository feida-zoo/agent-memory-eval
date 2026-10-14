/**
 * esbuild 打包示例：把被测系统的真实业务源码打成单文件 CJS，供 bridge 加载。
 *
 * 用法：
 *   npm i                 # 按需安装 esbuild（模板不预装依赖）
 *   node build.mjs [entry]  # 默认 entry 为 ./src/memory-entry.ts
 *
 * 关键点（对应 README 的常见坑）：
 *   - platform=node + format=cjs：Node 直接可跑，也可被 ESM 的 bridge import；
 *   - alias：把宿主运行时模块（electron / i18n）替成 stubs/ 下的最小实现，
 *     真实源码里 `import { app } from "electron"` 才不会在评测环境炸掉；
 *   - external：原生依赖（better-sqlite3 / sqlite-vec / node-llama-cpp）带 .node
 *     二进制，不能打进单文件，必须 external，运行时从 bridge 自身 node_modules 解析；
 *   - define：注入构建期常量（__DEV__ / __PRODUCT__），真实源码里通常有分支依赖它。
 */
import path from "node:path";
import { fileURLToPath } from "node:url";

import { build } from "esbuild";

const here = path.dirname(fileURLToPath(import.meta.url));
const entry = process.argv[2] || path.join(here, "src", "memory-entry.ts");

await build({
  entryPoints: [entry],
  outfile: path.join(here, "dist", "memory-entry.cjs"),
  bundle: true,
  platform: "node",
  format: "cjs",
  target: "node20",
  // 宿主运行时 stub：评测环境没有 Electron / 宿主语言包
  alias: {
    electron: path.join(here, "stubs", "electron.mjs"),
    i18n: path.join(here, "stubs", "i18n.mjs"),
  },
  // 原生依赖保持 external（含 .node），由运行时的 node_modules 解析
  external: ["better-sqlite3", "sqlite-vec", "node-llama-cpp"],
  // 构建期常量注入（值需是合法 JS 表达式：字符串要带引号）
  define: {
    __DEV__: "false",
    __PRODUCT__: '"membench-bridge"',
  },
  sourcemap: true,
  logLevel: "info",
});

// 补充：若真实源码用了 esbuild 无法从 import 图发现的全局标识符（如宿主注入的
// 全局单例），用 `inject: ["./stubs/inject-globals.mjs"]` 在打包时补齐，语义等同
// CLI 的 --inject。本模板未演示该文件，接入时按需添加。
