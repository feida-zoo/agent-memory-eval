/**
 * electron 的最小 stub。
 *
 * 被测系统跑在 Electron 主进程里，记忆模块常直接
 * `import { app, ipcMain } from "electron"` 拿用户数据目录 / 注册 IPC。
 * 评测环境没有 Electron 运行时，build.mjs 用 alias 把 "electron" 指到本文件，
 * 只实现记忆模块真正会用到的那几个成员。
 */
import os from "node:os";
import path from "node:path";

const dataDir =
  process.env.BRIDGE_DATA_DIR || path.join(os.tmpdir(), "membench-electron");

/** app.getPath 是记忆模块定位 SQLite/向量库目录的最常见入口。 */
export const app = {
  getName: () => "membench-bridge",
  getVersion: () => "0.0.0",
  isPackaged: false,
  getPath: (name) => {
    if (name === "userData") return dataDir;
    if (name === "temp") return os.tmpdir();
    if (name === "logs") return path.join(dataDir, "logs");
    if (name === "home") return os.homedir();
    return path.join(dataDir, String(name));
  },
  whenReady: () => Promise.resolve(),
  quit: () => {},
};

/** IPC 在评测里无对端，注册即空操作（避免源码里的 ipcMain.handle 抛错）。 */
export const ipcMain = {
  handle: () => {},
  handleOnce: () => {},
  on: () => {},
  once: () => {},
  removeHandler: () => {},
  emit: () => false,
};

/** 少数源码会在主进程里遍历窗口广播，这里返回空列表。 */
export const BrowserWindow = {
  getAllWindows: () => [],
};

export default { app, ipcMain, BrowserWindow };
