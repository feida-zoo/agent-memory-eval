/**
 * i18n 的最小 stub。
 *
 * 被测系统常用宿主 i18n 拼用户可见文案（例如"我没有相关记忆"）。评测环境没有
 * 宿主语言包，build.mjs 用 alias 把 "i18n" 指到本文件：t() 默认原样返回 key，
 * 接入方可用 setTranslations 覆盖成真实文案，避免把文案差异误当成记忆能力差异。
 */

const translations = new Map();

/** 取文案；未注册时回退为 key 本身（保证 t() 永不抛错）。 */
export function t(key, ...args) {
  const template = translations.get(key);
  if (typeof template === "function") return template(...args);
  if (typeof template === "string") return template;
  return typeof key === "string" ? key : String(key);
}

export function setTranslations(map) {
  for (const [key, value] of Object.entries(map || {})) {
    translations.set(key, value);
  }
}

export function getLocale() {
  return process.env.BRIDGE_LOCALE || "zh-CN";
}

export function setLocale() {}

export default { t, setTranslations, getLocale, setLocale };
