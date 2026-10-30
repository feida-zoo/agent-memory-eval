"""membench 自身的版本标识：包语义版本 + 代码哈希。

评测报告必须能回答"这个分数是哪一版评测代码跑出来的"——判分口径（拒答词表、
rubric 近失复核、归因顺序）一变，同一份回答的分数就变，光有被测系统的
``commit`` 无法解释历史分数漂移。

哈希的定稿时机在**打包时**：安装后的 skill 包没有 ``.git``，而且用户拿到的是
一个压缩包，事后无法追溯"这份代码对应哪个提交"。因此 ``skill/build_skill.py``
在打包前算出代码哈希、写进 :mod:`membench._build_hash`（该文件自身不参与哈希
计算，避免自引用），运行时 :func:`runner_version` 直接读常量带回报告。

开发态（还没打包过 / 打包后改过代码）读到的是空的或过期的常量，此时按当前代码
现算并用 ``hash_source`` 标明，绝不把过期的打包哈希当成事实。
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from typing import Union

try:  # 正常导入：包内相对导入
    from . import _build_hash as _build_hash_mod
except ImportError:  # 打包脚本按文件路径独立加载本模块时没有包上下文
    _build_hash_mod = None  # type: ignore[assignment]

__all__ = [
    "BUILD_HASH_FILE",
    "__version__",
    "hash_state",
    "package_hash",
    "runner_version",
    "version_display",
]

#: 包语义版本（与 ``membench/pyproject.toml`` 的 ``version`` 保持一致）。
__version__ = "0.1.0"

#: 内容指纹取 sha256 的前多少位十六进制（够短可读，改动一行也几乎不会撞）。
PACKAGE_HASH_CHARS = 12

#: 打包时写入哈希的文件名（相对包目录）。**必须**排除在哈希计算之外。
BUILD_HASH_FILE = "_build_hash.py"


def _hashable_files(package_dir: str) -> list[tuple[str, str]]:
    """参与哈希的文件：包内全部 ``*.py``，排除构建哈希文件与 ``__pycache__``。

    Returns:
        ``[(相对路径(``/`` 分隔), 绝对路径), ...]``，按相对路径排序（与文件系统
        遍历顺序无关，保证同一份代码在任何机器上哈希一致）。
    """
    pairs: list[tuple[str, str]] = []
    for root, dirs, files in os.walk(package_dir):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(files):
            if not name.endswith(".py") or name == BUILD_HASH_FILE:
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, package_dir).replace(os.sep, "/")
            pairs.append((rel, path))
    pairs.sort()
    return pairs


def compute_hash(package_dir: Union[str, None] = None) -> str:
    """现算包内代码的内容指纹（sha256 前 :data:`PACKAGE_HASH_CHARS` 位）。

    按"相对路径 + 文件字节"累加；读不到的文件跳过而不是抛错——版本标识永远
    不能成为跑评测的阻塞项。

    Args:
        package_dir: 包目录；缺省为本模块所在目录（测试里可指向夹具目录）。
    """
    if package_dir is None:
        package_dir = os.path.dirname(os.path.abspath(__file__))
    hasher = hashlib.sha256()
    for rel, path in _hashable_files(package_dir):
        try:
            with open(path, "rb") as handle:
                content = handle.read()
        except OSError:
            continue
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(content)
        hasher.update(b"\x00")
    return hasher.hexdigest()[:PACKAGE_HASH_CHARS]


@lru_cache(maxsize=1)
def hash_state() -> dict[str, Union[str, bool, None]]:
    """哈希及其来源。

    Returns:
        ``{"hash": 现行哈希, "source": "build"|"computed",
        "build_hash": 打包时写入的哈希或 None, "stale_build": 打包哈希是否已过期}``。

    判定：打包哈希存在且与现算一致 → ``source="build"``（正常安装态）；
    打包哈希存在但**不一致** → 代码在打包后被改过，``source="computed"`` 且
    ``stale_build=True``（报告据此提示重新打包）；没有打包哈希（开发态）→
    ``source="computed"``、``stale_build=False``。
    """
    built = str(getattr(_build_hash_mod, "BUILD_HASH", "") or "").strip()
    current = compute_hash()
    if built and built == current:
        return {
            "hash": current,
            "source": "build",
            "build_hash": built,
            "stale_build": False,
        }
    return {
        "hash": current,
        "source": "computed",
        "build_hash": built or None,
        "stale_build": bool(built),
    }


def package_hash() -> str:
    """现行代码哈希（同 :func:`hash_state` 的 ``hash``）。"""
    return str(hash_state()["hash"])


def runner_version() -> dict[str, Union[str, bool, None]]:
    """评测代码的版本标识，落进 ``metrics.json`` 与报告头部。

    Returns:
        ``{"version": 语义版本, "hash": 代码哈希, "hash_source": "build"|"computed",
        "stale_build": 打包哈希是否过期}``。
    """
    state = hash_state()
    return {
        "version": __version__,
        "hash": state["hash"],
        "hash_source": state["source"],
        "stale_build": bool(state["stale_build"]),
    }


def version_display(value: object) -> str:
    """把 :func:`runner_version` 的返回值渲染成一行可读文本（报告表格用）。

    形如 ``0.1.0+3f9c2a1b4d5e（构建哈希）``；开发态（现算）与打包后改动过代码
    会如实标注，不把过期哈希当事实。字段缺失/类型不对时写 ``n/a``，不猜。
    """
    if not isinstance(value, dict):
        return "n/a"
    version = value.get("version")
    hash_value = value.get("hash")
    if not version and not hash_value:
        return "n/a"
    text = f"{version}+{hash_value}" if hash_value else str(version)
    source = value.get("hash_source")
    if source == "build":
        text += "（构建哈希）"
    elif value.get("stale_build"):
        text += "（当前代码指纹；打包哈希已过期，请重新打包 skill）"
    else:
        text += "（当前代码指纹；未打包）"
    return text
