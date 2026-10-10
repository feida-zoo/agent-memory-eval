"""用户测试集管理：install / status / delete / resolve。

用户测试集存放于用户测试集目录（默认 ``./user_dataset/``，可用环境变量
``MEMBENCH_USER_DATA_DIR`` 覆盖），目录内包含：

- ``user_dataset.jsonl``：校验通过后复制的数据集本体；
- ``metadata.json``：安装时间、来源路径、dialogue/question 统计。

测试集选择优先级（:func:`resolve_dataset`）：

1. 显式指定的路径（``--dataset``）-> ``"user-specified"``；
2. 已安装的用户测试集 -> ``"user"``；
3. 内置 office_dataset.jsonl -> ``"builtin-office"``。
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from membench.data.schema import load_dataset

#: 用户测试集文件名（位于用户测试集目录内）。
USER_DATASET_FILENAME = "user_dataset.jsonl"

#: 元数据文件名（位于用户测试集目录内）。
METADATA_FILENAME = "metadata.json"


def builtin_office_dataset_path() -> Path:
    """内置 office 黄金集的绝对路径。"""
    return Path(__file__).resolve().parent / "data" / "office_dataset.jsonl"


def user_data_dir() -> Path:
    """用户测试集目录。

    优先取环境变量 ``MEMBENCH_USER_DATA_DIR``，否则取当前工作目录下的
    ``./user_dataset/``。
    """
    env = os.environ.get("MEMBENCH_USER_DATA_DIR")
    if env:
        return Path(env)
    return Path.cwd() / "user_dataset"


def user_dataset_path() -> Path:
    """用户测试集 JSONL 的完整路径。"""
    return user_data_dir() / USER_DATASET_FILENAME


def install_user_dataset(src_path: str) -> dict:
    """校验并安装用户测试集。

    先用 :func:`membench.data.schema.load_dataset` 完整校验源文件；校验通过
    才复制到 ``user_data_dir()/user_dataset.jsonl`` 并写 ``metadata.json``。
    已存在则覆盖（是否确认由调用方负责）。校验失败抛异常且不写入任何文件。

    Args:
        src_path: 源 JSONL 数据集路径。

    Returns:
        统计 dict：``{"path", "source", "installed_at", "n_dialogues", "n_questions"}``。

    Raises:
        FileNotFoundError: 源文件不存在。
        ValueError: 数据集校验失败（错误信息带行号）。
    """
    src = Path(src_path)
    # 先校验（load_dataset 内部会抛 FileNotFoundError / ValueError），
    # 校验失败时不创建目录、不写任何文件。
    dialogues = load_dataset(src)
    n_questions = sum(len(d.questions) for d in dialogues)

    dest_dir = user_data_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / USER_DATASET_FILENAME
    shutil.copyfile(src, dest)

    metadata = {
        "installed_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(src),
        "n_dialogues": len(dialogues),
        "n_questions": n_questions,
    }
    with (dest_dir / METADATA_FILENAME).open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return {"path": str(dest), **metadata}


def user_dataset_status() -> dict:
    """查询用户测试集安装状态。

    Returns:
        ``{"exists": bool, "path": str|None, "metadata": dict|None}``。
        metadata.json 缺失或损坏时 metadata 为 None。
    """
    path = user_dataset_path()
    if not path.is_file():
        return {"exists": False, "path": None, "metadata": None}
    metadata = None
    meta_path = user_data_dir() / METADATA_FILENAME
    if meta_path.is_file():
        try:
            with meta_path.open(encoding="utf-8") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            metadata = None
    return {"exists": True, "path": str(path), "metadata": metadata}


def delete_user_dataset() -> bool:
    """删除用户测试集（user_dataset.jsonl 与 metadata.json）。

    Returns:
        bool: 删除前用户测试集是否存在。
    """
    status = user_dataset_status()
    existed = bool(status["exists"])
    for fname in (USER_DATASET_FILENAME, METADATA_FILENAME):
        p = user_data_dir() / fname
        if p.is_file():
            p.unlink()
    return existed


def resolve_dataset(specified: str | None) -> tuple[str, str]:
    """按优先级解析实际使用的测试集。

    优先级：

    1. ``specified`` 非空 -> ``(specified, "user-specified")``；
    2. 用户测试集已安装 -> ``(user_dataset_path, "user")``；
    3. 否则 -> ``(内置 office_dataset.jsonl, "builtin-office")``。

    Args:
        specified: 用户显式指定的数据集路径（可为 None）。

    Returns:
        ``(数据集路径, 来源标签)`` 元组。
    """
    if specified:
        return (specified, "user-specified")
    status = user_dataset_status()
    if status["exists"]:
        return (str(status["path"]), "user")
    return (str(builtin_office_dataset_path()), "builtin-office")
