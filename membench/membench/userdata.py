"""用户测试集管理与数据集入参解析：install / status / delete / resolve。

用户测试集存放于用户测试集目录（默认 ``./user_dataset/``，可用环境变量
``MEMBENCH_USER_DATA_DIR`` 覆盖），目录内包含：

- ``user_dataset.jsonl``：校验通过后复制的数据集本体；
- ``metadata.json``：安装时间、来源路径、dialogue/question 统计。

``--dataset`` 的入参（数据集 key 或 JSONL 路径）统一由
:func:`resolve_dataset_arg` 解析；``run`` 与 ``evaluate`` 共用这一套规则：

1. 存在的 JSONL 文件 -> 原样使用，来源 ``"user-specified"``；
2. ``office`` -> 用户测试集已安装则用它（``"user"``），否则内置 office（``"builtin-office"``）；
3. 其它注册表 key -> 内置文件或 ``dataset fetch`` 下载目录（``"fetch"``）。

未指定 ``--dataset`` 时的回退优先级（:func:`resolve_dataset`）：用户测试集 >
内置 office_dataset.jsonl。
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from membench.data.info import (
    DATASET_REGISTRY,
    dataset_key_hint,
    infer_dataset_source,
    resolve_available_path,
)
from membench.data.schema import load_dataset

#: 用户测试集文件名（位于用户测试集目录内）。
USER_DATASET_FILENAME = "user_dataset.jsonl"

#: 元数据文件名（位于用户测试集目录内）。
METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class ResolvedDataset:
    """``--dataset`` 入参（key 或 JSONL 路径）的统一解析结果。

    Attributes:
        key: 数据集标识——路径入参取文件名 stem，key 入参即 key 本身。
        path: 数据集 JSONL 的绝对路径。
        source: 来源标签：``user-specified`` 显式文件、``user`` 用户测试集、
            ``builtin-office`` 内置办公集、``fetch`` 来自数据集注册表（内置文件或
            ``dataset fetch`` 下载目录；与 ``evaluate`` 的历史口径一致）。
    """

    key: str
    path: str
    source: str


def _looks_like_path(value: str) -> bool:
    """值看起来像文件路径：含路径分隔符 / ``~``，或以 ``.jsonl`` / ``.json`` 结尾。"""
    return (
        "/" in value
        or os.sep in value
        or "~" in value
        or value.lower().endswith((".jsonl", ".json"))
    )


def resolve_dataset_arg(value: str) -> ResolvedDataset:
    """把 ``--dataset`` 的单个值解析为数据集（key 与路径同一套规则）。

    解析顺序：

    1. 存在的文件（``~`` 会展开）-> 原样使用，key 取 stem；来源按路径反查
       （:func:`membench.data.info.infer_dataset_source`）：内置文件记
       ``builtin`` / ``builtin-office``、``dataset fetch`` 目录里的基准集记
       ``fetch``，其余 ``user-specified``；
    2. 存在但不是文件 -> 报错（数据集必须是 JSONL 文件，不能是目录）；
    3. 看起来像路径但不存在（含 ``/`` / ``~`` / 以 ``.jsonl`` / ``.json`` 结尾）
       -> 直接报错，避免下游把它当数据文件打开后抛 schema 错误；
    4. ``office`` -> 用户测试集已安装则用它（``"user"``），否则内置 office
       （``"builtin-office"``）；
    5. 其它注册表 key -> :func:`membench.data.info.resolve_available_path`（内置文件
       优先，其次 ``dataset fetch`` 下载目录），来源 ``"fetch"``；解析不到报错提示先
       下载并列出可用 key；
    6. 既不是文件、也不是注册表 key、也不像路径 -> 报错列出可用 key 与
       "也可以直接传 JSONL 路径"。

    Args:
        value: ``--dataset`` 原始值（数据集 key 或 JSONL 路径）。

    Returns:
        :class:`ResolvedDataset`。

    Raises:
        SystemExit: 空值、路径不存在 / 不是文件、key 未下载或 key 未知。
    """
    text = value.strip()
    if not text:
        raise SystemExit("--dataset 收到空值（key 或 JSONL 路径）")

    candidate = Path(text).expanduser()
    if candidate.is_file():
        resolved = candidate.resolve()
        # 显式路径不一定就是"用户自定义"：编排器 --common 会先把公开基准集
        # 解析成路径再下发，这里按路径反查来源，避免报告把基准集标成用户集。
        source = infer_dataset_source(resolved) or "user-specified"
        return ResolvedDataset(key=resolved.stem, path=str(resolved), source=source)
    if candidate.exists():
        raise SystemExit(f"数据集路径不是文件: {text}（--dataset 需要 JSONL 文件）")
    if _looks_like_path(text):
        raise SystemExit(
            f"数据集文件不存在: {text}；也可以直接传数据集 key"
            "（如 office / sample，全部 key 见 python -m membench.cli list-datasets）"
        )

    if text == "office":
        status = user_dataset_status()
        if status["exists"]:
            return ResolvedDataset(key="office", path=str(status["path"]), source="user")
        return ResolvedDataset(
            key="office",
            path=str(builtin_office_dataset_path()),
            source="builtin-office",
        )

    if text in DATASET_REGISTRY:
        path = resolve_available_path(text)
        if path is None:
            raise SystemExit(
                f"数据集 {text} 尚未下载，请先运行: "
                f"python -m membench.cli dataset fetch {text}；{dataset_key_hint()}"
            )
        return ResolvedDataset(key=text, path=str(path), source="fetch")

    raise SystemExit(
        f"未知数据集: {text}；既不是存在的文件路径，也不是数据集 key；"
        f"{dataset_key_hint()}；按需下载的集先运行 "
        "python -m membench.cli dataset fetch <key>，也可以直接传 JSONL 路径"
        "（如 ./my_dataset.jsonl）"
    )


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

    1. ``specified`` 非空 -> 交给 :func:`resolve_dataset_arg` 统一解析
       （key 或 JSONL 路径；来源由解析结果给出）；
    2. 用户测试集已安装 -> ``(user_dataset_path, "user")``；
    3. 否则 -> ``(内置 office_dataset.jsonl, "builtin-office")``。

    Args:
        specified: 用户显式指定的数据集 key 或路径（可为 None）。

    Returns:
        ``(数据集路径, 来源标签)`` 元组。

    Raises:
        SystemExit: 显式指定值无法解析（空值 / 路径不存在 / key 未知或未下载）。
    """
    if specified is not None:
        resolved = resolve_dataset_arg(specified)
        return (resolved.path, resolved.source)
    status = user_dataset_status()
    if status["exists"]:
        return (str(status["path"]), "user")
    return (str(builtin_office_dataset_path()), "builtin-office")
