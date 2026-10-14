"""公开基准数据集的按需下载与转换（``dataset fetch`` 的实现后端）。

把原先散落在 ``cli.py`` 的 ``_download_file``/``_fetch_beam`` 迁移到这里，
并补齐 LocMemEval/GateMem 等 entry，使 ``info.py`` 里 ``status="fetch"`` 的
每个 key 都有对应的自动下载实现——此前只有 BEAM 有，其余会以「暂无下载配置」
报错，与注册表和根 README 的承诺不一致。

产物统一落 ``dataset_out_path(key)``（缺省 ``info.datasets_dir()``，即
``MEMBENCH_DATASETS_DIR`` > ``~/.membench/datasets`` 下的 ``<key>.jsonl``）。
原始文件缓存在产物同级的 ``.<key>_raw/`` 下，便于重转换：已有可解析的完整文件时
跳过下载；不做 HTTP Range 续传，下载中断后整体重下。

下载源（2026-10 核实）::

    locomo            raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
                      （文件存在性经 GitHub API 确认）
    longmemeval_*     huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/
                      longmemeval_oracle.json 与 longmemeval_s_cleaned.json
                      （文件列表与 resolve 直链经 HF 核实；S 变体不是
                      longmemeval_s.json——该名不存在，直链 404）
    gatemem_office    raw.githubusercontent.com/rzhub/GateMem/main/bench/data/office/
                      {episodes,checkpoints}.jsonl
                      （路径经 GitHub API 确认存在；raw 直连未验证）
    beam_{1m,10m}     raw.githubusercontent.com/mohammadtavakoli78/BEAM/main/
                      chats/{1M,10M}/<i>/chat.json 与 .../probing_questions/probing_questions.json
                      （沿用既有实现；仓库结构经 GitHub API 确认）

GateMem 采用 ``ignore_privacy_refuse=True`` 转换：``(privacy, refuse)`` 是多
用户权限语义，单用户个人 agent 场景不适用（见 ``info.py`` 对 gatemem_office
「已标记 ignore=true」的描述与 ``convert_gatemem`` docstring）。
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import Callable

from membench.data import info as dataset_info

#: 支持按需下载（``dataset fetch``）的 key 全集。
FETCH_KEYS: tuple[str, ...] = (
    "locomo",
    "longmemeval_oracle",
    "longmemeval_s",
    "gatemem_office",
    "beam_1m",
    "beam_10m",
)

#: LoCoMo 官方样本（10 个长对话）。
_LOCOMO_RAW_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

#: LongMemEval cleaned 变体在 HuggingFace 的直链前缀。
_LONG_MEM_EVAL_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"

#: LongMemEval key -> 远端文件名。
#: 注意 S 变体的文件名是 ``longmemeval_s_cleaned.json``，不存在 ``longmemeval_s.json``
#: （2026-10 经 HF 文件列表与 resolve 直链核实，后者返回 404）。
_LONG_MEM_EVAL_FILES: dict[str, str] = {
    "longmemeval_oracle": "longmemeval_oracle.json",
    "longmemeval_s": "longmemeval_s_cleaned.json",
}

#: GateMem 办公域原始文件前缀（episodes.jsonl / checkpoints.jsonl）。
_GATEMEM_RAW_BASE = "https://raw.githubusercontent.com/rzhub/GateMem/main/bench/data/office/"

#: BEAM 原始数据在 GitHub raw 上的基础 URL。
_BEAM_RAW_BASE = "https://raw.githubusercontent.com/mohammadtavakoli78/BEAM/main/"

#: BEAM key -> (远端子目录, chat 数, convert_beam_split 的 scale 标识)。
_BEAM_CONFIG: dict[str, tuple[str, int, str]] = {
    "beam_1m": ("1M", 35, "1m"),
    "beam_10m": ("10M", 10, "10m"),
}

#: BEAM 每个 chat 目录需下载的文件（远端相对路径 -> 本地文件名）。
#: 官方结构是 chats/{scale}/{id}/probing_questions/probing_questions.json，
#: 下载后落到 chat 目录根供 convert_beam 读取。
_BEAM_CHAT_FILES: tuple[tuple[str, str], ...] = (
    ("chat.json", "chat.json"),
    ("probing_questions/probing_questions.json", "probing_questions.json"),
)


def is_fetchable(key: str) -> bool:
    """判断某 key 是否支持按需下载。"""
    return key in FETCH_KEYS


def dataset_out_path(key: str, out_dir: str | Path | None = None) -> Path:
    """返回某数据集 fetch 后的 jsonl 落盘路径。

    Args:
        key: 数据集键。
        out_dir: 输出目录；缺省 ``membench.data.info.datasets_dir()``
            （``MEMBENCH_DATASETS_DIR`` > ``~/.membench/datasets``）。

    Returns:
        Path: ``<out_dir>/<key>.jsonl``。
    """
    base = Path(out_dir).expanduser() if out_dir is not None else dataset_info.datasets_dir()
    return base / f"{key}.jsonl"


def _is_complete_json(path: Path) -> bool:
    """判断已下载文件是否完整：整体是合法 JSON，或每行都是合法 JSON（JSONL）。"""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    try:
        json.loads(text)
        return True
    except ValueError:
        pass
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    try:
        for line in lines:
            json.loads(line)
        return True
    except ValueError:
        return False


def _download_file(
    url: str, dest: Path, retries: int = 3, force: bool = False
) -> None:
    """下载单个文件（失败重试 retries 次）。

    目标已存在且内容完整（合法 JSON 或 JSONL）时直接返回（跳过联网）；这里**没有**
    HTTP Range 续传，下载中断后整体重下。``force=True`` 时无条件重新下载并覆盖。

    重试耗尽后抛 RuntimeError。
    """
    dest = Path(dest)
    if not force and dest.is_file() and _is_complete_json(dest):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_exc: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                dest.write_bytes(resp.read())
            return
        except Exception as exc:  # noqa: BLE001 - 网络错误统一重试
            last_exc = exc
            print(
                f"[membench] 下载失败（第 {attempt}/{retries} 次）: {url}: {exc}",
                file=sys.stderr,
            )
    raise RuntimeError(f"下载失败（已重试 {retries} 次）: {url}: {last_exc}")


def _fetch_locomo(key: str, raw_dir: Path, out_path: Path, force: bool = False) -> None:
    """下载 LoCoMo 原始数据并转换。"""
    raw = raw_dir / "locomo10.json"
    _download_file(_LOCOMO_RAW_URL, raw, force=force)
    from membench.data.convert_locomo import convert_locomo

    convert_locomo(str(raw), str(out_path))


def _fetch_longmemeval(
    key: str, raw_dir: Path, out_path: Path, force: bool = False
) -> None:
    """下载 LongMemEval（oracle / s 变体）原始数据并转换。"""
    remote_name = _LONG_MEM_EVAL_FILES[key]
    raw = raw_dir / remote_name
    _download_file(_LONG_MEM_EVAL_BASE + remote_name, raw, force=force)
    from membench.data.convert_longmemeval import convert_longmemeval

    convert_longmemeval(str(raw), str(out_path))


def _fetch_gatemem(key: str, raw_dir: Path, out_path: Path, force: bool = False) -> None:
    """下载 GateMem 办公域 episodes + checkpoints 并转换（单用户场景忽略权限拒答题）。"""
    episodes = raw_dir / "episodes.jsonl"
    checkpoints = raw_dir / "checkpoints.jsonl"
    _download_file(_GATEMEM_RAW_BASE + "episodes.jsonl", episodes, force=force)
    _download_file(_GATEMEM_RAW_BASE + "checkpoints.jsonl", checkpoints, force=force)
    from membench.data.convert_gatemem import convert_gatemem

    convert_gatemem(
        str(episodes), str(checkpoints), str(out_path), ignore_privacy_refuse=True
    )


def _fetch_beam(key: str, raw_dir: Path, out_path: Path, force: bool = False) -> None:
    """下载 BEAM 某规模的 chat 目录并转换。"""
    subdir, n_chats, scale = _BEAM_CONFIG[key]
    for i in range(1, n_chats + 1):
        chat_dir = raw_dir / str(i)
        chat_dir.mkdir(parents=True, exist_ok=True)
        for remote_rel, local_name in _BEAM_CHAT_FILES:
            _download_file(
                f"{_BEAM_RAW_BASE}chats/{subdir}/{i}/{remote_rel}",
                chat_dir / local_name,
                force=force,
            )
        print(f"[membench] 已下载 chat {i}/{n_chats}", file=sys.stderr)
    from membench.data.convert_beam import convert_beam_split

    convert_beam_split(str(raw_dir), scale, str(out_path))


#: key -> 下载+转换实现。缺失的 key 视为「下载源未确证」，fetch 时抛 RuntimeError。
_FETCHERS: dict[str, Callable[..., None]] = {
    "locomo": _fetch_locomo,
    "longmemeval_oracle": _fetch_longmemeval,
    "longmemeval_s": _fetch_longmemeval,
    "gatemem_office": _fetch_gatemem,
    "beam_1m": _fetch_beam,
    "beam_10m": _fetch_beam,
}


def _is_loadable(path: Path) -> bool:
    """判断产物能否被 schema.load_dataset 解析出**至少 1 个** dialogue。

    空文件 / 纯空白文件也会被 ``load_dataset`` 成功解析为 ``[]``（不报错），若据此
    判定"已完成"，``dataset fetch`` 会永久返回空集（有真实产生路径：``convert_beam_split``
    在全部 chat 损坏时只警告、写出 0 字节文件），故要求 dialogue 数 > 0。
    解析异常一律视为不可用。
    """
    try:
        from membench.data.schema import load_dataset

        return len(load_dataset(path)) > 0
    except Exception:  # noqa: BLE001 - 半成品/损坏文件都当作需要重做
        return False


def fetch_dataset(
    key: str,
    out_dir: str | Path | None = None,
    *,
    force: bool = False,
) -> Path:
    """下载并转换某数据集，返回产物 jsonl 路径。

    幂等：``<out_dir>/<key>.jsonl`` 已存在且能被 ``load_dataset`` 解析出至少 1 个
    dialogue 时直接返回（不重复下载/转换）。

    ``force=True`` 的语义边界：绕过 ``.<key>_raw/`` 原始缓存——即使缓存文件完整也
    重新下载原始数据，再重新转换覆盖写出 ``<out_dir>/<key>.jsonl``。

    Args:
        key: 数据集键，须在 ``FETCH_KEYS`` 内。
        out_dir: 输出目录；缺省 ``info.datasets_dir()``。
        force: 绕过 raw 缓存重新下载并转换。

    Returns:
        Path: 产物 ``<out_dir>/<key>.jsonl``。

    Raises:
        KeyError: key 不在 ``FETCH_KEYS``（未知 key 或非 fetch 数据集）。
        RuntimeError: key 已登记为可 fetch 但下载源未确证。
    """
    if key not in FETCH_KEYS:
        raise KeyError(
            f"不支持按需下载的数据集: {key}。可下载的 key: {', '.join(FETCH_KEYS)}"
        )

    out_path = dataset_out_path(key, out_dir)
    if not force and out_path.is_file() and _is_loadable(out_path):
        return out_path

    fetcher = _FETCHERS.get(key)
    if fetcher is None:
        entry = dataset_info.DATASET_REGISTRY.get(key)
        source = entry.source if entry is not None else "（注册表中无此 key）"
        raise RuntimeError(
            f"{key} 暂无可确证的自动下载源，请按以下来源手动获取并转换: {source}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = out_path.parent / f".{key}_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fetcher(key, raw_dir, out_path, force=force)
    return out_path
