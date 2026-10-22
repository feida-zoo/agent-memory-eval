"""状态快照与回放：按对话存档被测系统的记忆状态，改代码后免重跑 ingest。

分工：**框架只做编排**（目录与 tag、清单、哈希、跑前校验、模式协商、保真
校验），状态的产出与消费全部走适配器接口（``snapshot`` / ``restore`` /
``rebuild``）——框架不解析快照内容、不定义记忆应该长什么样。

为什么要按对话存档：runner 每个 dialogue 前都 ``reset()``，评测本身就是
"一个对话一个独立记忆生命周期"，所以"按对话存档 → 按对话恢复"天然等价，
不需要跨对话的状态连续性。

三种模式（``run_mode``）：

- ``full``：正常评测（可选 ``--snapshot-out`` 顺带导出）；
- ``replay-restore``：快照格式兼容、直接恢复（测**读取/检索/回答**侧改动）；
- ``replay-rebuild``：格式不兼容或需要重建派生结构，走系统自己的 ``rebuild``。

口径：``replay-*`` 固定写入侧，结论只覆盖读取/回答层（与 TranscriptAdapter
同纪律），不得以"完整评测"名义汇报；涉及写入侧改动的验证必须重 ingest。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from typing import Any, Union

__all__ = [
    "MANIFEST_NAME",
    "RUN_MODES",
    "read_capabilities",
    "tag_dir_name",
    "dir_sha256",
    "new_manifest",
    "save_manifest",
    "load_manifest",
    "write_snapshot",
    "plan_replay",
    "replay_dialogue",
    "verify_replay",
]

#: 快照根目录下的清单文件名。
MANIFEST_NAME = "manifest.json"

#: run_mode 取值（``full`` 之外的两种都固定写入侧）。
RUN_MODES: tuple[str, ...] = ("full", "replay-restore", "replay-rebuild")

_SAFE_TAG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def tag_dir_name(tag: str) -> str:
    """把对话 id 归一成安全的目录名（防路径穿越；空值兜底 ``dialogue``）。"""
    safe = _SAFE_TAG_RE.sub("_", str(tag)).strip("._-")
    return (safe or "dialogue")[:120]


def dir_sha256(path: str) -> str:
    """目录内容哈希（相对路径 + 内容，按路径排序）；空目录也能得到稳定值。"""
    digest = hashlib.sha256()
    if not os.path.isdir(path):
        return digest.hexdigest()
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            full = os.path.join(root, name)
            rel = os.path.relpath(full, path)
            digest.update(rel.encode("utf-8"))
            with open(full, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _dir_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def payload_sha256(sessions: list[list[dict]]) -> str:
    """该对话 ingest 输入的指纹（检测"数据集被局部改过"的漂移）。"""
    blob = json.dumps(sessions, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def read_capabilities(adapter: Any) -> dict[str, Any]:
    """归一化并校验适配器的快照能力声明。

    约定：适配器实现 ``snapshot_capabilities()`` 返回
    ``{"supported": bool, "restore": bool, "rebuild": bool,
    "format_version": Any, "compatible_from": [...]}``；缺省 base 实现返回
    ``{"supported": False}``。**声明了就必须实现**对应方法——缺方法直接报错，
    不静默降级。

    Raises:
        SystemExit: 声明 supported/restore/rebuild 但方法缺失。
    """
    fn = getattr(adapter, "snapshot_capabilities", None)
    declared = fn() if callable(fn) else {"supported": False}
    if not isinstance(declared, dict) or not declared.get("supported"):
        return {"supported": False}
    caps = {
        "supported": True,
        "restore": bool(declared.get("restore", True)),
        "rebuild": bool(declared.get("rebuild", False)),
        "format_version": declared.get("format_version", 1),
        "compatible_from": list(declared.get("compatible_from") or []),
    }
    if not callable(getattr(adapter, "snapshot", None)):
        raise SystemExit("适配器声明 snapshot supported=True，但未实现 snapshot() 方法")
    for flag in ("restore", "rebuild"):
        if caps[flag] and not callable(getattr(adapter, flag, None)):
            raise SystemExit(f"适配器声明支持 {flag}，但未实现 {flag}() 方法")
    return caps


def new_manifest(
    *,
    run_id: str,
    adapter: str,
    adapter_version: Union[str, None],
    platform: Union[str, None],
    dataset: str,
    dataset_path: str,
    dataset_fingerprint: dict[str, Any],
    shard_by: str,
    shards: int,
    shard_index: int,
    capabilities: dict[str, Any],
) -> dict[str, Any]:
    """新建（未完成的）快照清单。"""
    return {
        "schema": 1,
        "run_id": run_id,
        "adapter": adapter,
        "adapter_version": adapter_version,
        "platform": platform,
        "dataset": dataset,
        "dataset_path": dataset_path,
        "dataset_fingerprint": dataset_fingerprint,
        "shard_by": shard_by,
        "shards": shards,
        "shard_index": shard_index,
        "capabilities": capabilities,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "complete": False,
        "dialogues": {},
    }


def save_manifest(root: str, manifest: dict[str, Any]) -> str:
    """写清单并返回其 sha256（哈希在写入后按文件内容算，回放时可比对）。"""
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, MANIFEST_NAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return _file_sha256(path)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(root: str) -> dict[str, Any]:
    """读快照清单；缺失/损坏时抛出带修复建议的 SystemExit。"""
    path = os.path.join(root, MANIFEST_NAME)
    if not os.path.isfile(path):
        raise SystemExit(
            f"快照清单不存在: {path}——请确认 --replay 指向的目录是 "
            "--snapshot-out 导出的快照根目录"
        )
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except ValueError as exc:
        raise SystemExit(f"快照清单损坏（{path}）：{exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("dialogues"), dict):
        raise SystemExit(f"快照清单结构非法（{path}）：缺少 dialogues 段")
    return data


def write_snapshot(
    root: str,
    adapter: Any,
    manifest: dict[str, Any],
    *,
    dialogue_id: str,
    sessions: list[list[dict]],
    n_questions: int,
) -> dict[str, Any]:
    """导出单个对话的状态快照并登记到清单。

    Returns:
        该对话的清单条目（含目录名、sha256、字节数、条目统计）。
    """
    dir_name = tag_dir_name(dialogue_id)
    target = os.path.join(root, dir_name)
    os.makedirs(target, exist_ok=True)
    extra = adapter.snapshot(target, dialogue_id)
    entry = {
        "dir": dir_name,
        "tag": dialogue_id,
        "sha256": dir_sha256(target),
        "bytes": _dir_bytes(target),
        "n_sessions": len(sessions),
        "n_messages": sum(len(s) for s in sessions),
        "n_questions": n_questions,
        "ingest_payload_sha256": payload_sha256(sessions),
        "exported_at": datetime.now().isoformat(timespec="seconds"),
    }
    if isinstance(extra, dict) and extra:
        entry["adapter_report"] = extra
    manifest["dialogues"][str(dialogue_id)] = entry
    return entry


def _fingerprint_sha(manifest: dict[str, Any]) -> Union[str, None]:
    fp = manifest.get("dataset_fingerprint") or {}
    return fp.get("sha256") if isinstance(fp, dict) else None


def plan_replay(
    adapter: Any,
    root: str,
    *,
    dataset_fingerprint: dict[str, Any],
    dialogue_ids: list[str],
    requested_mode: str = "auto",
) -> dict[str, Any]:
    """跑前校验并决定回放模式（``restore`` / ``rebuild``）。

    校验：清单存在且可读、数据集指纹一致、本 run 要跑的对话都有快照。
    模式协商：``auto`` 优先 ``restore``（要求格式兼容且适配器声明支持），
    否则退回 ``rebuild``（要求适配器声明支持），都不行则拒绝。

    Raises:
        SystemExit: 任一项不满足（宁可跑前拒绝，也不跑完才发现不可比）。
    """
    manifest = load_manifest(root)
    caps = read_capabilities(adapter)
    if not caps.get("supported"):
        raise SystemExit(
            "该适配器未声明快照能力，无法回放（--replay）："
            "请在适配器里实现 snapshot/restore（或 rebuild）并让 "
            "snapshot_capabilities() 返回 supported=True；HTTP 桥同理"
        )

    want_fp = dataset_fingerprint.get("sha256") if isinstance(dataset_fingerprint, dict) else None
    snap_fp = _fingerprint_sha(manifest)
    if want_fp and snap_fp and want_fp != snap_fp:
        raise SystemExit(
            "数据集指纹与快照不一致，拒绝回放："
            f"当前数据集 sha256={want_fp} vs 快照 {snap_fp}——快照是另一份数据集"
            "上导出的，回放会得到不可归因的分数（请用同一数据集重新导出快照）"
        )

    missing = [d for d in dialogue_ids if str(d) not in manifest["dialogues"]]
    if missing:
        raise SystemExit(
            "快照缺少本 run 需要的对话，拒绝回放："
            + "、".join(missing[:5])
            + (f" 等 {len(missing)} 个" if len(missing) > 5 else "")
            + "——请确认 --snapshot-out 的导出覆盖了这些对话（分片导出时需覆盖同一分片口径）"
        )

    snap_version = (manifest.get("capabilities") or {}).get("format_version", 1)
    compatible = snap_version in (caps.get("compatible_from") or [])
    same_version = snap_version == caps.get("format_version", 1)

    def _restore_ok() -> bool:
        return same_version or compatible

    if requested_mode == "restore" and not _restore_ok():
        raise SystemExit(
            f"快照格式 v{snap_version} 与当前适配器 v{caps.get('format_version')} 不兼容，"
            "无法 --replay-mode restore（改用 rebuild 或重新导出快照）"
        )
    if requested_mode == "rebuild" and not caps.get("rebuild"):
        raise SystemExit(
            "适配器未声明 rebuild 能力，无法 --replay-mode rebuild"
            "（仅声明了 restore，或需要先用当前版本重新导出快照）"
        )

    if requested_mode == "rebuild":
        mode = "rebuild"
    elif requested_mode == "restore":
        mode = "restore"
    elif _restore_ok() and caps.get("restore"):
        mode = "restore"
    elif caps.get("rebuild"):
        mode = "rebuild"
    else:
        raise SystemExit(
            f"快照格式 v{snap_version} 与当前适配器 v{caps.get('format_version')} 不兼容，"
            "且适配器未声明 rebuild 能力：该改动必须重新 ingest（快照回放只能覆盖"
            "读取侧；派生结构/存储结构变更需要系统提供 rebuild，或重跑）"
        )
    return {
        "root": root,
        "mode": mode,
        "run_mode": f"replay-{mode}",
        "format_version": snap_version,
        "source_run": manifest.get("run_id"),
        "manifest_sha256": _file_sha256(os.path.join(root, MANIFEST_NAME)),
        "complete": bool(manifest.get("complete")),
    }


def replay_dialogue(adapter: Any, plan: dict[str, Any], dialogue_id: str) -> None:
    """按计划恢复单个对话的状态（回放模式下替代 reset + ingest）。"""
    entry = (load_manifest(plan["root"])["dialogues"].get(str(dialogue_id))) or {}
    dir_name = entry.get("dir") or tag_dir_name(dialogue_id)
    source = os.path.join(plan["root"], dir_name)
    if not os.path.isdir(source):
        raise SystemExit(f"快照目录不存在: {source}（清单里登记了 {dialogue_id}）")
    if plan["mode"] == "rebuild":
        adapter.rebuild(source, dialogue_id)
    else:
        adapter.restore(source, dialogue_id)


def verify_replay(source_run_dir: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """回放保真校验：与源 run 逐题比对（qid 对齐）。

    Returns:
        ``{"source_run", "n_source", "n_replay", "n_common", "n_answer_diff",
        "n_retrieved_diff", "n_correct_diff", "answer_match_rate",
        "mismatch_qids", ...}``；源 results.jsonl 缺失时返回 ``{"error": ...}``。
    """
    source_path = os.path.join(source_run_dir, "results.jsonl")
    if not os.path.isfile(source_path):
        return {"error": f"源 run 缺少 results.jsonl: {source_path}"}
    source: dict[str, dict[str, Any]] = {}
    with open(source_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            source[str(rec.get("qid", ""))] = rec

    replay = {str(r.get("qid", "")): r for r in records}
    common = [q for q in replay if q in source]
    answer_diff: list[str] = []
    retrieved_diff: list[str] = []
    correct_diff: list[str] = []
    for qid in common:
        a, b = source[qid], replay[qid]
        if str(a.get("prediction", "")) != str(b.get("prediction", "")):
            answer_diff.append(qid)
        if [str(x) for x in a.get("retrieved") or []] != [str(x) for x in b.get("retrieved") or []]:
            retrieved_diff.append(qid)
        if bool((a.get("judge") or {}).get("correct")) != bool((b.get("judge") or {}).get("correct")):
            correct_diff.append(qid)
    return {
        "source_run": os.path.basename(os.path.normpath(source_run_dir)),
        "n_source": len(source),
        "n_replay": len(replay),
        "n_common": len(common),
        "n_answer_diff": len(answer_diff),
        "n_retrieved_diff": len(retrieved_diff),
        "n_correct_diff": len(correct_diff),
        "answer_match_rate": (1 - len(answer_diff) / len(common)) if common else None,
        "mismatch_qids": sorted(set(answer_diff) | set(retrieved_diff) | set(correct_diff))[:10],
    }
