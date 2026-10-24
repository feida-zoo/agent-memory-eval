"""状态快照与回放：按对话存档被测系统的记忆状态，改代码后免重跑 ingest。

分工：**框架只做编排**（目录与 tag、清单、哈希、跑前校验、模式协商、保真
校验），状态的产出与消费全部走适配器接口（``snapshot`` / ``restore`` /
``rebuild``）——框架不解析快照内容、不定义记忆应该长什么样。

为什么按对话存档：runner 每个 dialogue 前都 ``reset()``，评测本身就是
"一个对话一个独立记忆生命周期"，所以"按对话存档 → 按对话恢复"天然等价，
不需要跨对话的状态连续性。

清单也**每个对话一份**，落在 ``<数据集目录>/<对话目录>/memory_snapshot/manifest.json``，
自包含（身份 + 数据集指纹 + 能力声明 + 该对话条目）——单个对话目录即可搬到
别处使用。整场评测的齐备性（对话覆盖 / 数据集指纹 / 格式版本 / 模式协商）
由 :func:`plan_replay` 扫描各对话清单统一校验，合并指纹见
:func:`manifests_digest`。

三种模式（``run_mode``）：

- ``full``：正常评测（默认导出对话状态快照，``--no-snapshot`` 关闭）；
- ``replay-restore``：快照格式兼容、直接恢复（测**读取/检索/回答**侧改动）；
- ``replay-rebuild``：格式不兼容或需要重建派生结构，走系统自己的 ``rebuild``。

口径：``replay-*`` 固定写入侧，结论只覆盖读取/回答层（与 TranscriptAdapter
同纪律），不得以"完整评测"名义汇报；涉及写入侧改动的验证必须重 ingest。
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import Any, Union

from membench import layout

__all__ = [
    "MANIFEST_NAME",
    "RUN_MODES",
    "read_capabilities",
    "dir_sha256",
    "payload_sha256",
    "new_manifest",
    "save_manifest",
    "load_manifest",
    "write_snapshot",
    "manifests_digest",
    "plan_replay",
    "replay_dialogue",
    "verify_replay",
]

#: 快照清单文件名（落在每个对话的 ``memory_snapshot/`` 目录里）。
MANIFEST_NAME = "manifest.json"

#: run_mode 取值（``full`` 之外的两种都固定写入侧）。
RUN_MODES: tuple[str, ...] = ("full", "replay-restore", "replay-rebuild")


def dir_sha256(path: str, exclude: tuple[str, ...] = ()) -> str:
    """目录内容哈希（相对路径 + 内容，按路径排序）；空目录也能得到稳定值。

    ``exclude`` 里的顶层文件名不计入——快照目录里的 ``manifest.json`` 由框架
    在算完哈希后才写，排除它才能让哈希只代表适配器产出的内容。
    """
    digest = hashlib.sha256()
    if not os.path.isdir(path):
        return digest.hexdigest()
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            if root == path and name in exclude:
                continue
            full = os.path.join(root, name)
            rel = os.path.relpath(full, path)
            digest.update(rel.encode("utf-8"))
            with open(full, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _dir_bytes(path: str, exclude: tuple[str, ...] = ()) -> int:
    """目录内文件字节数合计（``exclude`` 语义同 :func:`dir_sha256`）。"""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            if root == path and name in exclude:
                continue
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
    dialogue_id: str,
    capabilities: dict[str, Any],
) -> dict[str, Any]:
    """新建单对话快照清单（``entry`` 与 ``complete`` 由 :func:`write_snapshot` 落定）。"""
    return {
        "schema": 1,
        "run_id": run_id,
        "adapter": adapter,
        "adapter_version": adapter_version,
        "platform": platform,
        "dataset": dataset,
        "dataset_path": dataset_path,
        "dataset_fingerprint": dataset_fingerprint,
        "dialogue_id": str(dialogue_id),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "complete": False,
        "capabilities": capabilities,
    }


def save_manifest(snapshot_dir: str, manifest: dict[str, Any]) -> str:
    """写清单到 ``<snapshot_dir>/manifest.json`` 并返回其 sha256。"""
    os.makedirs(snapshot_dir, exist_ok=True)
    path = os.path.join(snapshot_dir, MANIFEST_NAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return _file_sha256(path)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(snapshot_dir: str) -> dict[str, Any]:
    """读单对话快照清单；缺失/损坏时抛出带修复建议的 SystemExit。"""
    path = os.path.join(snapshot_dir, MANIFEST_NAME)
    if not os.path.isfile(path):
        raise SystemExit(
            f"快照清单不存在: {path}——请确认 --replay 指向的是导出了快照的"
            "数据集目录（或含它的实验目录），且该目录覆盖了本次要跑的对话"
        )
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except ValueError as exc:
        raise SystemExit(f"快照清单损坏（{path}）：{exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("entry"), dict):
        raise SystemExit(f"快照清单结构非法（{path}）：缺少 entry 段")
    return data


def write_snapshot(
    snapshot_dir: str,
    adapter: Any,
    manifest: dict[str, Any],
    *,
    dialogue_id: str,
    sessions: list[list[dict]],
    n_questions: int,
) -> dict[str, Any]:
    """导出单个对话的状态快照、登记清单并落盘。

    清单在适配器 ``snapshot()`` 成功返回后才写——``complete=true`` 即"该对话
    快照已完整导出"，导出中途崩溃不会留下会被回放读到的半成品清单。

    Returns:
        该对话的清单条目（含目录名、sha256、字节数、条目统计）。
    """
    os.makedirs(snapshot_dir, exist_ok=True)
    extra = adapter.snapshot(snapshot_dir, dialogue_id)
    entry = {
        "dir": layout.tag_dir_name(dialogue_id),
        "tag": dialogue_id,
        "sha256": dir_sha256(snapshot_dir, exclude=(MANIFEST_NAME,)),
        "bytes": _dir_bytes(snapshot_dir, exclude=(MANIFEST_NAME,)),
        "n_sessions": len(sessions),
        "n_messages": sum(len(s) for s in sessions),
        "n_questions": n_questions,
        "ingest_payload_sha256": payload_sha256(sessions),
        "exported_at": datetime.now().isoformat(timespec="seconds"),
    }
    if isinstance(extra, dict) and extra:
        entry["adapter_report"] = extra
    manifest["entry"] = entry
    manifest["complete"] = True
    save_manifest(snapshot_dir, manifest)
    return entry


def manifests_digest(snapshot_dirs: dict[str, str]) -> str:
    """逐对话清单的合并指纹：按对话 id 排序，对 ``id\\tsha256(清单)`` 行求 sha256。

    Args:
        snapshot_dirs: ``{对话 id: 该对话的 memory_snapshot 目录}``。

    单份清单只覆盖一个对话，装不下"整场评测的快照有没有变过"；合并值让
    metrics / 报告能用一个字符串表达快照身份。
    """
    digest = hashlib.sha256()
    for dialogue_id in sorted(snapshot_dirs):
        path = os.path.join(snapshot_dirs[dialogue_id], MANIFEST_NAME)
        digest.update(f"{dialogue_id}\t{_file_sha256(path)}\n".encode("utf-8"))
    return digest.hexdigest()


def _fingerprint_sha(manifest: dict[str, Any]) -> Union[str, None]:
    fp = manifest.get("dataset_fingerprint") or {}
    return fp.get("sha256") if isinstance(fp, dict) else None


def _source_run_id(dataset_dir: str) -> str:
    """源 run 身份：优先取数据集级 metrics.json 的 run_id，缺失时退回目录名。"""
    path = os.path.join(dataset_dir, layout.DIR_REPORT, layout.FILE_METRICS)
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as handle:
                run_id = json.load(handle).get("run_id")
        except ValueError:
            run_id = None
        if run_id:
            return str(run_id)
    return os.path.basename(os.path.normpath(dataset_dir))


def plan_replay(
    root: str,
    dataset: str,
    adapter: Any,
    dialogue_ids: list[str],
    *,
    dataset_fingerprint: dict[str, Any] | None = None,
    requested_mode: str = "auto",
) -> dict[str, Any]:
    """扫描逐对话清单、跑前校验并决定回放模式（``restore`` / ``rebuild``）。

    Args:
        root: 实验目录（自动在其 ``runs/`` 下定位匹配 ``dataset`` 的数据集目录）
            或数据集目录（``<实验目录>/runs/<dataset>``）。
        dataset: 本次评测的数据集名（用于定位与身份校验）。
        adapter: 当前适配器（读其快照能力声明）。
        dialogue_ids: 本 run 要跑的对话 id——每一个都必须有完整快照。
        dataset_fingerprint: 当前数据集的指纹（``dataset_fingerprint()`` 结果）；
            与快照记录的不一致即拒绝。
        requested_mode: ``auto`` / ``restore`` / ``rebuild``。

    校验（全部在开跑之前）：清单齐备且 ``complete=true``、清单身份与本次一致
    （数据集 / 对话 id / 对话目录名）、数据集指纹一致、各对话格式版本一致。
    模式协商：``auto`` 优先 ``restore``（要求格式兼容且适配器声明支持），
    否则退回 ``rebuild``（要求适配器声明支持），都不行则拒绝。

    Returns:
        ``{"root": 数据集目录, "mode", "run_mode", "format_version",
        "source_run", "manifest_sha256", "snapshots": {对话 id: 快照目录}}``。

    Raises:
        SystemExit: 任一项不满足（宁可跑前拒绝，也不跑完才发现不可比）。
    """
    dataset_root = layout.resolve_dataset_dir(root, dataset=dataset)
    caps = read_capabilities(adapter)
    if not caps.get("supported"):
        raise SystemExit(
            "该适配器未声明快照能力，无法回放（--replay）："
            "请在适配器里实现 snapshot/restore（或 rebuild）并让 "
            "snapshot_capabilities() 返回 supported=True；HTTP 桥同理"
        )

    dialogue_ids = [str(d) for d in dialogue_ids]
    snapshot_dirs = {
        did: os.path.join(
            dataset_root, layout.tag_dir_name(did), layout.DIR_SNAPSHOT
        )
        for did in dialogue_ids
    }
    missing = [d for d, path in snapshot_dirs.items() if not os.path.isfile(
        os.path.join(path, MANIFEST_NAME)
    )]
    if missing:
        raise SystemExit(
            "快照缺少本 run 需要的对话，拒绝回放："
            + "、".join(missing[:5])
            + (f" 等 {len(missing)} 个" if len(missing) > 5 else "")
            + "——请确认快照导出覆盖了这些对话（分片导出时需覆盖同一分片口径）"
        )

    want_fp = (
        dataset_fingerprint.get("sha256")
        if isinstance(dataset_fingerprint, dict)
        else None
    )
    versions: dict[str, list[str]] = {}
    raw_versions: dict[str, Any] = {}
    run_ids: set[str] = set()
    for did, snapshot_dir in snapshot_dirs.items():
        path = os.path.join(snapshot_dir, MANIFEST_NAME)
        manifest = load_manifest(snapshot_dir)
        if not manifest.get("complete"):
            raise SystemExit(
                f"快照清单标记为未完成（complete=false），拒绝回放：{path}"
                "——快照导出被中断，请重新导出"
            )
        if str(manifest.get("dialogue_id")) != did:
            raise SystemExit(
                f"快照清单与对话不匹配，拒绝回放：{path} 登记的是 "
                f"{manifest.get('dialogue_id')!r}，本次需要 {did!r}"
            )
        if (manifest.get("entry") or {}).get("dir") != layout.tag_dir_name(did):
            raise SystemExit(
                f"快照清单登记的对话目录与所在位置不一致，拒绝回放：{path}"
                f"（entry.dir={(manifest.get('entry') or {}).get('dir')!r}）"
                "——快照目录可能被移动或改名，请重新导出"
            )
        if str(manifest.get("dataset")) != dataset:
            raise SystemExit(
                f"快照来自另一个数据集，拒绝回放：{path} 登记的是 "
                f"{manifest.get('dataset')!r}，本次评测 {dataset!r}"
            )
        snap_fp = _fingerprint_sha(manifest)
        if want_fp and snap_fp and want_fp != snap_fp:
            raise SystemExit(
                "数据集指纹与快照不一致，拒绝回放："
                f"当前数据集 sha256={want_fp} vs 快照 {snap_fp}——快照是另一份数据集"
                "上导出的，回放会得到不可归因的分数（请用同一数据集重新导出快照）"
            )
        version_raw = (manifest.get("capabilities") or {}).get("format_version", 1)
        versions.setdefault(str(version_raw), []).append(did)
        raw_versions.setdefault(str(version_raw), version_raw)
        run_ids.add(str(manifest.get("run_id") or "n/a"))

    if len(versions) > 1:
        detail = "；".join(
            f"v{ver}（{'、'.join(ids[:3])}）" for ver, ids in sorted(versions.items())
        )
        raise SystemExit(
            f"各对话的快照格式版本不一致，拒绝回放：{detail}"
            "——这些快照不是同一次导出的，请统一重新导出"
        )
    snap_version = next(iter(versions))
    compatible = snap_version in [str(v) for v in caps.get("compatible_from") or []]
    same_version = snap_version == str(caps.get("format_version", 1))

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
        "root": dataset_root,
        "mode": mode,
        "run_mode": f"replay-{mode}",
        "format_version": raw_versions[snap_version],
        "source_run": "、".join(sorted(run_ids)) if len(run_ids) > 1 else next(iter(run_ids)),
        "manifest_sha256": manifests_digest(snapshot_dirs),
        "snapshots": snapshot_dirs,
    }


def replay_dialogue(adapter: Any, plan: dict[str, Any], dialogue_id: str) -> None:
    """按计划恢复单个对话的状态（回放模式下替代 reset + ingest）。"""
    source = (plan.get("snapshots") or {}).get(str(dialogue_id))
    if not source or not os.path.isdir(source):
        raise SystemExit(f"快照目录不存在: {source}（计划里没有对话 {dialogue_id}）")
    if plan["mode"] == "rebuild":
        adapter.rebuild(source, dialogue_id)
    else:
        adapter.restore(source, dialogue_id)


def verify_replay(source_dataset_dir: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """回放保真校验：与源数据集目录的逐题明细比对（qid 对齐）。

    Args:
        source_dataset_dir: 源评测的数据集目录（``<实验目录>/runs/<dataset>``）。
        records: 本次回放的逐题记录。

    Returns:
        ``{"source_run", "n_source", "n_replay", "n_common", "n_answer_diff",
        "n_retrieved_diff", "n_correct_diff", "answer_match_rate",
        "mismatch_qids", ...}``；源逐题明细缺失时返回 ``{"error": ...}``。
    """
    source_path = os.path.join(
        source_dataset_dir, layout.DIR_REPORT, layout.FILE_RESULTS
    )
    if not os.path.isfile(source_path):
        return {"error": f"源数据集目录缺少逐题明细: {source_path}"}
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
        "source_run": _source_run_id(source_dataset_dir),
        "n_source": len(source),
        "n_replay": len(replay),
        "n_common": len(common),
        "n_answer_diff": len(answer_diff),
        "n_retrieved_diff": len(retrieved_diff),
        "n_correct_diff": len(correct_diff),
        "answer_match_rate": (1 - len(answer_diff) / len(common)) if common else None,
        "mismatch_qids": sorted(set(answer_diff) | set(retrieved_diff) | set(correct_diff))[:10],
    }
