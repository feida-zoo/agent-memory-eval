#!/usr/bin/env python3
"""交付前自检：校验一个 membench 任务目录的完整性与内部一致性。

用法::

    python verify_artifacts.py DIR [--json] [--allow-limit] [--allow-partial]

检查项（任一失败 → 退出码 1）：

1. 任务目录命名与骨架：``eval_*`` 有 ``runs/`` + ``reports/``；``compare_*`` 有
   ``A/`` + ``B/`` + ``reports/``；``matrix_*`` 有 ``reports/`` + ``runs/``。
2. 每个组 / 实验目录下的每个数据集：``report/{metrics.json,results.jsonl,report.md}``
   齐备且非空，``metrics.json`` 可解析。
3. 一致性：数据集级 ``results.jsonl`` 行数 == 各对话 ``report/results.jsonl`` 行数
   之和；qid 集合无缺失、无重复；逐对话 ``n_questions`` 之和 == 数据集级
   ``overall.n_questions``。
4. 不许 partial 冒充完整：数据集级 ``metrics.json`` 若 ``shards > 1`` 且
   ``shard_index`` 不为 ``None``（未融合的分片）→ 失败，``--allow-partial`` 放行；
   ``limit`` 不为 ``None``（冒烟产物）→ 失败，``--allow-limit`` 放行。
5. 报告内链：任务内所有 ``report.md`` / ``comparison.md`` 里的相对 markdown 链接目标
   必须存在（跳过 ``http(s)://``、纯锚点，以及快照 / runtime 目录里的适配器文件）。
6. ``compare_*`` 任务：``reports/comparison.md`` 存在且含门禁结论关键词；``A/``、``B/``
   各至少有一个数据集级报告。
7. 运行时数据（**警告级**，不影响退出码）：任务目录缺 ``runtime/commands.jsonl`` 时
   提示"没有运行时数据，无法回溯本次执行"。
8. 未登记条目（**警告级**，不影响退出码）：对话目录 / 数据集目录 / 实验根里出现框架
   产物之外的条目（如历史上的 ``memory_snapshots/``）→ 提示疑似接入侧副产物。

``--json`` 输出机器可读结构（stdout 只有 JSON）；人类模式成功时 stdout 只有
``OK <task>``，失败时逐条 ``FAIL …``。本模块也被 ``run_task.py`` 直接 import 复用。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

#: 任务目录名前缀 → 任务类型。
_TASK_PREFIXES: tuple[tuple[str, str], ...] = (
    ("eval_", "eval"),
    ("compare_", "compare"),
    ("matrix_", "matrix"),
)

#: 数据集级 / 对话级报告目录与必需文件。
DIR_REPORT = "report"
DIR_REPORTS = "reports"
DIR_RUNS = "runs"
DIR_RUNTIME = "runtime"
DIR_SNAPSHOT = "memory_snapshot"
DIR_TRACE_STATE = "trace_state"
FILE_COMMANDS = "commands.jsonl"
FILE_SHARDS = "shards.json"
REPORT_FILES: tuple[str, ...] = ("metrics.json", "results.jsonl", "report.md")

#: 框架在对话目录 / 实验根自建或持有的条目；其余给「未登记」警告。
_DIALOGUE_ENTRIES: frozenset[str] = frozenset(
    {DIR_REPORT, DIR_RUNTIME, DIR_SNAPSHOT, DIR_TRACE_STATE}
)
#: 任务根里**与组无关**的框架条目；组目录名由 ``groups.json`` 给（可以是中文），
#: 校验时动态并入，不能再写死 ``A`` / ``B``。
_ROOT_ENTRIES: frozenset[str] = frozenset(
    {DIR_RUNS, DIR_REPORTS, DIR_RUNTIME, "snapshots", "groups.json", "task.json"}
)


def _task_group_dirs(task: str) -> list[str]:
    """任务根里的组目录名（读 ``groups.json``）；单组 / 旧布局返回空列表。"""
    path = os.path.join(task, "groups.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    groups = payload.get("groups") if isinstance(payload, dict) else None
    if not isinstance(groups, list):
        return []
    return [str(item["dir"]) for item in groups if isinstance(item, dict) and item.get("dir")]

#: 门禁结论关键词（comparison.md 必须含其一）。
GATE_KEYWORDS: tuple[str, ...] = ("门禁", "verdict")

#: markdown 链接 / 图片：[text](target) 或 ![alt](target)。
_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")

#: 不检查的链接 scheme（本地文件之外的东西）。
_SKIP_SCHEMES: tuple[str, ...] = (
    "http",
    "https",
    "mailto",
    "tel",
    "data",
    "javascript",
    "ftp",
)

#: 链接检查时跳过的目录（适配器可能在快照里放任意文件）。
_LINK_SKIP_DIRS: frozenset[str] = frozenset(
    {DIR_SNAPSHOT, "snapshots", DIR_RUNTIME, "__pycache__", ".git"}
)


@dataclass
class Issue:
    """一条自检问题（失败或提示）。"""

    check: str
    code: str
    message: str
    path: Optional[str] = None
    severity: str = "error"

    def as_dict(self) -> dict[str, Any]:
        """转成可 JSON 序列化的结构。"""
        return {
            "check": self.check,
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "severity": self.severity,
        }


@dataclass
class DatasetInfo:
    """一个数据集产物的摘要（JSON 的 ``datasets`` 段）。"""

    group: str
    dataset: str
    path: str
    n_questions: Optional[int] = None
    n_results: Optional[int] = None
    n_dialogues: Optional[int] = None

    def as_dict(self) -> dict[str, Any]:
        """转成可 JSON 序列化的结构。"""
        return {
            "group": self.group,
            "dataset": self.dataset,
            "path": self.path,
            "n_questions": self.n_questions,
            "n_results": self.n_results,
            "n_dialogues": self.n_dialogues,
        }


@dataclass
class Verification:
    """一次自检的完整结果。"""

    task: str
    kind: Optional[str]
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    datasets: list[DatasetInfo] = field(default_factory=list)
    allow_limit: bool = False
    allow_partial: bool = False

    @property
    def ok(self) -> bool:
        """没有任何失败项即通过。"""
        return not self.errors

    def fail(self, check: str, code: str, message: str, path: Optional[str] = None) -> None:
        """记录一条失败。"""
        self.errors.append(Issue(check, code, message, path, "error"))

    def warn(self, check: str, code: str, message: str, path: Optional[str] = None) -> None:
        """记录一条提示（不影响通过与否）。"""
        self.warnings.append(Issue(check, code, message, path, "warning"))

    def as_dict(self) -> dict[str, Any]:
        """转成 ``--json`` 的机器可读结构。"""
        return {
            "schema_version": 1,
            "task": self.task,
            "kind": self.kind,
            "ok": self.ok,
            "allow_limit": self.allow_limit,
            "allow_partial": self.allow_partial,
            "errors": [item.as_dict() for item in self.errors],
            "warnings": [item.as_dict() for item in self.warnings],
            "datasets": [item.as_dict() for item in self.datasets],
        }


def _detect_kind(task: str) -> Optional[str]:
    """按目录名判定任务类型；不认识返回 ``None``。"""
    name = os.path.basename(os.path.normpath(task))
    for prefix, kind in _TASK_PREFIXES:
        if name.startswith(prefix):
            return kind
    return None


def _is_non_empty_file(path: str) -> bool:
    """文件存在且字节数大于 0。"""
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _read_json(path: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """读 JSON 对象；失败时返回 ``(None, 可操作文案)``。"""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        return None, f"无法读取 {path}: {exc}"
    except ValueError as exc:
        return None, f"{path} 不是合法 JSON: {exc}"
    if not isinstance(data, dict):
        return None, f"{path} 顶层必须是 JSON 对象，实际为 {type(data).__name__}"
    return data, None


def _read_jsonl(path: str) -> tuple[list[dict[str, Any]], list[str], int]:
    """读 JSONL：返回 ``(合法记录, 错误文案, 非空行数)``。

    坏行不中断读取——自检要一次把问题列全；行数按非空行计，避免尾部空行误判。
    """
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    total = 0
    try:
        with open(path, encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                text = line.strip()
                if not text:
                    continue
                total += 1
                try:
                    record = json.loads(text)
                except ValueError as exc:
                    errors.append(f"{path}:{lineno} 不是合法 JSON: {exc}")
                    continue
                if not isinstance(record, dict):
                    errors.append(f"{path}:{lineno} 不是 JSON 对象")
                    continue
                records.append(record)
    except OSError as exc:
        errors.append(f"无法读取 {path}: {exc}")
    return records, errors, total


def _qid_of(record: dict[str, Any]) -> Optional[str]:
    """记录里的 qid（去空白）；缺失 / 空串返回 ``None``。"""
    value = record.get("qid")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _n_questions(metrics: dict[str, Any]) -> Optional[int]:
    """数据集 / 对话 metrics 的题目数：优先 ``overall.n_questions``，其次顶层。"""
    overall = metrics.get("overall")
    if isinstance(overall, dict):
        value = overall.get("n_questions")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    value = metrics.get("n_questions")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _dialogue_dirs(dataset_dir: str) -> list[str]:
    """数据集目录下的对话目录（排除数据集级 ``report/`` 与构建缓存目录）。"""
    try:
        names = sorted(os.listdir(dataset_dir))
    except OSError:
        return []
    return [
        os.path.join(dataset_dir, name)
        for name in names
        if name not in (DIR_REPORT, "__pycache__")
        and not name.startswith(".")
        and os.path.isdir(os.path.join(dataset_dir, name))
    ]


def _dataset_dirs(exp_root: str) -> list[str]:
    """实验根 ``runs/`` 下的数据集目录（按路径排序；跳过隐藏 / 缓存目录）。"""
    runs = os.path.join(exp_root, DIR_RUNS)
    try:
        names = sorted(os.listdir(runs))
    except OSError:
        return []
    return [
        os.path.join(runs, name)
        for name in names
        if name != "__pycache__"
        and not name.startswith(".")
        and os.path.isdir(os.path.join(runs, name))
    ]


def _entry_names(directory: str) -> list[str]:
    """目录下的条目名（跳过隐藏条目与构建缓存）。"""
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return []
    return [name for name in names if not name.startswith(".") and name != "__pycache__"]


def _is_dialogue_dir(path: str) -> bool:
    """对话目录判据：有 ``report/metrics.json``（数据集目录的 ``report/`` 是同级目录）。"""
    return os.path.isfile(os.path.join(path, DIR_REPORT, "metrics.json"))


def _warn_unknown_entry(
    result: Verification, code: str, where: str, path: str
) -> None:
    """未登记条目：只警告（多半是接入侧副产物），不影响通过与否。"""
    result.warn(
        "unknown",
        code,
        f"{where}：未登记条目 {os.path.basename(path)!r} 不是框架产物，"
        "疑似接入侧副产物（如历史上的 memory_snapshots/），确认后可删除",
        path,
    )


def _check_dialogue_entries(
    dialogue_dir: str, result: Verification, *, where: str
) -> None:
    """对话目录只应有 ``report/`` / ``runtime/`` / ``memory_snapshot/`` /
    ``trace_state/``（第 8 项；逐 as_of 追踪状态，interleaved + safety 自动开启）。"""
    for name in _entry_names(dialogue_dir):
        if name not in _DIALOGUE_ENTRIES:
            _warn_unknown_entry(
                result, "dialogue_entry", where, os.path.join(dialogue_dir, name)
            )


def _check_dataset_entries(
    dataset_dir: str, result: Verification, *, where: str
) -> None:
    """数据集目录只应有 ``report/`` / 分片账本 / 对话目录（第 8 项）。"""
    for name in _entry_names(dataset_dir):
        path = os.path.join(dataset_dir, name)
        if name in (DIR_REPORT, FILE_SHARDS):
            continue
        if name.startswith("shards.") and name.endswith(".json") and os.path.isfile(path):
            continue
        if os.path.isdir(path) and _is_dialogue_dir(path):
            continue
        _warn_unknown_entry(result, "dataset_entry", where, path)


def _check_root_entries(root: str, result: Verification, *, where: str) -> None:
    """实验 / 任务根只应有骨架目录（第 8 项）。"""
    for name in _entry_names(root):
        if name not in _ROOT_ENTRIES and name not in set(_task_group_dirs(root)):
            _warn_unknown_entry(
                result, "root_entry", where, os.path.join(root, name)
            )


def _check_required_files(
    directory: str, result: Verification, *, check: str, where: str
) -> None:
    """``<directory>/report`` 下三个必需文件齐备且非空。"""
    report_dir = os.path.join(directory, DIR_REPORT)
    if not os.path.isdir(report_dir):
        result.fail(check, "missing_report_dir", f"{where} 缺 {DIR_REPORT}/ 目录", report_dir)
        return
    for name in REPORT_FILES:
        path = os.path.join(report_dir, name)
        if not os.path.isfile(path):
            result.fail(check, f"missing_{name}", f"{where} 缺 {DIR_REPORT}/{name}", path)
        elif os.path.getsize(path) == 0:
            result.fail(check, f"empty_{name}", f"{where} 的 {DIR_REPORT}/{name} 是空文件", path)


def _check_partial_flags(
    metrics: dict[str, Any],
    result: Verification,
    *,
    check: str,
    metrics_path: str,
    allow_limit: bool,
    allow_partial: bool,
) -> None:
    """分片 / 冒烟产物不得被当成完整数据集报告（第 4 项）。"""
    shards = metrics.get("shards")
    shard_index = metrics.get("shard_index")
    if isinstance(shards, int) and not isinstance(shards, bool) and shards > 1:
        if shard_index is not None:
            message = (
                f"{metrics_path} 是未融合的分片产物（shards={shards}, "
                f"shard_index={shard_index}）：分片产物不能冒充完整数据集报告，"
                "请先运行 fuse --task-dir 补齐"
            )
            if allow_partial:
                result.warn(check, "partial_shard_allowed", message, metrics_path)
            else:
                result.fail(check, "partial_shard", message, metrics_path)
    limit = metrics.get("limit")
    if limit is not None and not allow_limit:
        result.fail(
            check,
            "limit_smoke",
            f"{metrics_path} 是冒烟产物（limit={limit}），不能当正式结论；"
            "确认无误请加 --allow-limit",
            metrics_path,
        )


def _check_dataset(
    dataset_dir: str,
    group: str,
    result: Verification,
    *,
    allow_limit: bool,
    allow_partial: bool,
) -> None:
    """校验单个数据集目录（第 2、3、4 项）。"""
    dataset_name = os.path.basename(os.path.normpath(dataset_dir))
    where = f"数据集 {dataset_name}" + (f"（组 {group}）" if group else "")
    _check_required_files(dataset_dir, result, check="dataset_files", where=where)

    metrics_path = os.path.join(dataset_dir, DIR_REPORT, "metrics.json")
    metrics: Optional[dict[str, Any]] = None
    if _is_non_empty_file(metrics_path):
        metrics, error = _read_json(metrics_path)
        if error is not None:
            result.fail("dataset_files", "invalid_metrics", error, metrics_path)
    if metrics is not None:
        _check_partial_flags(
            metrics,
            result,
            check="partial",
            metrics_path=metrics_path,
            allow_limit=allow_limit,
            allow_partial=allow_partial,
        )

    results_path = os.path.join(dataset_dir, DIR_REPORT, "results.jsonl")
    dataset_records: list[dict[str, Any]] = []
    dataset_lines = 0
    if os.path.isfile(results_path):
        dataset_records, parse_errors, dataset_lines = _read_jsonl(results_path)
        for message in parse_errors:
            result.fail("dataset_files", "invalid_results", message, results_path)

    dataset_qids: list[str] = []
    for record in dataset_records:
        qid = _qid_of(record)
        if qid is None:
            result.fail(
                "consistency",
                "missing_qid",
                f"{results_path} 有记录缺 qid（qid 必需且非空）",
                results_path,
            )
            continue
        dataset_qids.append(qid)

    dialogue_dirs = _dialogue_dirs(dataset_dir)
    _check_dataset_entries(dataset_dir, result, where=where)
    if not dialogue_dirs:
        result.fail(
            "consistency",
            "no_dialogues",
            f"{where} 下没有对话目录（数据集级报告不能脱离对话级产物存在）",
            dataset_dir,
        )
    dialogue_qids: list[str] = []
    dialogue_lines = 0
    dialogue_n_sum = 0
    dialogue_n_known = True if dialogue_dirs else False
    for dialogue_dir in dialogue_dirs:
        dialogue_id = os.path.basename(dialogue_dir)
        _check_required_files(
            dialogue_dir,
            result,
            check="dialogue_files",
            where=f"对话 {dialogue_id}" + (f"（组 {group}）" if group else ""),
        )
        _check_dialogue_entries(
            dialogue_dir,
            result,
            where=f"对话 {dialogue_id}" + (f"（组 {group}）" if group else ""),
        )
        dialogue_metrics_path = os.path.join(dialogue_dir, DIR_REPORT, "metrics.json")
        dialogue_metrics: Optional[dict[str, Any]] = None
        if _is_non_empty_file(dialogue_metrics_path):
            dialogue_metrics, error = _read_json(dialogue_metrics_path)
            if error is not None:
                result.fail(
                    "dialogue_files", "invalid_dialogue_metrics", error, dialogue_metrics_path
                )
        dialogue_results_path = os.path.join(dialogue_dir, DIR_REPORT, "results.jsonl")
        if os.path.isfile(dialogue_results_path):
            records, parse_errors, lines = _read_jsonl(dialogue_results_path)
            dialogue_lines += lines
            for message in parse_errors:
                result.fail(
                    "dialogue_files", "invalid_dialogue_results", message, dialogue_results_path
                )
            for record in records:
                qid = _qid_of(record)
                if qid is None:
                    result.fail(
                        "consistency",
                        "missing_qid",
                        f"{dialogue_results_path} 有记录缺 qid（qid 必需且非空）",
                        dialogue_results_path,
                    )
                    continue
                dialogue_qids.append(qid)
        if dialogue_metrics is not None:
            n_dialogue = _n_questions(dialogue_metrics)
            if n_dialogue is None:
                dialogue_n_known = False
                result.fail(
                    "consistency",
                    "missing_dialogue_n_questions",
                    f"{dialogue_metrics_path} 缺 overall.n_questions，无法核对题数之和",
                    dialogue_metrics_path,
                )
            else:
                dialogue_n_sum += n_dialogue
        else:
            dialogue_n_known = False

    info = DatasetInfo(
        group=group,
        dataset=dataset_name,
        path=os.path.abspath(dataset_dir),
        n_questions=_n_questions(metrics) if metrics is not None else None,
        n_results=dataset_lines if os.path.isfile(results_path) else None,
        n_dialogues=len(dialogue_dirs),
    )
    result.datasets.append(info)

    # 3a. 行数：数据集级 == 各对话之和
    if os.path.isfile(results_path):
        if dataset_lines != dialogue_lines:
            result.fail(
                "consistency",
                "results_count_mismatch",
                f"{where}：数据集级 results.jsonl {dataset_lines} 行 != "
                f"各对话 results.jsonl 之和 {dialogue_lines} 行",
                dataset_dir,
            )

    # 3b. 题数：数据集级 == 逐对话 n_questions 之和
    dataset_n = _n_questions(metrics) if metrics is not None else None
    if dataset_n is None:
        if metrics is not None:
            result.fail(
                "consistency",
                "missing_n_questions",
                f"{metrics_path} 缺 overall.n_questions，无法核对题数",
                metrics_path,
            )
    elif dialogue_n_known and dataset_n != dialogue_n_sum:
        result.fail(
            "consistency",
            "n_questions_mismatch",
            f"{where}：数据集级 n_questions={dataset_n} != 逐对话之和 {dialogue_n_sum}",
            dataset_dir,
        )

    # 3c. qid：无重复、数据集级与对话级集合一致
    seen_dataset: set[str] = set()
    duplicates_dataset: list[str] = []
    for qid in dataset_qids:
        if qid in seen_dataset:
            duplicates_dataset.append(qid)
        seen_dataset.add(qid)
    if duplicates_dataset:
        result.fail(
            "consistency",
            "duplicate_qid",
            f"{where}：数据集级 results.jsonl 有重复 qid: {_brief(duplicates_dataset)}",
            results_path,
        )

    seen_dialogue: set[str] = set()
    duplicates_dialogue: list[str] = []
    for qid in dialogue_qids:
        if qid in seen_dialogue:
            duplicates_dialogue.append(qid)
        seen_dialogue.add(qid)
    if duplicates_dialogue:
        result.fail(
            "consistency",
            "duplicate_qid",
            f"{where}：对话级 results.jsonl 有重复 qid: {_brief(duplicates_dialogue)}",
            dataset_dir,
        )

    if dataset_qids or dialogue_qids:
        missing = sorted(set(dataset_qids) - set(dialogue_qids))
        extra = sorted(set(dialogue_qids) - set(dataset_qids))
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"数据集级有、对话级缺 {_brief(missing)}")
            if extra:
                parts.append(f"对话级有、数据集级缺 {_brief(extra)}")
            result.fail(
                "consistency",
                "qid_set_mismatch",
                f"{where}：qid 集合不一致（{'；'.join(parts)}）",
                dataset_dir,
            )


def _brief(items: Iterable[str], limit: int = 5) -> str:
    """列表的简短展示（最多 ``limit`` 项）。"""
    values = [str(item) for item in items]
    shown = "、".join(values[:limit])
    if len(values) > limit:
        shown += f" …（共 {len(values)} 项）"
    return shown


def _check_link(
    raw: str, md_path: str, result: Verification
) -> None:
    """检查 markdown 里的一个目标；只检查相对本地路径。"""
    target = raw.strip()
    if not target:
        return
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")].strip()
    elif " " in target:
        # 兼容 `path "title"`：优先按完整目标判断，不存在再取第一段
        candidate = target.strip()
        full = os.path.normpath(os.path.join(os.path.dirname(md_path), candidate))
        if not os.path.exists(full):
            target = candidate.split()[0]
    if target.startswith("#"):
        return
    lowered = target.lower()
    if lowered.startswith("//"):
        return
    scheme = lowered.split(":", 1)[0]
    if ":" in lowered and scheme in _SKIP_SCHEMES:
        return
    if os.path.isabs(target):
        return
    local = target.split("#", 1)[0].split("?", 1)[0]
    if not local:
        return
    local = _unquote(local)
    resolved = os.path.normpath(os.path.join(os.path.dirname(md_path), local))
    if not os.path.exists(resolved):
        result.fail(
            "links",
            "dead_link",
            f"报告内链死链: {md_path} -> {target}（解析为 {resolved}）",
            md_path,
        )


def _unquote(text: str) -> str:
    """URL 解码（``%20`` 等）；解码失败时原样返回。"""
    try:
        from urllib.parse import unquote

        return unquote(text)
    except Exception:  # pragma: no cover - 标准库不应失败
        return text


def _check_links(task: str, result: Verification) -> None:
    """扫描任务内所有 ``report.md`` / ``comparison.md`` 的相对链接（第 5 项）。

    快照 / runtime 目录整棵跳过：适配器可以在里面放任意文件（包括同名 ``report.md``），
    那些不是框架报告，检查它们只会误报。
    """
    for root, dirs, files in os.walk(task):
        dirs[:] = [name for name in dirs if name not in _LINK_SKIP_DIRS]
        for name in files:
            if name not in ("report.md", "comparison.md"):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, encoding="utf-8") as handle:
                    text = handle.read()
            except OSError as exc:
                result.fail("links", "unreadable_report", f"无法读取报告 {path}: {exc}", path)
                continue
            for match in _LINK_RE.finditer(text):
                _check_link(match.group(1), path, result)


def _check_compare(task: str, result: Verification) -> None:
    """compare 任务专属检查（第 6 项）。"""
    report_path = os.path.join(task, DIR_REPORTS, "comparison.md")
    if not os.path.isfile(report_path) or os.path.getsize(report_path) == 0:
        result.fail(
            "compare",
            "missing_comparison",
            f"compare 任务缺 {DIR_REPORTS}/comparison.md（对比报告未生成）",
            report_path,
        )
    else:
        try:
            with open(report_path, encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:
            text = ""
            result.fail(
                "compare",
                "unreadable_comparison",
                f"无法读取 {report_path}: {exc}",
                report_path,
            )
        if text and not any(word.lower() in text.lower() for word in GATE_KEYWORDS):
            result.fail(
                "compare",
                "missing_gate_conclusion",
                f"{report_path} 里没有门禁结论关键词（{' / '.join(GATE_KEYWORDS)}）",
                report_path,
            )
    for group in _task_group_dirs(task):
        group_path = os.path.join(task, group)
        if not os.path.isdir(group_path):
            continue  # 骨架检查已报
        datasets = _dataset_dirs(group_path)
        if not datasets:
            result.fail(
                "compare",
                "missing_group_dataset",
                f"compare 任务的组 {group}/ 下没有数据集级报告",
                group_path,
            )


def _experiment_roots(task: str, kind: str) -> list[tuple[str, str]]:
    """任务下的实验根：``[(根目录, 组标签)]``；matrix 的 label 本身是数据集目录。"""
    if kind == "eval":
        return [(task, "")]
    if kind == "compare":
        return [
            (os.path.join(task, group), group)
            for group in _task_group_dirs(task)
            if os.path.isdir(os.path.join(task, group))
        ]
    # matrix: runs/<label>/ 是数据集树本身
    runs = os.path.join(task, DIR_RUNS)
    try:
        names = sorted(os.listdir(runs))
    except OSError:
        return []
    return [
        (os.path.join(runs, name), name)
        for name in names
        if os.path.isdir(os.path.join(runs, name))
    ]


def verify_task(
    task_dir: str,
    *,
    allow_limit: bool = False,
    allow_partial: bool = False,
) -> Verification:
    """校验任务目录，返回 :class:`Verification`（不打印、不退出）。

    Args:
        task_dir: 任务目录（``eval_*`` / ``compare_*`` / ``matrix_*``）。
        allow_limit: 放行 ``limit`` 冒烟产物（只影响第 4 项）。
        allow_partial: 放行未融合分片产物（只影响第 4 项的分片判定）。
    """
    task = os.path.abspath(task_dir)
    result = Verification(
        task=task,
        kind=_detect_kind(task),
        allow_limit=allow_limit,
        allow_partial=allow_partial,
    )
    if not os.path.isdir(task):
        result.fail("skeleton", "not_a_directory", f"任务目录不存在: {task}", task)
        return result
    kind = result.kind
    if kind is None:
        result.fail(
            "skeleton",
            "unknown_task_dir",
            "目录名需以 eval_ / compare_ / matrix_ 开头（membench 任务目录约定）",
            task,
        )
        return result

    # 1. 骨架
    skeleton_ok = True
    if kind == "eval":
        for name in (DIR_RUNS, DIR_REPORTS):
            path = os.path.join(task, name)
            if not os.path.isdir(path):
                result.fail("skeleton", f"missing_{name}", f"eval 任务目录缺 {name}/", path)
                skeleton_ok = False
    elif kind == "compare":
        for name in (*_task_group_dirs(task), DIR_REPORTS):
            path = os.path.join(task, name)
            if not os.path.isdir(path):
                result.fail("skeleton", f"missing_{name}", f"compare 任务目录缺 {name}/", path)
                skeleton_ok = False
    else:  # matrix
        for name in (DIR_REPORTS, DIR_RUNS):
            path = os.path.join(task, name)
            if not os.path.isdir(path):
                result.fail("skeleton", f"missing_{name}", f"matrix 任务目录缺 {name}/", path)
                skeleton_ok = False

    if not skeleton_ok:
        return result

    # 7. 运行时数据（警告级）：没有它就不能回溯"这次评测是怎么跑出来的"
    commands = os.path.join(task, DIR_RUNTIME, FILE_COMMANDS)
    if not os.path.isfile(commands):
        result.warn(
            "runtime",
            "missing_runtime",
            "没有运行时数据（缺 runtime/commands.jsonl），无法回溯本次执行",
            commands,
        )

    # 8. 未登记条目（警告级）：框架产物之外的东西，多半是接入侧副产物
    _check_root_entries(task, result, where="任务目录")

    # 2/3/4. 每个组 / 实验目录下的数据集
    roots = _experiment_roots(task, kind)
    if not roots:
        result.fail("skeleton", "no_experiment", "任务目录里没有任何实验根（A/B 或 runs/<label>）", task)
        return result
    for exp_root, group in roots:
        if kind in ("eval", "compare"):
            if os.path.normpath(exp_root) != task:
                _check_root_entries(exp_root, result, where=f"组目录 {group}")
            datasets = _dataset_dirs(exp_root)
            if not datasets:
                result.fail(
                    "dataset_files",
                    "no_datasets",
                    f"实验目录 {exp_root} 的 runs/ 下没有数据集产物",
                    exp_root,
                )
                continue
            for dataset_dir in datasets:
                if not os.path.isdir(os.path.join(dataset_dir, DIR_REPORT)):
                    hint = ""
                    if _is_unfused_shard(dataset_dir):
                        hint = "；这是未融合的分片产物，请先运行 fuse --task-dir"
                    result.fail(
                        "dataset_files",
                        "missing_report_dir",
                        f"数据集目录缺 {DIR_REPORT}/（半成品或未融合分片）{hint}",
                        dataset_dir,
                    )
                    continue
                _check_dataset(
                    dataset_dir,
                    group,
                    result,
                    allow_limit=allow_limit,
                    allow_partial=allow_partial,
                )
        else:  # matrix: exp_root 本身是数据集目录
            if not os.path.isdir(os.path.join(exp_root, DIR_REPORT)):
                result.fail(
                    "dataset_files",
                    "missing_report_dir",
                    f"matrix label 目录缺 {DIR_REPORT}/（半成品）",
                    exp_root,
                )
                continue
            _check_dataset(
                exp_root,
                group,
                result,
                allow_limit=allow_limit,
                allow_partial=allow_partial,
            )

    # 5. 报告内链
    _check_links(task, result)

    # 6. compare 专属
    if kind == "compare":
        _check_compare(task, result)

    return result


def _is_unfused_shard(dataset_dir: str) -> bool:
    """数据集目录里是否有分片账本（``shards.json`` / ``shards.<i>.json``）。"""
    if os.path.isfile(os.path.join(dataset_dir, "shards.json")):
        return True
    try:
        return any(
            name.startswith("shards.") and name.endswith(".json")
            for name in os.listdir(dataset_dir)
        )
    except OSError:
        return False


def _print_human(result: Verification) -> None:
    """人类可读输出：成功 ``OK <task>``，失败逐条 ``FAIL …``。"""
    for item in result.warnings:
        print(f"WARN [{item.check}/{item.code}] {item.message}", file=sys.stderr)
    if result.ok:
        print(f"OK {result.task}")
        return
    for item in result.errors:
        location = f"（{item.path}）" if item.path else ""
        print(f"FAIL [{item.check}/{item.code}] {item.message}{location}")
    print(f"FAIL {result.task}（{len(result.errors)} 项问题）")


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="verify_artifacts.py",
        description=(
            "交付前自检：校验 membench 任务目录的骨架、数据集报告、对话一致性、"
            "partial 标记与报告内链。"
        ),
        epilog=(
            "退出码: 0=通过；1=有失败项。"
            " --allow-limit 放行 limit 冒烟产物；--allow-partial 放行未融合分片产物。"
        ),
    )
    parser.add_argument("dir", metavar="DIR", help="任务目录（eval_* / compare_* / matrix_*）")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON（stdout 只有 JSON）")
    parser.add_argument(
        "--allow-limit",
        action="store_true",
        help="允许 limit 冒烟产物（默认把 limit != None 当作不能冒充正式结论）",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="允许未融合的分片产物（shards>1 且 shard_index != None；默认失败）",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI 入口：返回 0（通过）/ 1（失败）。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    result = verify_task(
        args.dir,
        allow_limit=args.allow_limit,
        allow_partial=args.allow_partial,
    )
    if args.json:
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    else:
        _print_human(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
