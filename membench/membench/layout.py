"""实验产物布局的唯一入口：目录名 / 报告文件名常量与路径 helper。

终态结构（报告与数据就近归属，一次实验一个自包含子目录）::

    <root>/eval_<name>_<platform>_<ts>/
    ├── runs/<dataset>/
    │   ├── report/                      # 数据集级聚合报告（md + metrics + 逐题明细 + 图）
    │   └── <dialogue_id>/
    │       ├── runtime/                 # 运行时数据（如桥的 reset 前备份）
    │       ├── memory_snapshot/         # --snapshot 导出的对话快照（含 manifest.json）
    │       └── report/                  # 对话级报告（纯文字，不出图）
    └── reports/                         # 任务级报告（单数据集 run = 数据集级报告内容）

    <root>/compare_<name>_<platform>_<ts>/   A/ B/（各为完整 eval 实验形态）+ reports/
    <root>/matrix_<ts>/                      reports/ + runs/<label>/（内部同 eval 树）

``<root>`` 由 :func:`membench.paths.output_root` 解析（CLI ``--report-out DIR`` 即指它）。

本模块是这些目录名与报告文件名的**唯一来源**：其它模块拼路径一律调用这里的常量与
helper，不要手写字面量。helper 只计算路径、不建目录——需要落盘时调用方自行
``os.makedirs(..., exist_ok=True)``。读取侧把「产物目录」（实验目录或数据集目录）归一成
数据集目录也只走这里的 :func:`resolve_dataset_dir`。
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime

from membench.paths import output_root

#: 实验目录时间戳格式。
TS_FORMAT = "%Y%m%d-%H%M%S"

#: 实验目录下的各层目录名。
DIR_RUNS = "runs"
DIR_REPORT = "report"  # 对话级 / 数据集级报告目录
DIR_REPORTS = "reports"  # 实验根层的任务级报告目录（与 runs/ 平级）
DIR_RUNTIME = "runtime"
DIR_SNAPSHOT = "memory_snapshot"

#: 报告与数据文件名。
FILE_REPORT_MD = "report.md"
FILE_COMPARISON_MD = "comparison.md"
FILE_METRICS = "metrics.json"
FILE_RESULTS = "results.jsonl"
FILE_SUMMARY = "evaluate_summary.json"
FILE_MATRIX_RESULT = "matrix_result.json"

#: 图表文件名（报告内引用一律同目录相对文件名）。
FIG_CATEGORY = "fig_category_accuracy.png"
FIG_LATENCY = "fig_latency_hist.png"
FIG_METRIC_DIFF = "fig_metric_diff.png"
FIG_CATEGORY_DIFF = "fig_category_diff.png"
FIG_DATASET_OVERVIEW = "fig_dataset_overview.png"
FIG_MATRIX_ACCURACY = "fig_matrix_accuracy.png"

#: 实验目录名安全化：非字母数字与 ``._-`` 的字符替换为 ``-``。
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")

#: 对话目录名安全化：非字母数字与 ``._-`` 的字符替换为 ``_``。
_SAFE_TAG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def ts_now(now: datetime | None = None) -> str:
    """当前时间戳字符串（:data:`TS_FORMAT`）。"""
    return (now or datetime.now()).strftime(TS_FORMAT)


def slug(text: str) -> str:
    """把任意标签转成安全的目录名片段（空串回退为 ``default``）。"""
    cleaned = _UNSAFE_RE.sub("-", str(text)).strip("-")
    return cleaned or "default"


def tag_dir_name(tag: str) -> str:
    """把对话 id 归一成安全的目录名（防路径穿越；空值兜底 ``dialogue``）。"""
    safe = _SAFE_TAG_RE.sub("_", str(tag)).strip("._-")
    return (safe or "dialogue")[:120]


def eval_subdir(name: str, platform: str | None = None, ts: str | None = None) -> str:
    """单次评测实验目录名：``eval_<name>_<platform>_<ts>``。"""
    return f"eval_{slug(name)}_{slug(platform or 'default')}_{ts or ts_now()}"


def compare_subdir(name: str, platform: str | None = None, ts: str | None = None) -> str:
    """A/B 对比实验目录名：``compare_<name>_<platform>_<ts>``。"""
    return f"compare_{slug(name)}_{slug(platform or 'default')}_{ts or ts_now()}"


def matrix_subdir(ts: str | None = None) -> str:
    """矩阵对比实验目录名：``matrix_<ts>``。"""
    return f"matrix_{ts or ts_now()}"


def experiment_path(base: str | None, name: str) -> str:
    """在 ``base``（默认 :func:`membench.paths.output_root`）下建实验目录并返回。

    目标已存在且**非空**时拒绝新建：实验目录名只精确到秒，同一秒内的两次评测若落到
    同一目录，后一次会静默覆盖前一次的数据集级报告。上一次跑在校验阶段就失败留下的
    空目录可以复用，不算冲突。
    """
    path = os.path.join(base or output_root(), name)
    if os.path.isdir(path) and os.listdir(path):
        raise SystemExit(
            f"实验目录已存在且非空: {path}；"
            "同一秒内重复运行请换 --report-out，或稍后重试"
        )
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 实验目录内的路径
# ---------------------------------------------------------------------------

def runs_root(experiment_dir: str) -> str:
    """实验目录下的 ``runs/``。"""
    return os.path.join(experiment_dir, DIR_RUNS)


def dataset_dir(experiment_dir: str, dataset: str) -> str:
    """数据集产物目录：``<exp>/runs/<slug(dataset)>``。"""
    return os.path.join(runs_root(experiment_dir), slug(dataset))


def dataset_report_dir(experiment_dir: str, dataset: str) -> str:
    """数据集级报告目录：``<exp>/runs/<dataset>/report``。"""
    return os.path.join(dataset_dir(experiment_dir, dataset), DIR_REPORT)


def dialogue_dir(experiment_dir: str, dataset: str, dialogue_id: str) -> str:
    """对话目录：``<exp>/runs/<dataset>/<tag_dir_name(dialogue_id)>``。"""
    return os.path.join(dataset_dir(experiment_dir, dataset), tag_dir_name(dialogue_id))


def dialogue_report_dir(experiment_dir: str, dataset: str, dialogue_id: str) -> str:
    """对话级报告目录。"""
    return os.path.join(dialogue_dir(experiment_dir, dataset, dialogue_id), DIR_REPORT)


def dialogue_runtime_dir(experiment_dir: str, dataset: str, dialogue_id: str) -> str:
    """对话级运行时数据目录（如桥的 reset 前备份落 ``runtime/reset_backup/``）。"""
    return os.path.join(dialogue_dir(experiment_dir, dataset, dialogue_id), DIR_RUNTIME)


def dialogue_snapshot_dir(experiment_dir: str, dataset: str, dialogue_id: str) -> str:
    """对话级快照目录（``--snapshot`` 时含 ``manifest.json``）。"""
    return os.path.join(dialogue_dir(experiment_dir, dataset, dialogue_id), DIR_SNAPSHOT)


def task_report_dir(experiment_dir: str) -> str:
    """实验根层的任务级报告目录：``<exp>/reports``。"""
    return os.path.join(experiment_dir, DIR_REPORTS)


# ---------------------------------------------------------------------------
# 产物目录解析（读取方的唯一入口）
# ---------------------------------------------------------------------------

#: 旧版产物（平铺 run 目录、缺 ``report/`` 层级，或旧的 ``ab_`` 实验目录）的统一报错文案。
LEGACY_LAYOUT_MSG = "产物结构为旧版（缺少 report/ 层级），本版本不再兼容，请重新跑评测"


def _dialogue_id_of_report(path: str) -> str | None:
    """读 ``<dir>/report/metrics.json`` 落账的 ``dialogue_id``（没有 / 读不出返 None）。

    对话目录与数据集目录都带 ``report/`` 层级，只有 ``dialogue_id`` 能把两者分开：
    数据集级 metrics 不写该字段，对话级聚合写。读不出 / 解析失败按"不是对话目录"
    处理——文件本身的问题由 :func:`_require_metrics` 报错。
    """
    metrics_path = os.path.join(path, DIR_REPORT, FILE_METRICS)
    try:
        with open(metrics_path, encoding="utf-8") as handle:
            metrics = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(metrics, dict):
        return None
    dialogue_id = metrics.get("dialogue_id")
    return str(dialogue_id) if dialogue_id else None


def _is_dialogue_dir(path: str) -> bool:
    """对话级产物目录的判定：``report/metrics.json`` 里落账了 ``dialogue_id``。"""
    return _dialogue_id_of_report(path) is not None


def _is_dataset_dir(path: str) -> bool:
    """数据集目录的判定：带 ``report/`` 层级且不是对话级产物目录。

    ``report/`` 下文件是否齐备由 :func:`_require_metrics` 校验。
    """
    return os.path.isdir(os.path.join(path, DIR_REPORT)) and not _is_dialogue_dir(path)


def _require_metrics(dataset_dir: str) -> str:
    """数据集目录必须落账 ``report/metrics.json``。

    半成品产物（跑到一半被中断）要在这里就被拦下，不能等读取方打开文件时才抛错。
    """
    if not os.path.isfile(os.path.join(dataset_dir, DIR_REPORT, FILE_METRICS)):
        raise SystemExit(f"数据集目录缺少 {FILE_METRICS}（产物不完整）: {dataset_dir}")
    return dataset_dir


def resolve_dataset_dir(target: str, dataset: str | None = None) -> str:
    """把「产物目录」入参归一成数据集目录（``<实验目录>/runs/<dataset>``）。

    读取侧（compare / fuse / matrix / evaluate 汇总）统一用本函数解析入参：

    - **数据集目录**（自带 ``report/`` 层级）原样返回；
    - **实验目录**按 ``dataset``（数据集 key，目录名经 :func:`slug` 归一后比对）命中，
      否则取唯一的数据集子目录；有多个而无法定位时报错，要求直接传数据集目录；
    - **对话目录**（``report/metrics.json`` 里落账了 ``dialogue_id``）一律拒绝：
      一段对话的指标不是数据集汇总，拿它当数据集会得到以偏概全的结论；
    - 两者都不是（缺 ``report/`` 层级）→ 按旧版产物报错（:data:`LEGACY_LAYOUT_MSG`）。

    Args:
        target: 数据集目录或实验目录。
        dataset: 数据集 key，用于在实验目录下优先定位（默认 None）。

    Returns:
        数据集目录的绝对路径。

    Raises:
        SystemExit: 入参目录不存在、入参是对话级产物目录、数据集目录缺
            ``report/metrics.json``（产物不完整）、实验目录下有多个数据集目录而
            调用方未指定，或入参是旧版产物结构。
    """
    if not os.path.isdir(target):
        raise SystemExit(f"产物目录不存在: {target}")
    abspath = os.path.abspath(target)
    dialogue_id = _dialogue_id_of_report(abspath)
    if dialogue_id is not None:
        raise SystemExit(
            f"入参是对话级产物目录（对话 {dialogue_id}），请传实验目录或数据集目录: {abspath}"
        )
    if _is_dataset_dir(abspath):
        return _require_metrics(abspath)

    runs = runs_root(abspath)
    if os.path.isdir(runs):
        if dataset:
            candidate = dataset_dir(abspath, dataset)
            if _is_dataset_dir(candidate):
                return _require_metrics(candidate)
        found = sorted(
            os.path.join(runs, name)
            for name in os.listdir(runs)
            if _is_dataset_dir(os.path.join(runs, name))
        )
        if len(found) == 1:
            return _require_metrics(found[0])
        if len(found) > 1:
            names = "、".join(os.path.basename(path) for path in found)
            raise SystemExit(
                f"实验目录 {abspath} 下有多个数据集目录（{names}），无法确定读哪一个；"
                f"请把入参指向具体的数据集目录 {runs}/<dataset>"
            )
    raise SystemExit(f"{LEGACY_LAYOUT_MSG}：{abspath}")
