"""实验产物布局的唯一入口：目录名 / 报告文件名常量与路径 helper。

终态结构（报告与数据就近归属，一次实验一个自包含子目录）::

    <root>/eval_<name>_<platform>_<ts>[_s<i>][_l<limit>]/
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

分片 / 冒烟标签（``_s<i>`` / ``_l<limit>``）只加在目录名末尾，见 :func:`experiment_tags`；
评估报告里的 ``run_id`` 用另一套更完整的词表（含分片总数、分片维度与融合标记），
由 :func:`run_id` 组装、词表见 :func:`run_id_tags`——普通 run 与 fuse 融合产物都走它。

本模块是这些目录名与报告文件名的**唯一来源**：其它模块拼路径一律调用这里的常量与
helper，不要手写字面量。helper 只计算路径、不建目录——需要落盘时调用方自行
``os.makedirs(..., exist_ok=True)``。读取侧把「产物目录」（实验目录或数据集目录）归一成
数据集目录也只走这里的 :func:`resolve_dataset_dir`。
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
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

#: A/B 对比实验目录（compare / evaluate-compare）下两侧副本的目录名：A = 基线、B = 候选。
#: 兼作报告里的侧标签。
DIR_ARM_A = "A"
DIR_ARM_B = "B"

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
FIG_PER_DATASET_DIFF = "fig_per_dataset_diff.png"
FIG_FUSED_DIFF = "fig_fused_diff.png"

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


def eval_subdir(
    name: str,
    platform: str | None = None,
    ts: str | None = None,
    tags: Iterable[str] = (),
) -> str:
    """单次评测实验目录名：``eval_<name>_<platform>_<ts>[_<tag>...]``。

    ``tags`` 依次以 ``_<tag>`` 追加（每个 tag 经 :func:`slug` 归一）；空序列时
    与历史形态逐字节相同。分片 / 冒烟场景传 :func:`experiment_tags` 的结果。
    """
    suffix = "".join(f"_{slug(tag)}" for tag in tags)
    return f"eval_{slug(name)}_{slug(platform or 'default')}_{ts or ts_now()}{suffix}"


def run_id_tags(
    shards: int,
    shard_index: int | None,
    shard_by: str,
    limit: int | None = None,
) -> list[str]:
    """``run_id`` 里的分片 / 冒烟 / 融合标记词表（run_id 的字符串格式由此处唯一决定）。

    ``shards`` 传**解析后**的有效分片数（``--shards max`` 由 Runner 按数据集上限
    结算），``<= 1`` 时不出分片标记；``shard_by="dialogue"`` 用 ``dshard`` 前缀与
    题目级分片区分。``shard_index=None`` 表示这不是某一分片而是**整体融合产物**
    （fuse 把 0..N-1 各片合并成一份），此时出 ``shard<N>`` / ``dshard<N>``（只记
    片数、不记序号）——冒烟（``limitN``）、分片与融合结果都不能被误当全量结果。
    """
    tags: list[str] = []
    if shards > 1:
        shard_mode = "dshard" if shard_by == "dialogue" else "shard"
        if shard_index is None:
            tags.append(f"{shard_mode}{shards}")
        else:
            tags.append(f"{shard_mode}{shard_index}of{shards}")
    if limit is not None:
        tags.append(f"limit{limit}")
    return tags


def run_id(
    adapter: str,
    platform: str | None,
    dataset: str,
    ts: str,
    *,
    commit: str | None = None,
    tags: Iterable[str] = (),
) -> str:
    """run 身份字符串：``<adapter>_<platform>[_<commit 前 8 位>]_<dataset>[_<tag>...]_<ts>``。

    本函数是 run_id 格式的**唯一来源**——Runner（普通 run）与 fuse（融合产物）都走它，
    两边不会再各自手写词表。``ts`` 由调用方给出，普通 run 用 ``%Y%m%d_%H%M%S``；
    ``commit`` 取前 8 位，避免矩阵评测中同 adapter + platform 同秒撞身份；``tags``
    传 :func:`run_id_tags` 的结果（分片 / 冒烟 / 融合标记），保证冒烟、分片与融合
    产物都不会与全量产物撞成同一身份。
    """
    commit_tag = f"_{str(commit)[:8]}" if commit else ""
    tag_text = "".join(f"_{tag}" for tag in tags)
    return f"{adapter}_{platform or 'default'}{commit_tag}_{dataset}{tag_text}_{ts}"


def experiment_tags(shards: int | str, shard_index: int, limit: int | None = None) -> list[str]:
    """实验目录名末尾的分片 / 冒烟标签：分片 ``s<i>``、冒烟 ``l<limit>``。

    目录名只精确到秒，同一秒并行的两个分片否则会撞进同一目录被非空守卫拒绝，
    因此把分片序号写进目录名——两片可以共用同一个 ``--report-out``。只记序号不记
    分片总数：总数要解析数据集才知道（``--shards max``），且 ``s<i>`` 已足以区分
    同一次评测的各片。不分片且无 limit 时返回空列表，目录名保持
    ``eval_<name>_<platform>_<ts>`` 原样。

    已知取舍：``--shards max`` 在**单分片数据集**上实际只跑 1 片，但分片数要解析
    数据集才知道，CLI 建目录时已按"传了 --shards"记下 ``s<i>``，故目录名仍带
    ``_s0``——宁可多带标记，也不能让分片产物与不分片产物撞进同一目录名。
    """
    tags: list[str] = []
    if shards != 1:
        tags.append(f"s{shard_index}")
    if limit is not None:
        tags.append(f"l{limit}")
    return tags


def compare_subdir(name: str, platform: str | None = None, ts: str | None = None) -> str:
    """A/B 对比实验目录名：``compare_<name>_<platform>_<ts>``。"""
    return f"compare_{slug(name)}_{slug(platform or 'default')}_{ts or ts_now()}"


def matrix_subdir(ts: str | None = None) -> str:
    """矩阵对比实验目录名：``matrix_<ts>``。"""
    return f"matrix_{ts or ts_now()}"


def experiment_path(base: str | None, name: str) -> str:
    """在 ``base``（默认 :func:`membench.paths.output_root`）下建实验目录并返回。

    目标已存在且**非空**时拒绝新建：实验目录名只精确到秒且不含完整身份，同一秒内
    同名的两次评测若落到同一目录，后一次会静默覆盖前一次的数据集级报告。并行分片的
    目录名带 ``_s<i>``（见 :func:`experiment_tags`），因此各片不会撞进同一目录。上一次
    跑在校验阶段就失败留下的空目录可以复用，不算冲突。
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

#: 实验目录里没有任何可用数据集产物的统一报错文案（空目录、跑到一半中断的目录）。
#: 与 :data:`LEGACY_LAYOUT_MSG` 分开：那种结构是**旧版本**写的，这种只是**没有产物**，
#: 报错要说清真实原因，否则用户会去重跑评测而不是换一个目录。
NO_DATASET_PRODUCT_MSG = (
    "实验目录下没有任何数据集产物（缺 runs/<dataset>/report/），无法读取"
)

#: 旧版产物落在 ``runs/`` 层（平铺 run）或实验目录根时的特征文件。
_LEGACY_PRODUCT_FILES: tuple[str, ...] = (
    FILE_METRICS,
    FILE_RESULTS,
    FILE_REPORT_MD,
    "compare_report.md",  # 旧版对比报告（新的叫 comparison.md）
)


def _looks_like_legacy_product(path: str) -> bool:
    """旧版产物形态判定：报告 / 指标直接落在目录根（平铺 run），或旧版 A/B 实验目录
    （``A/`` / ``B/`` 各是一份平铺 run）。"""
    if any(os.path.isfile(os.path.join(path, name)) for name in _LEGACY_PRODUCT_FILES):
        return True
    return any(
        os.path.isfile(os.path.join(path, arm, FILE_METRICS))
        for arm in (DIR_ARM_A, DIR_ARM_B)
    )


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
    - 两者都不是（缺 ``report/`` 层级）→ 按形态报错：像旧版产物（报告 / 指标落在
      目录根，``runs/`` 下是平铺 run）报 :data:`LEGACY_LAYOUT_MSG`；目录里根本没有
      数据集产物（空目录、跑到一半崩掉）报 :data:`NO_DATASET_PRODUCT_MSG`。

    Args:
        target: 数据集目录或实验目录。
        dataset: 数据集 key，用于在实验目录下优先定位（默认 None）。

    Returns:
        数据集目录的绝对路径。

    Raises:
        SystemExit: 入参目录不存在、入参是对话级产物目录、数据集目录缺
            ``report/metrics.json``（产物不完整）、实验目录下有多个数据集目录而
            调用方未指定、入参是旧版产物结构，或实验目录下没有任何数据集产物。
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
        # runs/ 下没有可用的数据集产物：平铺形态是旧版结构，其余如实说"没有产物"
        if any(
            _looks_like_legacy_product(os.path.join(runs, name))
            for name in os.listdir(runs)
        ):
            raise SystemExit(f"{LEGACY_LAYOUT_MSG}：{abspath}")
        raise SystemExit(f"{NO_DATASET_PRODUCT_MSG}：{abspath}")
    if _looks_like_legacy_product(abspath):
        raise SystemExit(f"{LEGACY_LAYOUT_MSG}：{abspath}")
    raise SystemExit(f"{NO_DATASET_PRODUCT_MSG}：{abspath}")
