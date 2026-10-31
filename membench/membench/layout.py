"""实验产物布局的唯一入口：目录名 / 报告文件名常量与路径 helper。

一次评测 = 一个自包含的**任务目录**（报告与对应数据就近归属）：

- 单组任务（``groups==1``）是 ``eval_<name>_<platform>_<ts>``，本身就是完整评测产物根；
- 双组任务（``groups==2``）是 ``compare_<name>_<platform>_<ts>``，``A/`` 与 ``B/`` 是任务
  目录下的两个组，各为一个完整的单组评测产物根。

任务目录的命名与骨架由 :func:`task_subdir` / :func:`init_task_dir` 唯一决定、组目录经
:func:`group_path` 取；同一次评测的多个分片写进同一个任务目录时，各片的进度与归属记在
分片账本里——**每个分片一个条目文件** ``shards.<i>.json``（:func:`shard_entry_path`，
并行分片各写各的），``shards.json`` 是由条目物化的视图（:func:`read_shard_ledger` /
:func:`update_shard_ledger` / :func:`check_shard_ledger`——写入方（Runner）与读取方
（fuse 等）共用同一套账本口径）。::

    <root>/eval_<name>_<platform>_<ts>[_s<i>][_l<limit>]/
    ├── runtime/                          # 实验级运行时数据（命令 / 日志 / 环境；见 membench.runtime）
    ├── runs/<dataset>/
    │   ├── report/                      # 数据集级聚合报告（md + metrics + 逐题明细 + 图）
    │   ├── shards.<i>.json, shards.json  # 分片账本：逐片条目（真源）+ 物化视图（分片评测才有）
    │   └── <dialogue_id>/
    │       ├── runtime/                 # **对话级**运行时数据（归适配器使用的运行时文件）
    │       ├── memory_snapshot/         # 对话快照（默认导出，含 manifest.json）
    │       └── report/                  # 对话级报告（纯文字，不出图）
    └── reports/                         # 任务级报告（单数据集 run = 数据集级报告内容）

    <root>/compare_<name>_<platform>_<ts>/   A/ B/（各为完整评测产物根）+ runtime/ + reports/
    <root>/matrix_<ts>/                      runtime/ + reports/ + runs/<label>/（内部同 eval 树）
    <root>/series_<ts>/                      runtime/ + reports/ + runs/<线序>-<slug>/（各线数据集树副本）

**回放任务**（``task init --replay``）在任务目录根多一个内容寻址快照池 ``snapshots/``：
回放导出的状态按内容哈希去重成桶 ``snapshots/<sha256 前 12 位>/``，**每个对话一个
条目文件** ``snapshots/entries/<组>/<数据集>/<对话>.json``（真源，路径见
:func:`snapshot_entry_path`），任务级 ``snapshots/index.json`` 是由条目树 + 各桶清单
重算出来的**派生视图**（:func:`build_snapshot_index` / :func:`materialize_snapshot_index`）。
对话级分片下各进程只写自己负责的对话条目 ⇒ 并行回放不需要任何锁。池属于**任务**
而不是组：双组回放同一源时两组指向同一批桶，差异本身即"回放不保真"的证据。

``<root>`` 由 :func:`membench.paths.output_root` 解析（CLI ``--report-out DIR`` 即指它）。

分片 / 冒烟标签（``_s<i>`` / ``_l<limit>``）只加在目录名末尾，见 :func:`experiment_tags`；
评估报告里的 ``run_id`` 用另一套更完整的词表（含分片总数、分片维度与融合标记），
由 :func:`run_id` 组装、词表见 :func:`run_id_tags`——普通 run 与 fuse 融合产物都走它。

本模块是这些目录名与报告文件名的**唯一来源**：其它模块拼路径一律调用这里的常量与
helper，不要手写字面量。helper 只计算路径、不建目录——需要落盘时调用方自行
``os.makedirs(..., exist_ok=True)``（:func:`experiment_path` / :func:`init_task_dir` 例外，
它们按契约建目录）。读取侧把「产物目录」（实验目录或数据集目录）归一成数据集目录
也只走这里的 :func:`resolve_dataset_dir`。
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Sequence, Union

from membench.paths import output_root

#: 实验目录时间戳格式。
TS_FORMAT = "%Y%m%d-%H%M%S"

#: 实验目录下的各层目录名。
DIR_RUNS = "runs"
DIR_REPORT = "report"  # 对话级 / 数据集级报告目录
DIR_REPORTS = "reports"  # 实验根层的任务级报告目录（与 runs/ 平级）
DIR_RUNTIME = "runtime"  # 对话级运行时数据 / 实验根层运行时数据（后者见 membench.runtime）
DIR_SNAPSHOT = "memory_snapshot"
#: 逐 as_of 的追踪状态目录（``--trace-state``，interleaved 对话专用）：每个有提问的
#: 时点一份 ``trace_state/t<as_of>/``，用于失败归因（"答错时证据写进状态没有"）。
DIR_TRACE_STATE = "trace_state"
#: 回放任务的内容寻址快照池（任务根层，与 runs/ 平级）：桶 = ``<池>/<sha256 前 12 位>``。
DIR_SNAPSHOTS = "snapshots"
#: 快照池里的逐对话条目树（``<池>/entries/<组>/<数据集>/<对话>.json``，池的真源）。
DIR_SNAPSHOT_ENTRIES = "entries"

#: 任务根里的**组**角色（取代旧的"组"）：基线与候选各一组。
ROLE_BASELINE = "baseline"
ROLE_CANDIDATE = "candidate"

#: 基线组的默认显示名（报告/清单里用它）。
DEFAULT_BASELINE_LABEL = "基线"

#: 组目录的默认 ASCII 名（组名可以是中文，目录名不用中文）。
DIR_GROUP_BASELINE = "baseline"
DIR_GROUP_CANDIDATE = "candidate"

#: 组清单文件名：``<任务目录>/groups.json``，记"角色 → 目录名 → 名字"。
#: 组目录名由用户给（可以是中文），因此**所有读取方都必须查清单**，不能靠固定词表猜。
FILE_GROUPS = "groups.json"

#: 组目录名的保留词：与框架目录 / 清单同名会撞。
RESERVED_GROUP_DIRS = frozenset(
    {
        DIR_RUNS,
        DIR_REPORTS,
        DIR_RUNTIME,
        DIR_SNAPSHOTS,
        "entries",
        "A",
        "B",
        FILE_GROUPS,
        "manifest.json",
        "shards.json",
        "evaluate_summary.json",
        "matrix_result.json",
    }
)

#: 组目录名的长度上限（够放一句中文短语，又不至于把路径撑爆）。
GROUP_DIR_MAX_CHARS = 48

#: 路径不安全字符：目录名保留中文，但分隔符 / 通配符 / 控制字符必须换掉。
_PATH_UNSAFE_RE = re.compile(r'[\/:**?"<>|\x00-\x1f]+')

#: 报告与数据文件名。
FILE_REPORT_MD = "report.md"
FILE_COMPARISON_MD = "comparison.md"
FILE_METRICS = "metrics.json"
FILE_RESULTS = "results.jsonl"
FILE_SUMMARY = "evaluate_summary.json"
FILE_MATRIX_RESULT = "matrix_result.json"
#: 分片账本（落数据集目录下）：同一次评测各分片的进度与归属，见 :func:`shard_ledger_path`。
#: 真源是逐片条目文件 ``shards.<i>.json``（并行分片各写各的），本文件是物化视图。
FILE_SHARDS = "shards.json"
#: 快照池索引文件名（落回放任务的 ``snapshots/`` 下），见 :func:`snapshot_index_path`。
FILE_SNAPSHOT_INDEX = "index.json"
#: 快照桶 / 对话快照目录里的清单文件名（框架写，适配器不碰），池的派生索引也读它。
FILE_SNAPSHOT_MANIFEST = "manifest.json"

#: 图表文件名（报告内引用一律同目录相对文件名）。
FIG_CATEGORY = "fig_category_accuracy.png"
FIG_LATENCY = "fig_latency_hist.png"
FIG_METRIC_DIFF = "fig_metric_diff.png"
FIG_CATEGORY_DIFF = "fig_category_diff.png"
FIG_DATASET_OVERVIEW = "fig_dataset_overview.png"
FIG_MATRIX_ACCURACY = "fig_matrix_accuracy.png"
FIG_PER_DATASET_DIFF = "fig_per_dataset_diff.png"
FIG_FUSED_DIFF = "fig_fused_diff.png"
FIG_SERIES_ACCURACY = "fig_series_accuracy.png"
#: 多线对比：融合层各指标 × 各线的分组柱。
FIG_SERIES_FUSED = "fig_series_fused.png"
#: 多线对比：逐数据集关键指标（每个数据集一个面板，柱按线分组）。
FIG_SERIES_PER_DATASET = "fig_series_per_dataset.png"
#: 多线对比：每条线的 accuracy 差值（vs 基线 / vs 上一步）。
FIG_SERIES_DELTA = "fig_series_delta.png"

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


def group_dir_name(label: str, *, ascii_fallback: Union[str, None] = None) -> str:
    """组名 → **目录名（ASCII）**：``slug`` 化，中文名必须由调用方另给英文目录名。

    为什么目录名不用中文：中文路径会在报告内链编码、跨工具（Windows / CI / 归档）
    和 ``ls`` 排序上各带一份麻烦，而组名的可读性只需要出现在**报告与清单**里。
    所以职责拆开——``label`` 是给人看的（可以是「索引优化」），``dir`` 是 ASCII
    短名（``index-opt``）。中文 label 里的 ASCII 片段仍会被利用，实在没有就退
    ``ascii_fallback``（调用方给，如 commit8 / ``candidate``），再退 ``group``。

    Args:
        label: 用户给的组名。
        ascii_fallback: label 里没有任何 ASCII 片段时用的目录名。

    Returns:
        目录名（ASCII）。
    """
    def _clean(value: str) -> str:
        return slug(value).strip(".-") or "default"

    text = _clean(label)
    if text != "default":
        return text
    fallback = _clean(ascii_fallback) if ascii_fallback else ""
    return fallback if fallback and fallback != "default" else "group"


def validate_group_labels(
    labels: "Sequence[str]", fallbacks: "Sequence[str] | None" = None
) -> list[str]:
    """校验一组组名：非空、不撞保留名、**大小写不敏感**唯一。

    为什么大小写不敏感：``Rerank`` 与 ``rerank`` 在 macOS 默认文件系统上是同一个
    目录，判成两个组会静默覆盖。用 ``casefold()`` 而不是 ``lower()``，避免某些
    Unicode 大小写映射不等价。

    Args:
        labels: 组名列表（含基线）。
        fallbacks: 与 labels 等长的 ASCII 目录名兜底（中文名没有 ASCII 片段时用，
            如 ``baseline`` / ``candidate`` / commit8）。

    Returns:
        归一后的**目录名**列表（ASCII，顺序与入参一致）。

    Raises:
        ValueError: 名字非法 / 与保留名冲突 / 重复。
    """
    dirs: list[str] = []
    seen: dict[str, str] = {}
    fb = list(fallbacks or [])
    for index, label in enumerate(labels):
        if not str(label or "").strip():
            raise ValueError("组名不能为空")
        name = group_dir_name(
            label, ascii_fallback=(fb[index] if index < len(fb) else None)
        )
        if name.casefold() in RESERVED_GROUP_DIRS_CASEFOLD:
            raise ValueError(
                f"组名 {name!r} 与框架目录/文件同名（保留名："
                + "、".join(sorted(RESERVED_GROUP_DIRS))
                + "），请换一个"
            )
        if name.casefold() in seen:
            raise ValueError(
                f"组名重复（大小写视为相同）：{seen[name.casefold()]!r} 与 {name!r}"
            )
        seen[name.casefold()] = name
        dirs.append(name)
    return dirs


#: 保留名的大小写不敏感集合（模块加载时算一次）。
RESERVED_GROUP_DIRS_CASEFOLD = frozenset(name.casefold() for name in RESERVED_GROUP_DIRS)


def groups_manifest_path(task_dir: str) -> str:
    """任务根里的组清单路径：``<task_dir>/groups.json``。"""
    return os.path.join(task_dir, FILE_GROUPS)


def write_groups_manifest(task_dir: str, groups: "Sequence[dict[str, Any]]") -> str:
    """写组清单（原子替换），返回路径。

    Args:
        task_dir: 任务目录。
        groups: ``[{"role","label","dir","commit","version"}, …]``，顺序即报告顺序。
    """
    path = groups_manifest_path(task_dir)
    payload = {"schema": 1, "groups": list(groups)}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp, path)
    return path


def read_groups_manifest(task_dir: str) -> "list[dict[str, Any]] | None":
    """读组清单；不存在或结构不认识时返回 ``None``（不猜）。"""
    path = groups_manifest_path(task_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    groups = payload.get("groups") if isinstance(payload, dict) else None
    if not isinstance(groups, list) or not groups:
        return None
    for item in groups:
        if not isinstance(item, dict) or not item.get("role") or not item.get("dir"):
            return None
    return groups


def group_dir(task_dir: str, role: str) -> str:
    """按**角色**取组目录：``<task_dir>/<清单里的 dir>``。

    组目录名由用户给（可以是中文），不能再靠 ``A`` / ``B`` 猜。缺 ``groups.json``
    的任务是旧布局——按"不兼容旧产物"的口径直接报错，给出可操作提示。

    Args:
        task_dir: 任务目录（含 ``groups.json``）。
        role: :data:`ROLE_BASELINE` 或 :data:`ROLE_CANDIDATE`。

    Raises:
        SystemExit: 缺清单，或清单里没有该角色。
    """
    groups = read_groups_manifest(task_dir)
    if groups is None:
        raise SystemExit(
            f"任务目录缺 {FILE_GROUPS}：{task_dir}（旧布局不再兼容）；"
            "请用新版 `task init` 重建任务目录后重跑"
        )
    for item in groups:
        if str(item.get("role")) == role:
            return os.path.join(task_dir, str(item["dir"]))
    available = "、".join(f"{item.get('role')}={item.get('dir')}" for item in groups)
    raise SystemExit(f"任务目录 {task_dir} 里没有角色 {role!r}（现有：{available}）")


def ensure_dir_name(value: str, *, what: str = "任务名") -> str:
    """校验"会进目录名"的标签：``slug`` 后退化成 ``default`` 就报错。

    目录名一律 ASCII（中文路径会在报告内链编码、跨工具、排序上各带一份麻烦）；
    显示名可以中文，但它要和目录名分开给（组名有 ``--*-name`` / ``--*-dir`` 两套）。
    任务名没有独立的显示名，所以这里要求它自身就能 slug 出可用片段，并在报错里
    给出可操作示例——否则所有中文任务名都会撞进 ``eval_default_<ts>``。

    Args:
        value: 待进目录名的标签。
        what: 报错里的称呼（任务名 / 对比标签 / 平台…）。

    Returns:
        原样返回 ``value``（便于链式使用）。

    Raises:
        ValueError: ``slug(value)`` 退化为 ``default``（纯中文 / 纯符号）。
    """
    if slug(value) == "default":
        raise ValueError(
            f"{what}需要含 ASCII 字符（目录名不用中文）：{value!r}；"
            f"例如 --name index-opt，中文描述放在报告 / 提交信息里"
        )
    return value


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


def series_subdir(ts: str | None = None) -> str:
    """多线对比实验目录名：``series_<ts>``（与 ``matrix_<ts>`` 同形，不带线数）。"""
    return f"series_{ts or ts_now()}"


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
# 任务目录：一次评测的产物根（单组 eval_*，双组 A/B compare_*）
# ---------------------------------------------------------------------------

def task_subdir(
    name: str,
    platform: str | None = None,
    groups: int = 2,
    ts: str | None = None,
) -> str:
    """任务目录名：``groups==2`` 走 :func:`compare_subdir`，``groups==1`` 走 :func:`eval_subdir`。

    任务目录的命名只有这一个来源：两种 groups 共用 ``<name>_<platform>_<ts>`` 词表，
    差别只在前缀（``eval_`` / ``compare_``）。
    """
    if groups == 2:
        return compare_subdir(name, platform, ts)
    if groups == 1:
        return eval_subdir(name, platform, ts)
    raise ValueError(f"groups 必须是 1 或 2，当前为 {groups!r}")


def init_task_dir(
    base: str | None,
    name: str,
    platform: str | None = None,
    groups: int = 2,
    ts: str | None = None,
    group_labels: "Sequence[str] | None" = None,
    group_dirs: "Sequence[str] | None" = None,
    *,
    replay: bool = False,
) -> str:
    """在 ``base``（默认 :func:`membench.paths.output_root`）下建任务目录骨架，返回绝对路径。

    骨架：``groups==2`` 按 ``group_dirs``（缺省 ``baseline`` / ``candidate``）建两个组目录
    （各是一个完整评测产物根）并写 ``groups.json``（角色 → 目录名 → 显示名）；``groups==1``
    建 ``runs/``；两种形态都建任务级 ``reports/``。``replay=True``（回放任务）在任务根
    额外建内容寻址快照池 ``snapshots/``（见 :func:`snapshot_pool_root`）：回放导出的
    状态按内容去重后落在这里，A/B 两组共用同一批桶。目标**已存在且非空**时拒绝新建
    （守卫语义与报错文案复用 :func:`experiment_path`：上一次跑在校验阶段失败留下的
    空目录可以复用，已有产物的目录一律不覆盖）。
    """
    path = experiment_path(base, task_subdir(name, platform, groups, ts))
    if groups == 2:
        labels = list(group_labels or [DEFAULT_BASELINE_LABEL, "候选"])
        if len(labels) != 2:
            raise ValueError(
                "双组任务需要两个组名（--baseline-name / --candidate-name）；"
                f"当前为 {labels!r}"
            )
        fallbacks = list(group_dirs or ["baseline", "candidate"])
        if len(fallbacks) != 2:
            raise ValueError(f"group_dirs 需要两个目录名，当前为 {fallbacks!r}")
        dirs = validate_group_labels(labels, fallbacks=fallbacks)
        write_groups_manifest(
            path,
            [
                {"role": ROLE_BASELINE, "label": labels[0], "dir": dirs[0]},
                {"role": ROLE_CANDIDATE, "label": labels[1], "dir": dirs[1]},
            ],
        )
        subdirs: tuple[str, ...] = tuple(dirs)
    else:
        subdirs = (DIR_RUNS,)
    if replay:
        subdirs = (*subdirs, DIR_SNAPSHOTS)
    for subdir in (*subdirs, DIR_REPORTS):
        os.makedirs(os.path.join(path, subdir), exist_ok=True)
    return os.path.abspath(path)


# ---------------------------------------------------------------------------
# 回放任务的快照池：内容寻址的桶 + 逐对话条目（真源）+ 派生索引
# ---------------------------------------------------------------------------

#: 快照池索引结构版本：结构变了就 +1，读到不认识的版本即拒绝。
#: v2（本版）：索引里的 ``groups`` 段取代旧的 ``groups`` 段（组目录名由用户给，不再固定 A/B）。
SNAPSHOT_POOL_SCHEMA = 2

#: 逐对话快照条目（``snapshots/entries/**/<对话>.json``）的结构版本。
SNAPSHOT_ENTRY_SCHEMA = 1

#: 桶名长度：状态内容 sha256 的前 12 位（既够短、又足以区分一次评测里的各对话状态）。
SNAPSHOT_BUCKET_CHARS = 12

#: 池内临时导出目录前缀（导出算完哈希后整体移入桶，失败即删；不是桶，遍历时必须跳过）。
SNAPSHOT_TMP_PREFIX = ".tmp-"

#: 空组键（单组任务 / 普通实验根）在条目树里的目录名：目录名不能是空串。
_EMPTY_GROUP_SEGMENT = "_"

#: 桶名只允许小写十六进制：索引里的条目指向什么目录必须能一眼校验，防止手改索引越出池。
_BUCKET_RE = re.compile(r"^[0-9a-f]{%d}$" % SNAPSHOT_BUCKET_CHARS)


def snapshot_task_root(target: str) -> str:
    """``target`` 所属的任务目录：它是任务下的**组目录**时取父任务目录。

    组目录名由用户给（可以是中文，记在父任务的 ``groups.json``），因此这里查清单，
    不再依赖固定词表。不是组目录时 ``target`` 自己就是任务目录（单组任务 / 普通实验）。
    """
    path = os.path.abspath(target)
    parent = os.path.dirname(path)
    groups = read_groups_manifest(parent)
    if groups and os.path.basename(path).casefold() in {
        str(item["dir"]).casefold() for item in groups
    }:
        return parent
    return path


def snapshot_pool_root(target: str) -> str:
    """快照池根目录：``<target 所属任务目录>/snapshots``（普通 run 不会写它）。"""
    return os.path.join(snapshot_task_root(target), DIR_SNAPSHOTS)


def snapshot_group_key(target: str) -> str:
    """组键（条目树目录名 / 索引键的来源）：目标产物根相对任务目录的路径。

    ``A`` / ``B``；任务自身（单组任务 / 普通实验根）为 ``""``（条目树里映射为 ``_``）。
    """
    rel = os.path.relpath(os.path.abspath(target), snapshot_task_root(target))
    return "" if rel == os.curdir else rel.replace(os.sep, "/")


def snapshot_entry_key(group: str, dataset: str, dialogue_id: str) -> str:
    """索引 ``entries`` 的键：``<group>|<dataset>|<dialogue_id>``。

    组键由 :func:`snapshot_group_key` 决定（单组任务为空串），这三段拼起来才能唯一确定
    "哪个组的哪个对话指向哪个桶"——桶本身只按内容寻址，不含身份。
    """
    return f"{group}|{dataset}|{dialogue_id}"


def parse_snapshot_entry_key(key: str) -> Union[tuple[str, str, str], None]:
    """:func:`snapshot_entry_key` 的反解；不是三段形态时返回 None。"""
    parts = str(key).split("|", 2)
    return (parts[0], parts[1], parts[2]) if len(parts) == 3 else None


def snapshot_bucket_id(state_sha256: str) -> str:
    """桶名：状态内容 sha256（不含 manifest）的前 :data:`SNAPSHOT_BUCKET_CHARS` 位。"""
    return str(state_sha256)[:SNAPSHOT_BUCKET_CHARS]


def snapshot_bucket_dir(pool_root: str, digest: str) -> str:
    """池里的桶目录：``<pool_root>/<digest>``（digest 不合法直接拒绝）。"""
    if not _BUCKET_RE.match(str(digest)):
        raise SystemExit(
            f"快照桶名非法: {digest!r}（应为 {SNAPSHOT_BUCKET_CHARS} 位小写十六进制）"
        )
    return os.path.join(pool_root, digest)


def new_snapshot_tmp_dir(pool_root: str) -> str:
    """在池内建一个临时导出目录（``.tmp-<uuid>``），返回它的路径。

    桶必须由完整内容一次性出现：先导出到这里、算完哈希再整体 ``os.replace`` 进桶，
    中途崩溃最多留下一个 ``.tmp-*`` 目录，不会被误读成桶。
    """
    os.makedirs(pool_root, exist_ok=True)
    path = os.path.join(pool_root, f"{SNAPSHOT_TMP_PREFIX}{uuid.uuid4().hex}")
    os.makedirs(path)
    return path


def snapshot_index_path(pool_root: str) -> str:
    """快照池索引路径：``<pool_root>/index.json``。"""
    return os.path.join(pool_root, FILE_SNAPSHOT_INDEX)


def has_snapshot_pool(task_dir: str) -> bool:
    """该任务目录是不是**回放任务**（任务根有 ``snapshots/`` 目录）。

    判定只看目录存在：``task init --replay`` 建骨架时就建了它，还没导出任何状态的池
    同样算回放任务——把它当 ``--replay`` 源要给出"池里还没有快照"的明确报错，而不是
    退回"目录里没有数据集产物"。普通任务不建 ``snapshots/``。
    """
    return os.path.isdir(os.path.join(task_dir, DIR_SNAPSHOTS))


def snapshot_entries_root(pool_root: str) -> str:
    """条目树根：``<pool_root>/entries``（每个对话一个条目文件，池的真源）。"""
    return os.path.join(pool_root, DIR_SNAPSHOT_ENTRIES)


def snapshot_arm_segment(group: str) -> str:
    """组键在条目树里的目录名：空组键（单组任务 / 普通实验根）用 ``_``。

    目录名不能是空串；组键口径（``""`` / ``A`` / ``B``）仍由 :func:`snapshot_group_key`
    决定，反向解析见 :func:`snapshot_group_from_segment`。
    """
    return tag_dir_name(group) if group else _EMPTY_GROUP_SEGMENT


def snapshot_group_from_segment(segment: str) -> str:
    """条目树目录名回到组键（``_`` → ``""``）。"""
    return "" if segment == _EMPTY_GROUP_SEGMENT else segment


def snapshot_entry_path(
    pool_root: str, group: str, dataset: str, dialogue_id: str
) -> str:
    """条目文件路径：``<池>/entries/<组>/<数据集>/<对话目录名>.json``。

    条目自足（含身份、桶名、内容哈希、来源、复用标记、run_id、写入时间），读取方
    不必再读那个跨组共享的 ``index.json``。对话级分片下各进程只写自己负责的对话
    文件，因此并行回放没有任何共享可变状态要抢。
    """
    return os.path.join(
        snapshot_entries_root(pool_root),
        snapshot_arm_segment(group),
        slug(dataset),
        f"{tag_dir_name(dialogue_id)}.json",
    )


def _load_snapshot_entry_file(path: str) -> dict[str, Any]:
    """读一个条目文件并校验结构；损坏 / 版本不认识当场报错（不猜）。"""
    try:
        with open(path, encoding="utf-8") as handle:
            entry = json.load(handle)
    except ValueError as exc:
        raise SystemExit(f"快照池条目损坏（{path}）：{exc}；请修复或删除该条目后重跑")
    if not isinstance(entry, dict) or entry.get("schema") != SNAPSHOT_ENTRY_SCHEMA:
        found = entry.get("schema") if isinstance(entry, dict) else None
        raise SystemExit(
            f"快照池条目结构版本不受支持（{path}）：schema={found!r}，"
            f"本版本支持 {SNAPSHOT_ENTRY_SCHEMA}；请换一个任务目录重跑"
        )
    for key in ("digest", "dataset", "dialogue_id", "group"):
        if not isinstance(entry.get(key), str):
            raise SystemExit(f"快照池条目结构非法（{path}）：缺少 {key} 字段")
    if not entry["dataset"] or not entry["dialogue_id"]:
        raise SystemExit(f"快照池条目结构非法（{path}）：dataset / dialogue_id 不能为空")
    if not _BUCKET_RE.match(entry["digest"]):
        raise SystemExit(
            f"快照池条目里的桶名非法（{path}）：{entry['digest']!r}"
            f"（应为 {SNAPSHOT_BUCKET_CHARS} 位小写十六进制）"
        )
    return entry


def read_snapshot_entry(
    pool_root: str, group: str, dataset: str, dialogue_id: str
) -> Union[dict[str, Any], None]:
    """读一个对话的条目（没有返回 None）；条目身份与所在位置不一致即报错。"""
    path = snapshot_entry_path(pool_root, group, dataset, dialogue_id)
    if not os.path.isfile(path):
        return None
    entry = _load_snapshot_entry_file(path)
    expected = {"group": group, "dataset": dataset, "dialogue_id": str(dialogue_id)}
    for key, want in expected.items():
        if entry.get(key) != want:
            raise SystemExit(
                f"快照池条目的身份与所在位置不一致（{path}）：{key}={entry.get(key)!r}，"
                f"本次需要 {want!r}——条目可能被移动或改名，请重新导出"
            )
    return entry


def write_snapshot_entry(pool_root: str, entry: dict[str, Any]) -> str:
    """原子写一个对话的条目文件，返回其路径（临时文件 + ``os.replace``）。

    每对话一个文件 ⇒ 分片进程只写自己的对话，互不覆盖。同一对话被重复写（同片重跑）
    按"覆盖自己的条目"处理，但 digest 必须一致：状态变了还覆盖，就会让"哪个对话是
    哪个桶"与桶内实际内容脱节，宁可拒绝并要求检查。
    """
    if entry.get("schema") != SNAPSHOT_ENTRY_SCHEMA:
        raise SystemExit(
            f"快照池条目结构版本不受支持：schema={entry.get('schema')!r}，"
            f"本版本要求 {SNAPSHOT_ENTRY_SCHEMA}"
        )
    group = str(entry.get("group") or "")
    dataset = str(entry.get("dataset") or "")
    dialogue_id = str(entry.get("dialogue_id") or "")
    if not dataset or not dialogue_id:
        raise SystemExit("快照池条目缺少 dataset / dialogue_id，无法落盘")
    digest = str(entry.get("digest") or "")
    snapshot_bucket_dir(pool_root, digest)  # 只做桶名形状校验
    path = snapshot_entry_path(pool_root, group, dataset, dialogue_id)
    existing = read_snapshot_entry(pool_root, group, dataset, dialogue_id)
    if existing is not None and existing.get("digest") != digest:
        raise SystemExit(
            f"快照池条目已存在且内容不同（{path}）：原桶 {existing.get('digest')} vs "
            f"本次 {digest}——同一对话重复写入必须是同一份状态（状态变了说明回放不"
            "保真或池被改动）；请检查后换一个任务目录重跑"
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(entry, handle, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def iter_snapshot_entries(pool_root: str) -> list[dict[str, Any]]:
    """条目树里的全部条目（按 组 / 数据集 / 对话 排序；条目树不存在时为空）。"""
    root = snapshot_entries_root(pool_root)
    entries: list[dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if name.endswith(".json"):
                entries.append(_load_snapshot_entry_file(os.path.join(dirpath, name)))
    entries.sort(
        key=lambda e: (e["group"], e["dataset"], e["dialogue_id"])
    )
    return entries


def has_snapshot_entries(pool_root: str) -> bool:
    """条目树里有没有任何条目文件（旧产物只有 ``index.json``，没有条目树）。

    ``task status`` / ``fuse`` 用它区分"新产物"与"旧产物"：前者从条目 + 桶重算并
    刷新索引，后者索引是唯一记录，只能原样读、绝不能拿空条目重算覆盖掉。
    """
    root = snapshot_entries_root(pool_root)
    if not os.path.isdir(root):
        return False
    for _dirpath, _dirnames, filenames in os.walk(root):
        if any(name.endswith(".json") for name in filenames):
            return True
    return False


def snapshot_pool_groups(pool_root: str) -> list[str]:
    """池里出现过的组键（条目树优先；只有旧产物、没有条目树时才读索引的 ``groups`` 段）。"""
    root = snapshot_entries_root(pool_root)
    if os.path.isdir(root):
        groups = sorted(
            snapshot_group_from_segment(name)
            for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        )
        if groups:
            return groups
    index = read_snapshot_index(pool_root)
    return sorted(str(name) for name in index["groups"]) if index else []


def list_snapshot_buckets(pool_root: str) -> list[str]:
    """池里的桶名（按名排序；``.tmp-*`` / ``entries`` / 索引文件都不算桶）。"""
    if not os.path.isdir(pool_root):
        return []
    return sorted(
        name
        for name in os.listdir(pool_root)
        if _BUCKET_RE.match(name) and os.path.isdir(os.path.join(pool_root, name))
    )


def _bucket_stats(bucket: str) -> tuple[int, int]:
    """桶的内容统计 ``(字节数, 文件数)``：排除顶层 ``manifest.json``。

    与 :func:`membench.snapshot.dir_stats` 的口径一致（``n_files`` 不含框架写的清单），
    这里不 import snapshot（它会 import 本模块），故自带同样的小遍历。
    """
    total = 0
    count = 0
    for root, _dirs, files in os.walk(bucket):
        for name in files:
            if root == bucket and name == FILE_SNAPSHOT_MANIFEST:
                continue
            count += 1
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total, count


def _read_bucket_manifest(bucket: str) -> dict[str, Any]:
    """读桶内清单（派生索引的桶身份来源）；缺失 / 损坏当场报错。"""
    path = os.path.join(bucket, FILE_SNAPSHOT_MANIFEST)
    if not os.path.isfile(path):
        raise SystemExit(
            f"快照桶缺少清单: {path}——桶被改动或未完整落位，请检查或删除该桶后重跑"
        )
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except ValueError as exc:
        raise SystemExit(f"快照桶清单损坏（{path}）：{exc}；请检查或删除该桶后重跑")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("entry"), dict):
        raise SystemExit(f"快照桶清单结构非法（{path}）：缺少 entry 段")
    return manifest


def build_snapshot_index(pool_root: str) -> dict[str, Any]:
    """从条目树 + 各桶清单重算索引（不落盘）。

    真源是 ``snapshots/entries/``（逐对话自足条目）与各桶的 ``manifest.json``；
    ``index.json`` 只是便于人看、兼容旧读者的派生视图。并行回放时没有任何进程需要
    读-改-写它，因此不需要锁。

    ``groups`` 段的 ``source`` 取该组**最近一次**写入的条目（同秒时按 run_id 定序），
    计数是该组当前全部条目的 ``reused`` / ``reused=false`` 统计。
    """
    buckets: dict[str, Any] = {}
    for digest in list_snapshot_buckets(pool_root):
        bucket = os.path.join(pool_root, digest)
        manifest = _read_bucket_manifest(bucket)
        state_sha256 = (manifest.get("entry") or {}).get("sha256")
        if not isinstance(state_sha256, str) or not state_sha256:
            raise SystemExit(
                f"快照桶清单缺少 entry.sha256: "
                f"{os.path.join(bucket, FILE_SNAPSHOT_MANIFEST)}"
            )
        if snapshot_bucket_id(state_sha256) != digest:
            raise SystemExit(
                f"快照桶名与内容哈希不一致（{bucket}）：清单记 {state_sha256}，"
                f"桶名应为 {snapshot_bucket_id(state_sha256)}——桶被搬动或改动过"
            )
        n_bytes, n_files = _bucket_stats(bucket)
        buckets[digest] = {
            "dataset": manifest.get("dataset"),
            "dialogue_id": str(manifest.get("dialogue_id")),
            "bytes": n_bytes,
            "n_files": n_files,
            "state_sha256": state_sha256,
        }

    entries = iter_snapshot_entries(pool_root)
    index_entries: dict[str, str] = {}
    latest: dict[str, tuple[tuple[str, str], dict[str, Any]]] = {}
    counts: dict[str, list[int]] = {}
    for entry in entries:
        group = entry["group"]
        index_entries[snapshot_entry_key(group, entry["dataset"], entry["dialogue_id"])] = (
            entry["digest"]
        )
        counts.setdefault(group, [0, 0])[0 if entry.get("reused") else 1] += 1
        stamp = (str(entry.get("written_at") or ""), str(entry.get("run_id") or ""))
        if group not in latest or stamp >= latest[group][0]:
            latest[group] = (stamp, entry)

    groups: dict[str, Any] = {}
    for group, (reused, new) in counts.items():
        source = latest[group][1]
        groups[group] = {
            "source": source.get("source"),
            "source_manifest_sha256": source.get("source_manifest_sha256"),
            "n_dialogues": reused + new,
            "n_reused": reused,
            "n_new": new,
        }
    return {
        "schema": SNAPSHOT_POOL_SCHEMA,
        "buckets": buckets,
        "entries": index_entries,
        "groups": groups,
    }


def materialize_snapshot_index(pool_root: str) -> dict[str, Any]:
    """重算并原子写回 ``snapshots/index.json``，返回写回的索引。

    这是 fuse / task status 的收尾动作（分片回放结束后刷新派生视图）；写入方（回放）
    不碰索引，因此并行分片不需要任何锁。结构见 :func:`build_snapshot_index`。
    """
    index = build_snapshot_index(pool_root)
    write_snapshot_index(pool_root, index)
    return index


def read_snapshot_index(pool_root: str) -> Union[dict[str, Any], None]:
    """读池索引（没有索引返回 None）；结构非法当场报错。

    结构：``buckets`` 按桶名登记内容身份（``dataset`` / ``dialogue_id`` / ``bytes`` /
    ``n_files`` / ``state_sha256``），``entries`` 是 ``group|dataset|dialogue -> 桶名``，
    ``groups`` 记每个组的回放来源与计数。索引是**派生视图**（真源 = ``entries/`` 树 +
    各桶清单，见 :func:`build_snapshot_index`），只有旧产物才把它当真源读。
    """
    path = snapshot_index_path(pool_root)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            index = json.load(handle)
    except ValueError as exc:
        raise SystemExit(f"快照池索引损坏（{path}）：{exc}；请修复或删除该文件后重跑")
    if not isinstance(index, dict) or index.get("schema") != SNAPSHOT_POOL_SCHEMA:
        found = index.get("schema") if isinstance(index, dict) else None
        raise SystemExit(
            f"快照池索引结构版本不受支持（{path}）：schema={found!r}，"
            f"本版本支持 {SNAPSHOT_POOL_SCHEMA}；请换一个任务目录重跑"
        )
    for key in ("buckets", "entries", "groups"):
        if not isinstance(index.get(key), dict):
            raise SystemExit(f"快照池索引结构非法（{path}）：缺少 {key} 段")
    for digest in index["buckets"]:
        if not _BUCKET_RE.match(str(digest)):
            raise SystemExit(
                f"快照池索引里的桶名非法（{path}）：{digest!r}"
                f"（应为 {SNAPSHOT_BUCKET_CHARS} 位小写十六进制）"
            )
    return index


def write_snapshot_index(pool_root: str, index: dict[str, Any]) -> str:
    """落盘池索引，返回索引路径（唯一名临时文件 + ``os.replace``，同目录内原子替换）。

    索引会被多个进程各自重算（fuse / task status 可能同时刷新），临时文件名必须唯一，
    否则两个进程会往同一个 ``index.json.tmp`` 里写、互相截断；替换本身是原子的，
    最后落盘的是一份完整快照。
    """
    path = snapshot_index_path(pool_root)
    os.makedirs(pool_root, exist_ok=True)
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(index, handle, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
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


def dataset_dirs(experiment_dir: str) -> list[str]:
    """实验目录 ``runs/`` 下的全部数据集目录（按目录名排序）。

    读取方需要"这份产物里有哪几个数据集"时用它（如多线对比的每条线）；只要一个
    数据集的场景仍走 :func:`resolve_dataset_dir` 的唯一命中解析。``runs/`` 下的对话
    目录（``report/metrics.json`` 里落了 ``dialogue_id``）与没有 ``report/`` 的
    半成品目录都不算数据集目录。
    """
    runs = runs_root(experiment_dir)
    if not os.path.isdir(runs):
        return []
    return sorted(
        os.path.join(runs, name)
        for name in os.listdir(runs)
        if _is_dataset_dir(os.path.join(runs, name))
    )


def series_line_dir(
    series_dir: str,
    index: int,
    name: str,
    commit: str | None = None,
    dir_hint: str | None = None,
) -> str:
    """多线对比实验里一条线的副本目录：``<series>/runs/<线序>-<stem>``。

    ``stem`` 优先级：

    1. ``dir_hint``——产物来自带 ``groups.json`` 的任务时，直接复用该组的**目录名**
       （``baseline`` / ``index-opt``）：用户已经给过 ASCII 目录名，不该在多线对比里
       又退化成 ``line1``；
    2. ``slug(name)``——线名里有 ASCII 片段时用它；
    3. ``commit`` 前 8 位 / ``line<i>``——中文线名且没有命中清单时的兜底。

    线序前缀（``01-`` / ``02-``…）保证一条线一个目录，且顺序即时间顺序。
    """
    if dir_hint:
        stem = slug(dir_hint)
    else:
        stem = slug(name)
        if stem == "default":
            stem = str(commit)[:8] if commit else f"line{index + 1}"
    return os.path.join(runs_root(series_dir), f"{index + 1:02d}-{stem}")


def shard_ledger_path(experiment_dir: str, dataset: str) -> str:
    """分片账本**视图**路径：``<exp>/runs/<dataset>/shards.json``。

    账本记同一次评测各分片的进度与归属（谁在跑、跑完没、覆盖了哪些对话），因此落在
    数据集目录里。真源是逐片条目文件 ``shards.<i>.json``（:func:`shard_entry_path`，
    并行分片各写各的）；``shards.json`` 由条目物化，供人直接翻，读取方走
    :func:`read_shard_ledger` 的合并结果。
    """
    return os.path.join(dataset_dir(experiment_dir, dataset), FILE_SHARDS)


def dialogue_dir(experiment_dir: str, dataset: str, dialogue_id: str) -> str:
    """对话目录：``<exp>/runs/<dataset>/<tag_dir_name(dialogue_id)>``。"""
    return os.path.join(dataset_dir(experiment_dir, dataset), tag_dir_name(dialogue_id))


def dialogue_report_dir(experiment_dir: str, dataset: str, dialogue_id: str) -> str:
    """对话级报告目录。"""
    return os.path.join(dialogue_dir(experiment_dir, dataset, dialogue_id), DIR_REPORT)


def dialogue_runtime_dir(experiment_dir: str, dataset: str, dialogue_id: str) -> str:
    """对话级运行时数据目录（归适配器使用的运行时文件，runner 每对话建出）。"""
    return os.path.join(dialogue_dir(experiment_dir, dataset, dialogue_id), DIR_RUNTIME)


def dialogue_snapshot_dir(experiment_dir: str, dataset: str, dialogue_id: str) -> str:
    """对话级快照目录（默认导出，含框架写的 ``manifest.json``；``--no-snapshot`` 关闭）。"""
    return os.path.join(dialogue_dir(experiment_dir, dataset, dialogue_id), DIR_SNAPSHOT)


def dialogue_trace_state_dir(
    experiment_dir: str, dataset: str, dialogue_id: str
) -> str:
    """对话级逐 as_of 追踪状态根：``<对话目录>/trace_state/``（``--trace-state``）。

    目录下每个有提问的时点一个子目录 ``t<as_of>/``（:func:`dialogue_trace_state_path`），
    每份都有自己的 ``manifest.json``——与 ``memory_snapshot/`` 同构，只是时点不同。
    """
    return os.path.join(dialogue_dir(experiment_dir, dataset, dialogue_id), DIR_TRACE_STATE)


def dialogue_trace_state_path(
    experiment_dir: str, dataset: str, dialogue_id: str, as_of: int
) -> str:
    """某个提问时点的追踪状态目录：``<对话目录>/trace_state/t<as_of>/``。

    ``as_of`` 是已 ingest 的消息条数（:attr:`membench.data.schema.Question.as_of`），
    目录名原样写它：翻产物时能直接对上题目里的时点。
    """
    return os.path.join(
        dialogue_trace_state_dir(experiment_dir, dataset, dialogue_id), f"t{int(as_of)}"
    )


def task_report_dir(experiment_dir: str) -> str:
    """实验根层的任务级报告目录：``<exp>/reports``。"""
    return os.path.join(experiment_dir, DIR_REPORTS)


# ---------------------------------------------------------------------------
# 分片账本：同一次分片评测各片的进度与归属（读 / 写 / 校验的唯一实现）
# ---------------------------------------------------------------------------

#: 分片账本结构版本：结构变了就 +1，读到不认识的版本即拒绝（别让新账本被旧读者改写）。
SHARD_LEDGER_SCHEMA = 1


def shard_key(shard_index: int) -> str:
    """账本里分片的键：``s<i>``（与 ``--shard-index`` 同一套序号，从 0 起）。"""
    return f"s{shard_index}"


def shard_key_index(key: str) -> Union[int, None]:
    """账本键 ``s<i>`` 反解出的分片序号；不是这个形状时返回 None。

    键的格式只由 :func:`shard_key` 决定，反解必须走本函数——读取方（fuse）要靠它把
    账本条目对回 ``--shard-index`` 的序号，自己切字符串就会在格式变化时静默错位。
    """
    if not isinstance(key, str) or not key.startswith("s"):
        return None
    digits = key[1:]
    return int(digits) if digits.isdigit() else None


def has_shard_ledger(dataset_dir: str) -> bool:
    """该目录是不是**分片产物目录**（有账本视图，或有逐片条目文件）。"""
    return os.path.isfile(os.path.join(dataset_dir, FILE_SHARDS)) or bool(
        shard_entry_files(dataset_dir)
    )


#: 逐片条目文件名：``shards.<i>.json``（i 即 ``--shard-index``）。落在数据集目录**文件**
#: 而不是子目录：数据集目录下的子目录一律被当成对话目录（fuse 按这个约定遍历）。
_SHARD_ENTRY_RE = re.compile(r"^shards\.(\d+)\.json$")


def shard_entry_files(dataset_dir: str) -> list[str]:
    """逐片条目文件（按分片序号排序；目录不存在时为空）。"""
    if not os.path.isdir(dataset_dir):
        return []
    found = [
        (int(match.group(1)), name)
        for name in os.listdir(dataset_dir)
        if (match := _SHARD_ENTRY_RE.match(name))
        and os.path.isfile(os.path.join(dataset_dir, name))
    ]
    return [os.path.join(dataset_dir, name) for _index, name in sorted(found)]


def shard_entry_path(dataset_dir: str, key: str) -> str:
    """某一分片的条目文件路径：``<数据集目录>/shards.<i>.json``。

    键走 :func:`shard_key_index` 反解（``s<i>`` → ``i``），不合法直接拒绝（防路径穿越）。
    """
    index = shard_key_index(key)
    if index is None:
        raise ValueError(f"分片键非法: {key!r}（应为 s<i>，见 layout.shard_key）")
    return os.path.join(dataset_dir, f"shards.{index}.json")


def shard_ledger_header(
    *,
    dataset: str,
    shards: int,
    shard_by: str,
    limit: Union[int, None],
    fingerprint: Union[dict[str, Any], None],
) -> dict[str, Any]:
    """本次运行的分片口径（账本头部）：写入新账本时落盘，也是比对的基准。

    字段：``schema`` / ``dataset`` / ``shards`` / ``shard_by`` / ``limit`` /
    ``dataset_fingerprint``。同一个数据集目录里的账本只能记**同一次评测**的各片：
    口径或数据集指纹换了却写进同一目录，各片产物就不可归因（融合得到的是混片结果）。
    """
    return {
        "schema": SHARD_LEDGER_SCHEMA,
        "dataset": dataset,
        "shards": shards,
        "shard_by": shard_by,
        "limit": limit,
        "dataset_fingerprint": fingerprint,
    }


def _read_shard_entry_items(dataset_dir: str) -> list[dict[str, Any]]:
    """读逐片条目文件并按分片序号排序；任何一份损坏 / 身份对不上都当场报错。"""
    items: list[dict[str, Any]] = []
    for path in shard_entry_files(dataset_dir):
        try:
            with open(path, encoding="utf-8") as handle:
                item = json.load(handle)
        except ValueError as exc:
            raise SystemExit(f"分片账本条目损坏（{path}）：{exc}；请修复或删除该文件后重跑")
        if (
            not isinstance(item, dict)
            or item.get("schema") != SHARD_LEDGER_SCHEMA
            or not isinstance(item.get("header"), dict)
            or not isinstance(item.get("entry"), dict)
        ):
            raise SystemExit(
                f"分片账本条目结构非法（{path}）：需要 schema / header / entry 三段"
            )
        key = item.get("key")
        index = shard_key_index(key) if isinstance(key, str) else None
        if index is None:
            raise SystemExit(f"分片账本条目里的分片键非法（{path}）：{key!r}")
        if os.path.basename(path) != f"shards.{index}.json":
            raise SystemExit(
                f"分片账本条目的键与文件名不一致（{path}）：{key!r}——"
                "条目可能被移动或改名，请重新跑该分片"
            )
        items.append(item)
    return items


def read_shard_ledger(dataset_dir: str) -> Union[dict[str, Any], None]:
    """读数据集目录下的分片账本（没有账本返回 None）。

    **逐片条目文件优先**（``shards.<i>.json``：并行分片各写各的、原子落盘，不会互相
    覆盖），合并成账本；只有旧产物（没有逐片条目）才读物化视图 ``shards.json``。
    所有条目的口径（header）必须一致——同一个数据集目录里只能放同一次评测的各片。
    结构非法当场报错：坏账本被静默覆盖，就等于丢掉"哪些分片已跑过、哪一片还在跑"
    的记录。账本条目 ``entries["s<i>"]`` 的字段：``status``（``running`` / ``done``）、
    ``dialogues``（该片负责的对话 id）、``n_questions``（实测题数，running 时可能还
    没有）、``run_id``、``started_at``、``finished_at``。
    """
    items = _read_shard_entry_items(dataset_dir)
    path = os.path.join(dataset_dir, FILE_SHARDS)
    if not items:
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                ledger = json.load(handle)
        except ValueError as exc:
            raise SystemExit(f"分片账本损坏（{path}）：{exc}；请修复或删除该文件后重跑")
        if not isinstance(ledger, dict) or not isinstance(ledger.get("entries"), dict):
            raise SystemExit(
                f"分片账本结构非法（{path}）：缺少 entries 段；请修复或删除该文件后重跑"
            )
        return ledger

    header = items[0]["header"]
    entries: dict[str, Any] = {}
    for item in items:
        if item["header"] != header:
            raise SystemExit(
                f"分片账本的逐片条目口径不一致（{shard_entry_path(dataset_dir, item['key'])}）："
                f"{item['header']!r} vs {header!r}——同一数据集目录里只能放同一次评测的分片"
            )
        entries[item["key"]] = item["entry"]
    return {**header, "entries": entries}


def write_shard_ledger(dataset_dir: str, ledger: dict[str, Any]) -> str:
    """落盘账本**视图** ``shards.json``，返回路径（唯一名临时文件 + 原子替换）。

    真源是逐片条目文件（:func:`update_shard_ledger` 写）；这个视图只为"直接翻文件"
    方便，读者仍以 :func:`read_shard_ledger` 的合并结果为准。临时文件名必须唯一，
    否则并行进程会互相截断 / 抢走对方正在写的临时文件（固定名 ``.tmp`` 就这么崩过）。
    """
    path = os.path.join(dataset_dir, FILE_SHARDS)
    os.makedirs(dataset_dir, exist_ok=True)
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(ledger, handle, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def update_shard_ledger(
    dataset_dir: str,
    header: dict[str, Any],
    key: str,
    entry: dict[str, Any],
) -> dict[str, Any]:
    """写一条分片条目并物化账本视图，返回合并后的账本。

    每个分片一个条目文件（``shards.<i>.json``，单文件原子写）⇒ 并行分片只写自己的
    那一片，读-改-写不会擦掉别人的条目，也不需要锁。``shards.json`` 随后按全部条目
    重新物化，供人直接翻；并发下它可能短暂落后，读者（fuse / status）走
    :func:`read_shard_ledger` 的合并结果。
    """
    path = shard_entry_path(dataset_dir, key)
    os.makedirs(dataset_dir, exist_ok=True)
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema": SHARD_LEDGER_SCHEMA,
                    "header": header,
                    "key": key,
                    "entry": entry,
                },
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    ledger = read_shard_ledger(dataset_dir) or {**header, "entries": {}}
    write_shard_ledger(dataset_dir, ledger)
    return ledger


def check_shard_ledger(
    existing: Union[dict[str, Any], None],
    header: dict[str, Any],
    *,
    path: str,
) -> None:
    """已有账本必须与本次运行属于同一次评测（:func:`shard_ledger_header` 逐项比对）。

    同一个数据集目录里的账本只能记同一次评测：``--shards`` / ``--shard-by`` /
    ``limit`` / 数据集指纹换了却写进同一目录，各片产物就不可归因（融合得到的是混片
    结果），必须在写盘前拒绝。``existing`` 为 None（还没账本）时直接通过。

    Raises:
        SystemExit: 账本 schema 不受支持，或口径 / 数据集指纹与本次不一致。
    """
    if existing is None:
        return
    if existing.get("schema") != SHARD_LEDGER_SCHEMA:
        raise SystemExit(
            f"分片账本结构版本不受支持（{path}）：账本 schema={existing.get('schema')!r}，"
            f"本版本支持 {SHARD_LEDGER_SCHEMA}；请换一个实验目录重跑"
        )
    for key in ("dataset", "shards", "shard_by", "limit"):
        if existing.get(key) != header.get(key):
            raise SystemExit(
                f"分片账本与本次运行不一致（{key}）：账本 {existing.get(key)!r} vs "
                f"本次 {header.get(key)!r}（{path}）——同一个实验目录里只能放同一次评测的分片"
            )
    known = (existing.get("dataset_fingerprint") or {}).get("sha256")
    want = (header.get("dataset_fingerprint") or {}).get("sha256")
    if known and want and known != want:
        raise SystemExit(
            f"分片账本与本次运行的数据集指纹不一致：账本 {known} vs 本次 {want}（{path}）"
            "——数据集变过就换实验目录，否则各片结果不可归因"
        )


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

#: 未融合的分片产物（数据集目录下有 :data:`FILE_SHARDS`）的统一报错文案。分片只落各片
#: 负责的对话目录，数据集级产物要由 ``fuse`` 补齐——这与"没有产物"是两回事。
UNFUSED_SHARD_MSG = (
    f"这是未融合的分片产物（含 {FILE_SHARDS}），请先用 fuse 补齐数据集级产物"
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
    # 旧版 A/B 实验目录（平铺 run）——历史产物识别用，与"组目录名由用户给"无关
    return any(
        os.path.isfile(os.path.join(path, group, FILE_METRICS))
        for group in ("A", "B")
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
      目录根，``runs/`` 下是平铺 run）报 :data:`LEGACY_LAYOUT_MSG`；未融合的分片产物
      （数据集目录下有 :data:`FILE_SHARDS`，只落了各片的对话目录）报
      :data:`UNFUSED_SHARD_MSG`；目录里根本没有数据集产物（空目录、跑到一半崩掉）
      报 :data:`NO_DATASET_PRODUCT_MSG`。

    Args:
        target: 数据集目录或实验目录。
        dataset: 数据集 key，用于在实验目录下优先定位（默认 None）。

    Returns:
        数据集目录的绝对路径。

    Raises:
        SystemExit: 入参目录不存在、入参是对话级产物目录、数据集目录缺
            ``report/metrics.json``（产物不完整）、实验目录下有多个数据集目录而
            调用方未指定、入参是旧版产物结构、入参是未融合的分片产物，或实验目录下
            没有任何数据集产物。
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
    if has_shard_ledger(abspath):
        raise SystemExit(f"{UNFUSED_SHARD_MSG}：{abspath}")

    runs = runs_root(abspath)
    if os.path.isdir(runs):
        if dataset:
            candidate = dataset_dir(abspath, dataset)
            if _is_dataset_dir(candidate):
                return _require_metrics(candidate)
        names_in_runs = os.listdir(runs)
        found = sorted(
            os.path.join(runs, name)
            for name in names_in_runs
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
        # runs/ 下没有可用的数据集产物：分片（未融合）与旧版结构各有专门文案，
        # 其余如实说"没有产物"
        if any(has_shard_ledger(os.path.join(runs, name)) for name in names_in_runs):
            raise SystemExit(f"{UNFUSED_SHARD_MSG}：{abspath}")
        if any(
            _looks_like_legacy_product(os.path.join(runs, name))
            for name in names_in_runs
        ):
            raise SystemExit(f"{LEGACY_LAYOUT_MSG}：{abspath}")
        raise SystemExit(f"{NO_DATASET_PRODUCT_MSG}：{abspath}")
    if _looks_like_legacy_product(abspath):
        raise SystemExit(f"{LEGACY_LAYOUT_MSG}：{abspath}")
    raise SystemExit(f"{NO_DATASET_PRODUCT_MSG}：{abspath}")
