"""一次实验一个自包含子目录的产物布局。

约定（报告与其 run 就近归属，不再与全局 run 混放）：

单次评测（run / evaluate）::

    <root>/eval_<adapter>_<platform>_<ts>/
    ├── report.md / metrics.json / fig_*.png     # 报告与图表
    └── runs/<run_id>/                            # 本次评测的 run（含分片）

A/B 对比（compare）::

    <root>/ab_<tag>_<ts>/
    ├── A/{report.md,metrics.json,fig_*.png,runs/<run_id>/}
    ├── B/{report.md,metrics.json,fig_*.png,runs/<run_id>/}
    ├── compare_report.md / fig_metric_diff.png
    └── optimization_report.md（可选，由调用方写入）

矩阵对比（matrix）::

    <root>/matrix_<ts>/
    ├── matrix_report.md / matrix_result.json / fig_matrix_accuracy.png
    └── runs/<label>/                             # 各臂 run 副本

``<root>`` 由 :func:`membench.paths.output_root` 解析（或 CLI ``--report-out DIR``）。
命名 helper（:func:`subdir_for` / :func:`ab_subdir` / :func:`matrix_subdir`）是
Runner API 使用者可复用的公开接口，不必自己重写约定。
"""

from __future__ import annotations

import os
import re
import shutil
from datetime import datetime
from typing import Iterable, Sequence

from membench.paths import output_root
from membench.report import copy_report

#: 时间戳后缀格式（与 CLI 一致）。
TS_FORMAT = "%y%m%d_%H%M%S"

#: 目录名安全化：非字母数字与 ``._-`` 的字符替换为 ``-``。
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def ts_now(now: datetime | None = None) -> str:
    """当前时间戳字符串（``%y%m%d_%H%M%S``）。"""
    return (now or datetime.now()).strftime(TS_FORMAT)


def slug(text: str) -> str:
    """把任意标签转成安全的目录名片段（空串回退为 ``default``）。"""
    cleaned = _UNSAFE_RE.sub("-", str(text)).strip("-")
    return cleaned or "default"


def subdir_for(adapter: str, platform: str | None = None, ts: str | None = None) -> str:
    """单次评测实验目录名：``eval_<adapter>_<platform>_<ts>``。"""
    return f"eval_{slug(adapter)}_{slug(platform or 'default')}_{ts or ts_now()}"


def ab_subdir(tag: str, ts: str | None = None) -> str:
    """A/B 对比实验目录名：``ab_<tag>_<ts>``。"""
    return f"ab_{slug(tag)}_{ts or ts_now()}"


def matrix_subdir(ts: str | None = None) -> str:
    """矩阵对比实验目录名：``matrix_<ts>``。"""
    return f"matrix_{ts or ts_now()}"


def experiment_path(base: str | None, name: str) -> str:
    """在 ``base``（默认 :func:`membench.paths.output_root`）下建实验目录并返回。"""
    path = os.path.join(base or output_root(), name)
    os.makedirs(path, exist_ok=True)
    return path


def copy_report_into(md_path: str, dest_dir: str, dest_filename: str = "report.md") -> str:
    """把 markdown 报告连同其引用的本地图片复制到 ``dest_dir``（保持原文件名）。"""
    return copy_report(md_path, dest_dir, dest_filename, "")


def _plan_run_targets(
    dest_runs_dir: str, run_dirs: Sequence[str], labels: Sequence[str] | None
) -> list[tuple[str, str]]:
    """计算每个源 run 的目标目录，先整体校验再复制。

    先算完再复制，避免发现冲突时已写入半个目录。

    Returns:
        ``[(dest, run_dir), ...]``。

    Raises:
        ValueError: 两个**不同**源目录映射到同一目标目录（``labels`` 缺省时是
            basename 相同，给了 labels 时是 ``slug()`` 把不同标签归一成同名），
            否则 ``copytree`` 会把两份 run 静默混进同一目录。
    """
    targets: list[tuple[str, str]] = []
    seen: dict[str, tuple[str, str | None]] = {}
    for i, run_dir in enumerate(run_dirs):
        label = labels[i] if labels is not None else None
        if label is not None:
            name = slug(label)
        else:
            name = os.path.basename(os.path.normpath(run_dir))
        dest = os.path.join(dest_runs_dir, name)
        prev = seen.get(os.path.abspath(dest))
        if prev is not None and os.path.abspath(prev[0]) != os.path.abspath(run_dir):
            raise ValueError(
                "copy_runs_into: 目标目录冲突——"
                f"{_src_desc(prev[0], prev[1])} 与 {_src_desc(run_dir, label)} "
                f"都会写入 {dest}；请改用互不相同的 run_id / labels"
            )
        seen.setdefault(os.path.abspath(dest), (run_dir, label))
        targets.append((dest, run_dir))
    return targets


def _src_desc(run_dir: str, label: str | None) -> str:
    """冲突消息里的来源描述：带标签时一并给出。"""
    return f"{run_dir}（标签 {label!r}）" if label is not None else run_dir


def copy_runs_into(
    dest_runs_dir: str,
    run_dirs: Sequence[str],
    labels: Sequence[str] | None = None,
) -> list[str]:
    """把若干 run 目录复制到 ``dest_runs_dir`` 下，返回目标目录列表。

    Args:
        dest_runs_dir: 目标 ``runs/`` 目录（不存在则创建）。
        run_dirs: 源 run 目录列表。
        labels: 可选，与 ``run_dirs`` 一一对应的目录名；缺省用源目录 basename。

    Raises:
        ValueError: 同一次调用内两个**不同**源目录映射到同一目标目录（同源目录
            重复出现不算冲突）。
    """
    os.makedirs(dest_runs_dir, exist_ok=True)
    out: list[str] = []
    for dest, run_dir in _plan_run_targets(dest_runs_dir, run_dirs, labels):
        if os.path.abspath(dest) == os.path.abspath(run_dir):
            out.append(dest)  # 已是同一目录，避免 copytree 自我复制
            continue
        shutil.copytree(run_dir, dest, dirs_exist_ok=True)
        out.append(dest)
    return out


def materialize_run_experiment(exp_dir: str, run_dir: str) -> str:
    """单次评测：报告 + 图表落实验目录根，run 副本落 ``<exp_dir>/runs/``。"""
    os.makedirs(exp_dir, exist_ok=True)
    report_md = os.path.join(run_dir, "report.md")
    if os.path.isfile(report_md):
        copy_report_into(report_md, exp_dir, "report.md")
    for extra in ("metrics.json", "results.jsonl"):
        src = os.path.join(run_dir, extra)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(exp_dir, extra))
    copy_runs_into(os.path.join(exp_dir, "runs"), [run_dir])
    return exp_dir


def materialize_ab_experiment(
    ab_dir: str,
    run_a: str,
    run_b: str,
    compare_dir: str,
) -> str:
    """A/B 对比：A/B 各自带 ``runs/``，对比报告落实验目录根。

    Args:
        ab_dir: A/B 实验目录（``<root>/ab_<tag>_<ts>``）。
        run_a: 基线 run 目录；run_b: 候选 run 目录。
        compare_dir: compare 生成的目录（含 compare_report.md 与配图）。
    """
    for arm, run_dir in (("A", run_a), ("B", run_b)):
        arm_dir = os.path.join(ab_dir, arm)
        os.makedirs(arm_dir, exist_ok=True)
        report_md = os.path.join(run_dir, "report.md")
        if os.path.isfile(report_md):
            copy_report_into(report_md, arm_dir, "report.md")
        for extra in ("metrics.json", "results.jsonl"):
            src = os.path.join(run_dir, extra)
            if os.path.isfile(src):
                shutil.copyfile(src, os.path.join(arm_dir, extra))
        copy_runs_into(os.path.join(arm_dir, "runs"), [run_dir])

    compare_md = os.path.join(compare_dir, "compare_report.md")
    if os.path.isfile(compare_md):
        copy_report_into(compare_md, ab_dir, "compare_report.md")
    return ab_dir


def materialize_matrix_experiment(
    exp_dir: str,
    labeled_runs: Iterable[tuple[str, str]],
) -> str:
    """矩阵对比：各臂 run 副本落 ``<exp_dir>/runs/<label>/``。"""
    os.makedirs(exp_dir, exist_ok=True)
    labels: list[str] = []
    run_dirs: list[str] = []
    for label, run_dir in labeled_runs:
        labels.append(label)
        run_dirs.append(run_dir)
    copy_runs_into(os.path.join(exp_dir, "runs"), run_dirs, labels=labels)
    return exp_dir
