"""产物布局（membench.layout）测试：命名常量、路径 helper、实验目录冲突与产物目录解析。

覆盖终态布局的唯一路径来源：``eval_<name>_<platform>_<ts>`` / ``compare_...`` /
``matrix_<ts>`` 的命名、``runs/<dataset>/<对话>/{report,runtime,memory_snapshot}``
与实验根层 ``reports/`` 的路径拼接、``experiment_path`` 拒绝复用非空实验目录，以及
``resolve_dataset_dir`` 对「数据集目录 / 实验目录 / 对话目录（拒绝）/ 旧版产物」几态的处理。

另断言旧版 API（``subdir_for`` / ``ab_subdir`` / ``copy_runs_into`` / ``materialize_*``）
已随布局重构删除：它们若被恢复，目录命名与落位就出现了第二个来源。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from membench import layout


def _fake_dataset_dir(exp_dir: Path, dataset: str, *, metrics: bool = True) -> Path:
    """造一个数据集形状的产物目录：``<exp>/runs/<dataset>/report/{metrics.json,report.md}``。

    ``metrics=False`` 时只建报告目录、不落 ``metrics.json``，用于验证半成品被拦下。
    """
    report_dir = exp_dir / "runs" / layout.slug(dataset) / layout.DIR_REPORT
    report_dir.mkdir(parents=True)
    (report_dir / layout.FILE_REPORT_MD).write_text("# r\n", encoding="utf-8")
    if metrics:
        (report_dir / layout.FILE_METRICS).write_text(
            json.dumps({"run_id": dataset, "dataset": dataset}, ensure_ascii=False),
            encoding="utf-8",
        )
    return report_dir.parent


def _fake_dialogue_dir(exp_dir: Path, dataset: str, dialogue_id: str) -> Path:
    """造一个对话形状的产物目录：``<exp>/runs/<dataset>/<对话>/report/{metrics.json,report.md}``。

    对话目录与数据集目录一样带 ``report/`` 层级，只有 ``metrics.json`` 里的
    ``dialogue_id`` 能把两者分开——这正是 resolve 必须识别的形状。
    """
    report_dir = Path(layout.dialogue_report_dir(str(exp_dir), dataset, dialogue_id))
    report_dir.mkdir(parents=True)
    (report_dir / layout.FILE_REPORT_MD).write_text("# d\n", encoding="utf-8")
    (report_dir / layout.FILE_METRICS).write_text(
        json.dumps({"run_id": f"run_{dialogue_id}", "dialogue_id": dialogue_id}),
        encoding="utf-8",
    )
    return report_dir.parent


# ---------------------------------------------------------------------------
# 命名与常量
# ---------------------------------------------------------------------------


def test_layout_constants_are_single_source() -> None:
    """目录名 / 报告文件名 / 图名常量：其它模块拼路径一律引用它们。"""
    assert layout.TS_FORMAT == "%Y%m%d-%H%M%S"
    assert (
        layout.DIR_RUNS,
        layout.DIR_REPORT,
        layout.DIR_REPORTS,
        layout.DIR_RUNTIME,
        layout.DIR_SNAPSHOT,
    ) == ("runs", "report", "reports", "runtime", "memory_snapshot")
    assert (
        layout.FILE_REPORT_MD,
        layout.FILE_COMPARISON_MD,
        layout.FILE_METRICS,
        layout.FILE_RESULTS,
        layout.FILE_SUMMARY,
        layout.FILE_MATRIX_RESULT,
        layout.FILE_SHARDS,
    ) == (
        "report.md",
        "comparison.md",
        "metrics.json",
        "results.jsonl",
        "evaluate_summary.json",
        "matrix_result.json",
        "shards.json",
    )
    assert (
        layout.FIG_CATEGORY,
        layout.FIG_LATENCY,
        layout.FIG_METRIC_DIFF,
        layout.FIG_CATEGORY_DIFF,
        layout.FIG_DATASET_OVERVIEW,
        layout.FIG_MATRIX_ACCURACY,
    ) == (
        "fig_category_accuracy.png",
        "fig_latency_hist.png",
        "fig_metric_diff.png",
        "fig_category_diff.png",
        "fig_dataset_overview.png",
        "fig_matrix_accuracy.png",
    )


def test_naming_helpers() -> None:
    """eval_subdir / compare_subdir / matrix_subdir / slug 的命名约定。"""
    assert layout.slug("mock strong/a") == "mock-strong-a"
    assert layout.slug("") == "default"
    assert layout.slug("my set/v2") == "my-set-v2"
    assert layout.eval_subdir("mock-strong", "wx", ts="20260101-120000") == (
        "eval_mock-strong_wx_20260101-120000"
    )
    assert layout.eval_subdir("mock-strong", None, ts="20260101-120000") == (
        "eval_mock-strong_default_20260101-120000"
    )
    assert layout.compare_subdir("a1_vs_b1", "wx", ts="20260101-120000") == (
        "compare_a1_vs_b1_wx_20260101-120000"
    )
    # platform 缺省（两臂同平台时也走这里）固定为 default
    assert layout.compare_subdir("a1_vs_b1", ts="20260101-120000") == (
        "compare_a1_vs_b1_default_20260101-120000"
    )
    assert layout.matrix_subdir(ts="20260101-120000") == "matrix_20260101-120000"


def test_eval_subdir_tags_and_run_id_tags() -> None:
    """实验目录名的分片/冒烟后缀（`_s<i>` / `_l<limit>`）与 run_id 标记词表。"""
    base = layout.eval_subdir("mock-strong", "wx", ts="20260101-120000")
    assert layout.eval_subdir("mock-strong", "wx", ts="20260101-120000", tags=[]) == base
    assert layout.eval_subdir(
        "mock-strong", "wx", ts="20260101-120000", tags=layout.experiment_tags(2, 1, 3)
    ) == f"{base}_s1_l3"
    # tag 与 name / platform 同口径经 slug 归一（非法字符不落进目录名）
    assert layout.eval_subdir("a", ts="20260101-120000", tags=["s 0/x"]) == (
        "eval_a_default_20260101-120000_s-0-x"
    )

    # 不分片且无 limit -> 无后缀；--shards max 只记得分片序号（总数要解析数据集才知道）
    assert layout.experiment_tags(1, 0) == []
    assert layout.experiment_tags(1, 0, 5) == ["l5"]
    assert layout.experiment_tags(4, 2) == ["s2"]
    assert layout.experiment_tags("max", 3) == ["s3"]

    # run_id 词表：分片在前、limit 在后，分片/冒烟结果不会被误当全量结果
    assert layout.run_id_tags(1, 0, "question") == []
    assert layout.run_id_tags(2, 0, "question") == ["shard0of2"]
    assert layout.run_id_tags(2, 1, "dialogue") == ["dshard1of2"]
    assert layout.run_id_tags(2, 1, "dialogue", 3) == ["dshard1of2", "limit3"]
    assert layout.run_id_tags(1, 0, "question", 7) == ["limit7"]


def test_ts_now_follows_ts_format() -> None:
    """ts_now() 的产物是 TS_FORMAT 形状，实验目录名可被按 ts 解析出来。"""
    assert re.fullmatch(r"\d{8}-\d{6}", layout.ts_now())
    assert layout.eval_subdir("a", "b").startswith("eval_a_b_")


def test_tag_dir_name_normalizes_dialogue_ids() -> None:
    """对话目录名归一：非法字符换 ``_``、防路径穿越、超长截断、空值兜底。"""
    assert layout.tag_dir_name("d 001") == "d_001"
    assert layout.tag_dir_name("../../etc/passwd") == "etc_passwd"
    assert layout.tag_dir_name("") == "dialogue"
    assert layout.tag_dir_name("   ") == "dialogue"
    assert layout.tag_dir_name("x" * 200) == "x" * 120


def test_legacy_layout_api_is_gone() -> None:
    """旧版 API 不得复活：命名与落位只能走本模块的 helper。"""
    for name in (
        "subdir_for",
        "ab_subdir",
        "copy_runs_into",
        "_plan_run_targets",
        "copy_report_into",
        "materialize_run_experiment",
        "materialize_ab_experiment",
        "materialize_matrix_experiment",
    ):
        assert not hasattr(layout, name), f"旧 API {name} 不应存在"
    # 快照池的派生索引物化是当前 API（真源 = entries/，索引只是视图）
    assert callable(layout.materialize_snapshot_index)


# ---------------------------------------------------------------------------
# 实验目录内的路径
# ---------------------------------------------------------------------------


def test_path_helpers_build_expected_tree(tmp_path: Path) -> None:
    """runs_root / dataset_dir / *_report_dir / task_report_dir 的逐层拼接。"""
    exp = tmp_path / "eval_mock-strong_wx_20260101-120000"
    assert layout.runs_root(str(exp)) == str(exp / "runs")
    assert layout.dataset_dir(str(exp), "office") == str(exp / "runs" / "office")
    # 数据集目录名按 layout.slug 归一（含空格 / 斜杠的 key 也安全）
    assert layout.dataset_dir(str(exp), "my set/v2") == str(exp / "runs" / "my-set-v2")
    assert layout.dataset_report_dir(str(exp), "office") == str(
        exp / "runs" / "office" / "report"
    )
    assert layout.dialogue_dir(str(exp), "office", "d 001") == str(
        exp / "runs" / "office" / "d_001"
    )
    assert layout.dialogue_report_dir(str(exp), "office", "d 001") == str(
        exp / "runs" / "office" / "d_001" / "report"
    )
    assert layout.dialogue_runtime_dir(str(exp), "office", "d 001") == str(
        exp / "runs" / "office" / "d_001" / "runtime"
    )
    assert layout.dialogue_snapshot_dir(str(exp), "office", "d 001") == str(
        exp / "runs" / "office" / "d_001" / "memory_snapshot"
    )
    assert layout.task_report_dir(str(exp)) == str(exp / "reports")
    assert layout.shard_ledger_path(str(exp), "office") == str(
        exp / "runs" / "office" / layout.FILE_SHARDS
    )


# ---------------------------------------------------------------------------
# experiment_path：建目录与冲突拒绝
# ---------------------------------------------------------------------------


def test_experiment_path_creates_and_rejects_non_empty(tmp_path: Path) -> None:
    """同秒同名实验目录：首次建出、空目录可复用、非空即拒绝（不静默覆盖）。"""
    base = tmp_path / "output"
    name = layout.eval_subdir("mock-strong", "wx", ts="20260101-120000")

    exp = Path(layout.experiment_path(str(base), name))
    assert exp.is_dir() and exp.name == name
    # 上一次跑在校验阶段失败留下的空目录可以复用
    assert layout.experiment_path(str(base), name) == str(exp)

    (exp / "runs").mkdir()
    (exp / "runs" / "office").mkdir()
    with pytest.raises(SystemExit) as excinfo:
        layout.experiment_path(str(base), name)
    message = str(excinfo.value)
    assert str(exp) in message
    assert "非空" in message


def test_experiment_path_defaults_to_output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """base 缺省时落 MEMBENCH_OUTPUT_DIR（默认产物根的唯一入口）。"""
    root = tmp_path / "custom-root"
    monkeypatch.setenv("MEMBENCH_OUTPUT_DIR", str(root))
    exp = Path(layout.experiment_path(None, layout.matrix_subdir(ts="20260101-120000")))
    assert exp.parent == root
    assert exp.is_dir()


# ---------------------------------------------------------------------------
# 任务目录：命名、骨架与守卫
# ---------------------------------------------------------------------------


def test_task_subdir_reuses_single_arm_and_two_arm_naming() -> None:
    """task_subdir：arms==2 就是 compare_*，arms==1 就是 eval_*（任务目录命名只有一个来源）。"""
    assert layout.task_subdir("a1_vs_b1", "wx", ts="20260101-120000") == layout.compare_subdir(
        "a1_vs_b1", "wx", ts="20260101-120000"
    )
    assert layout.task_subdir(
        "mock-strong", "wx", arms=1, ts="20260101-120000"
    ) == layout.eval_subdir("mock-strong", "wx", ts="20260101-120000")
    with pytest.raises(ValueError, match="arms"):
        layout.task_subdir("x", arms=3)


def test_init_task_dir_skeleton_and_non_empty_guard(tmp_path: Path) -> None:
    """双组任务建 baseline/ candidate/ + groups.json，单组任务建 runs/ reports/；非空目录拒绝复用。"""
    base = tmp_path / "output"
    task = Path(layout.init_task_dir(str(base), "a1_vs_b1", "wx", arms=2, ts="20260101-120000"))
    assert task.is_absolute() and task.name == "compare_a1_vs_b1_wx_20260101-120000"
    assert sorted(p.name for p in task.iterdir()) == ["baseline", "candidate", "groups.json", "reports"]
    groups = layout.read_groups_manifest(str(task))
    assert groups is not None
    assert [g["role"] for g in groups] == [layout.ROLE_BASELINE, layout.ROLE_CANDIDATE]
    assert [g["dir"] for g in groups] == ["baseline", "candidate"]
    assert layout.group_dir(str(task), layout.ROLE_BASELINE) == str(task / "baseline")
    assert layout.group_dir(str(task), layout.ROLE_CANDIDATE) == str(task / "candidate")
    with pytest.raises(SystemExit, match="没有角色"):
        layout.group_dir(str(task), "third")

    single = Path(
        layout.init_task_dir(str(base), "mock-strong", "wx", arms=1, ts="20260101-120000")
    )
    assert single.name == "eval_mock-strong_wx_20260101-120000"
    assert sorted(p.name for p in single.iterdir()) == ["reports", "runs"]

    # 已有骨架（非空）的任务目录拒绝复用：上一场评测的现场不被覆盖
    with pytest.raises(SystemExit, match="非空"):
        layout.init_task_dir(str(base), "a1_vs_b1", "wx", arms=2, ts="20260101-120000")

    # 校验阶段失败留下的空目录可以复用（与 experiment_path 同一套守卫语义）
    (base / layout.task_subdir("empty-task", "wx", arms=1, ts="20260101-130000")).mkdir(
        parents=True
    )
    reused = layout.init_task_dir(str(base), "empty-task", "wx", arms=1, ts="20260101-130000")
    assert reused == str(base / "eval_empty-task_wx_20260101-130000")
    assert sorted(p.name for p in Path(reused).iterdir()) == ["reports", "runs"]


def test_init_task_dir_defaults_to_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """base 缺省时落 MEMBENCH_OUTPUT_DIR（默认产物根的唯一入口）。"""
    root = tmp_path / "custom-root"
    monkeypatch.setenv("MEMBENCH_OUTPUT_DIR", str(root))
    task = Path(layout.init_task_dir(None, "x", arms=1, ts="20260101-120000"))
    assert task.parent == root
    assert task.name == "eval_x_default_20260101-120000"


# ---------------------------------------------------------------------------
# resolve_dataset_dir：产物目录归一
# ---------------------------------------------------------------------------


def test_resolve_dataset_dir_accepts_dataset_and_experiment_dirs(tmp_path: Path) -> None:
    """数据集目录原样返回；实验目录按唯一数据集子目录或 dataset 提示命中。"""
    exp = tmp_path / "eval_x_wx_20260101-120000"
    dataset_dir = _fake_dataset_dir(exp, "office")

    assert layout.resolve_dataset_dir(str(dataset_dir)) == str(dataset_dir)
    assert layout.resolve_dataset_dir(str(exp)) == str(dataset_dir)
    assert layout.resolve_dataset_dir(str(exp), dataset="office") == str(dataset_dir)
    # dataset 提示不命中（另一次评测的数据集名）时退回唯一的数据集子目录
    assert layout.resolve_dataset_dir(str(exp), dataset="locomo") == str(dataset_dir)


def test_resolve_dataset_dir_requires_hint_when_multiple_datasets(tmp_path: Path) -> None:
    """实验目录下有多个数据集产物：给 dataset 提示能定位，不给则报错要求指具体目录。"""
    exp = tmp_path / "eval_x_wx_20260101-120000"
    office = _fake_dataset_dir(exp, "office")
    locomo = _fake_dataset_dir(exp, "locomo")

    assert layout.resolve_dataset_dir(str(exp), dataset="locomo") == str(locomo)
    with pytest.raises(SystemExit) as excinfo:
        layout.resolve_dataset_dir(str(exp))
    message = str(excinfo.value)
    assert "多个数据集目录" in message
    assert "office" in message and "locomo" in message
    assert str(exp / "runs") in message


def test_resolve_dataset_dir_rejects_missing_and_incomplete(tmp_path: Path) -> None:
    """入参不存在 / 数据集产物缺 metrics.json（半成品）都在读取前被拦下。"""
    with pytest.raises(SystemExit) as missing:
        layout.resolve_dataset_dir(str(tmp_path / "nope"))
    assert "产物目录不存在" in str(missing.value)

    exp = tmp_path / "eval_x_wx_20260101-120000"
    dataset_dir = _fake_dataset_dir(exp, "office", metrics=False)
    with pytest.raises(SystemExit) as incomplete:
        layout.resolve_dataset_dir(str(dataset_dir))
    assert "metrics.json" in str(incomplete.value)
    assert "产物不完整" in str(incomplete.value)


def test_resolve_dataset_dir_rejects_dialogue_dir(tmp_path: Path) -> None:
    """对话级产物目录被拒：一段对话的指标不是数据集汇总，也不能当数据集用。"""
    exp = tmp_path / "eval_x_wx_20260101-120000"
    dataset_dir = _fake_dataset_dir(exp, "office")
    dialogue_dir = _fake_dialogue_dir(exp, "office", "d 001")

    with pytest.raises(SystemExit) as excinfo:
        layout.resolve_dataset_dir(str(dialogue_dir))
    message = str(excinfo.value)
    assert "对话级产物目录" in message
    assert "d 001" in message
    assert "请传实验目录或数据集目录" in message

    # 实验目录入参仍定位到真正的数据集目录：对话目录不会被当成数据集目录
    assert layout.resolve_dataset_dir(str(exp)) == str(dataset_dir)

    # runs/ 下混进对话形状目录（非合法布局）同样不当作数据集目录
    stray_report = exp / "runs" / "stray_dialogue" / layout.DIR_REPORT
    stray_report.mkdir(parents=True)
    (stray_report / layout.FILE_METRICS).write_text(
        json.dumps({"run_id": "stray", "dialogue_id": "stray_dialogue"}),
        encoding="utf-8",
    )
    assert layout.resolve_dataset_dir(str(exp)) == str(dataset_dir)


def test_resolve_dataset_dir_rejects_legacy_layout(tmp_path: Path) -> None:
    """旧版产物（平铺 run 目录 / 缺 report/ 层级 / ab_ 实验目录）统一报不兼容。"""
    # 1) 平铺 run 目录：报告与数据在目录根
    legacy_run = tmp_path / "runs" / "mock-weak_wx_dg_20250101_000000"
    legacy_run.mkdir(parents=True)
    (legacy_run / layout.FILE_METRICS).write_text('{"run_id": "legacy"}', encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        layout.resolve_dataset_dir(str(legacy_run))
    assert layout.LEGACY_LAYOUT_MSG in str(excinfo.value)

    # 2) 旧版 A/B 实验目录：A/ B/ 各是一份平铺 run，根层没有 runs/
    legacy_ab = tmp_path / "ab_a1_vs_b1_20250101_000000"
    for arm in ("A", "B"):
        (legacy_ab / arm).mkdir(parents=True)
        (legacy_ab / arm / layout.FILE_METRICS).write_text(
            json.dumps({"run_id": arm}), encoding="utf-8"
        )
    with pytest.raises(SystemExit) as excinfo_ab:
        layout.resolve_dataset_dir(str(legacy_ab))
    assert layout.LEGACY_LAYOUT_MSG in str(excinfo_ab.value)

    # 3) 实验目录里有 runs/<dataset>，但该层缺 report/（旧结构或跑到一半）
    half_baked = tmp_path / "eval_x_wx_20250101-000000"
    (half_baked / "runs" / "office").mkdir(parents=True)
    (half_baked / "runs" / "office" / layout.FILE_METRICS).write_text(
        '{"run_id": "flat"}', encoding="utf-8"
    )
    with pytest.raises(SystemExit) as excinfo_half:
        layout.resolve_dataset_dir(str(half_baked))
    assert layout.LEGACY_LAYOUT_MSG in str(excinfo_half.value)


def test_resolve_dataset_dir_reports_missing_product(tmp_path: Path) -> None:
    """空 / 无效实验目录报"没有数据集产物"，不误报成旧版结构（原因不同，处置也不同）。"""
    # 1) 空实验目录
    empty = tmp_path / "eval_x_wx_20260101-000000"
    empty.mkdir()
    with pytest.raises(SystemExit) as excinfo:
        layout.resolve_dataset_dir(str(empty))
    message = str(excinfo.value)
    assert layout.NO_DATASET_PRODUCT_MSG in message
    assert layout.LEGACY_LAYOUT_MSG not in message

    # 2) runs/ 下只有非法目录（跑到一半崩掉的现场：对话形状目录 + 空目录）
    half = tmp_path / "eval_y_wx_20260101-000000"
    stray = half / layout.DIR_RUNS / "stray_dialogue" / layout.DIR_REPORT
    stray.mkdir(parents=True)
    (stray / layout.FILE_METRICS).write_text(
        json.dumps({"run_id": "stray", "dialogue_id": "d_stray"}), encoding="utf-8"
    )
    (half / layout.DIR_RUNS / "empty_dir").mkdir()
    with pytest.raises(SystemExit) as excinfo_half:
        layout.resolve_dataset_dir(str(half))
    assert layout.NO_DATASET_PRODUCT_MSG in str(excinfo_half.value)

    # 3) 对照组：平铺产物仍按旧版结构报错（两者文案必须区分开）
    legacy = tmp_path / "legacy_run"
    legacy.mkdir()
    (legacy / layout.FILE_METRICS).write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo_legacy:
        layout.resolve_dataset_dir(str(legacy))
    assert layout.LEGACY_LAYOUT_MSG in str(excinfo_legacy.value)
    assert layout.NO_DATASET_PRODUCT_MSG not in str(excinfo_legacy.value)


def test_group_dir_name_keeps_ascii_and_rejects_illegal() -> None:
    """组名 → 目录名：中文 label 不进目录（用 ASCII 兜底），非法字符清掉。"""
    assert layout.group_dir_name("index-opt") == "index-opt"
    # 中文 label 只有兜底目录名
    assert layout.group_dir_name("索引优化", ascii_fallback="index-opt") == "index-opt"
    assert layout.group_dir_name("索引优化") == "group"
    # 路径分隔符 / 通配符清掉，首尾点与空格去掉
    assert layout.group_dir_name("a/b:c*?. ") == "a-b-c"


def test_validate_group_labels_uniqueness_and_reserved() -> None:
    """组名校验：大小写不敏感唯一、不撞保留名、目录名是 ASCII。"""
    dirs = layout.validate_group_labels(
        ["基线", "索引优化"], fallbacks=["baseline", "index-opt"]
    )
    assert dirs == ["baseline", "index-opt"]
    # 两个中文名都退同一兜底 → 判重复
    with pytest.raises(ValueError, match="重复"):
        layout.validate_group_labels(["甲", "乙"], fallbacks=["same", "SAME"])
    with pytest.raises(ValueError, match="保留名"):
        layout.validate_group_labels(["runs"])
    with pytest.raises(ValueError, match="不能为空"):
        layout.validate_group_labels(["   "])


def test_groups_manifest_roundtrip(tmp_path: Path) -> None:
    """清单读写：角色 → 目录 → 显示名；缺失 / 坏文件返回 None（不猜）。"""
    task = Path(
        layout.init_task_dir(
            str(tmp_path),
            "ab",
            "mac",
            arms=2,
            ts="20260101-000000",
            group_labels=["基线", "索引优化"],
            group_dirs=["baseline", "index-opt"],
        )
    )
    groups = layout.read_groups_manifest(str(task))
    assert groups is not None
    assert [(g["role"], g["label"], g["dir"]) for g in groups] == [
        (layout.ROLE_BASELINE, "基线", "baseline"),
        (layout.ROLE_CANDIDATE, "索引优化", "index-opt"),
    ]
    assert layout.group_dir(str(task), layout.ROLE_CANDIDATE) == str(task / "index-opt")
    # 坏清单：返回 None，group_dir 报可操作错误
    (task / layout.FILE_GROUPS).write_text("{坏", encoding="utf-8")
    assert layout.read_groups_manifest(str(task)) is None
    with pytest.raises(SystemExit, match="groups.json"):
        layout.group_dir(str(task), layout.ROLE_BASELINE)
