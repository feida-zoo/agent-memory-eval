"""task 子命令与 --task-dir / --no-snapshot 的 CLI 面（冻结契约）。

- ``task init --name NAME [--platform P] [--groups 1|2] [--report-out DIR] [--replay]``：建任务
  目录骨架（groups=2 -> ``A/`` ``B/``；groups=1 -> ``runs/``，都带 ``reports/``；``--replay``
  额外建任务级快照池 ``snapshots/``，见 ``tests/test_replay_snapshots.py``），**stdout
  只有一行绝对路径**（脚本 ``TASK=$(...)`` 要能直接接走），非空目标目录拒绝；
- ``task status DIR``：进度视图——骨架、各组各数据集的分片账本进度（s<i> 的 status /
  题数 / 对话数 / run_id）、数据集级产物是否齐备、快照目录数与体积、下一步该做什么；
- ``--task-dir DIR``（与 ``--report-out`` 互斥）：把产物写进已存在的任务目录而不是新建
  ``eval_*`` / ``compare_*``——``run``/``evaluate``/``evaluate-compare`` 直接写进该目录，
  ``compare`` 就地读 ``DIR/A`` ``DIR/B``，``fuse`` 就地补齐数据集级产物；
- ``--no-snapshot``：关掉默认导出的对话状态快照（``run``/``evaluate``/``evaluate-compare``）。
"""

from __future__ import annotations

import io
import json
import shutil
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from membench import cli, layout
from membench.data import SAMPLE_DATASET_PATH


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """产物根与持久化适配器目录都指向 tmp_path：不落仓库、不受本机已安装适配器影响。"""
    root = tmp_path / "output-root"
    monkeypatch.setenv("MEMBENCH_OUTPUT_DIR", str(root))
    monkeypatch.delenv("MEMBENCH_ADAPTER_MODULES", raising=False)
    adapters_dir = tmp_path / "adapters"
    adapters_dir.mkdir()
    monkeypatch.setenv("MEMBENCH_ADAPTERS_DIR", str(adapters_dir))
    return root


def _dataset(path: Path, *, n_dialogues: int = 4, per: int = 2) -> Path:
    """每个对话都有题的密集数据集（对话级分片下每片都有活干）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for d in range(n_dialogues):
            handle.write(
                json.dumps(
                    {
                        "dialogue_id": f"dd{d}",
                        "sessions": [
                            [
                                {
                                    "role": "user",
                                    "content": f"dd{d} 的记录：项目截止日是周五。",
                                    "timestamp": None,
                                }
                            ]
                        ],
                        "questions": [
                            {
                                "qid": f"dd{d}_q{q}",
                                "question": "项目截止日是哪天？",
                                "gold_answer": "周五",
                                "category": "single_hop",
                                "evidence": ["项目截止日是周五"],
                            }
                            for q in range(per)
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def _init_task(base: Path, name: str, *, platform: str | None = "t", groups: int = 2) -> Path:
    """跑 ``task init`` 并返回它打印到 stdout 的任务目录（stdout 必须只有这一行路径）。"""
    argv = ["task", "init", "--name", name, "--report-out", str(base), "--groups", str(groups)]
    if platform is not None:
        argv += ["--platform", platform]
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        assert cli.main(argv) == 0
    lines = [line for line in buffer.getvalue().strip().splitlines() if line.strip()]
    assert len(lines) == 1, f"task init 的 stdout 只能有路径一行，实际: {lines}"
    return Path(lines[0])


def _run_shard(task_or_arm: Path, dataset: Path, *, shards: int, index: int) -> int:
    """在给定任务目录 / 组目录里跑一个分片（对话级分片，共享写入）。"""
    return cli.main(
        [
            "run",
            "--adapter",
            "mock-strong",
            "--dataset",
            str(dataset),
            "--judge",
            "heuristic",
            "--task-dir",
            str(task_or_arm),
            "--shards",
            str(shards),
            "--shard-index",
            str(index),
            "--shard-by",
            "dialogue",
        ]
    )


def _metrics(dataset_dir: Path) -> dict:
    return json.loads(
        (dataset_dir / layout.DIR_REPORT / layout.FILE_METRICS).read_text(encoding="utf-8")
    )


def _experiment_under(base: Path) -> Path:
    """``base`` 下唯一的实验 / 任务目录。"""
    subs = [p for p in base.iterdir() if p.is_dir()]
    assert len(subs) == 1, f"期望唯一实验目录，实际 {[p.name for p in subs]}"
    return subs[0]


# ---------------------------------------------------------------------------
# task init
# ---------------------------------------------------------------------------


def test_task_init_arms2_prints_only_abs_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """groups=2（默认）：骨架 baseline/ candidate/ + groups.json，stdout 只有绝对路径一行。"""
    base = tmp_path / "out"
    assert (
        cli.main(["task", "init", "--name", "ab", "--platform", "t", "--report-out", str(base)])
        == 0
    )
    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().splitlines() if line.strip()]
    assert len(lines) == 1, f"stdout 只能有路径一行（脚本 $() 要接走），实际: {lines}"
    task = Path(lines[0])
    assert task.is_absolute(), f"stdout 必须是绝对路径，实际 {task}"
    assert task.is_dir() and task.parent == base
    assert task.name.startswith("compare_ab_t_")
    assert sorted(p.name for p in task.iterdir()) == ["baseline", "candidate", "groups.json", "reports", "runtime"]
    # 骨架 / 下一步提示都在 stderr（不污染 stdout）
    assert "骨架: groups=2（baseline/ candidate/ + reports/）" in captured.err
    assert f"--task-dir {task / layout.DIR_GROUP_BASELINE}" in captured.err
    assert f"task status {task}" in captured.err


def test_task_init_arms1_builds_runs_skeleton(tmp_path: Path) -> None:
    """groups=1：单组骨架 runs/ + reports/，目录名前缀 eval_。"""
    task = _init_task(tmp_path / "out", "solo", groups=1)
    assert task.name.startswith("eval_solo_t_")
    assert sorted(p.name for p in task.iterdir()) == ["reports", "runs", "runtime"]


def test_task_init_default_report_out_is_output_root(
    _isolated_env: Path,
) -> None:
    """缺省 --report-out 落 output_root()（MEMBENCH_OUTPUT_DIR），缺省 groups=2。"""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        assert cli.main(["task", "init", "--name", "solo"]) == 0
    task = Path(buffer.getvalue().strip())
    assert task.parent == _isolated_env
    assert task.name.startswith("compare_solo_default_")


def test_task_init_rejects_non_empty_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """目标目录已存在且非空时拒绝新建（已有产物的目录一律不覆盖）。"""
    monkeypatch.setattr(layout, "ts_now", lambda *args, **kwargs: "20260101-120000")
    base = tmp_path / "out"
    task = _init_task(base, "ab")
    (task / "baseline" / "keep.txt").write_text("已有产物", encoding="utf-8")

    with pytest.raises(SystemExit, match="已存在且非空"):
        cli.main(["task", "init", "--name", "ab", "--platform", "t", "--report-out", str(base)])
    # 现场不被改动
    assert (task / "baseline" / "keep.txt").read_text(encoding="utf-8") == "已有产物"


# ---------------------------------------------------------------------------
# task status
# ---------------------------------------------------------------------------


def test_task_status_fresh_task_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """未跑：给出骨架与"尚无产物"，并指出下一步该跑什么。"""
    task = _init_task(tmp_path / "out", "ab")
    capsys.readouterr()
    assert cli.main(["task", "status", str(task)]) == 0
    out = capsys.readouterr().out
    assert f"任务目录: {task}" in out
    assert "骨架: groups=2（baseline/ candidate/ + reports/）" in out
    assert "组 基线（baseline）:" in out and "组 候选（candidate）:" in out
    assert out.count("数据集: （尚无产物）") == 2
    assert "下一步:" in out
    assert "--task-dir" in out and "--shard-by dialogue" in out


def test_task_status_progress_then_fuse_in_place(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """三种状态：部分完成（缺片）/ 分片跑齐（提示 fuse）/ 融合完成（产物齐备 + 汇总提示）。"""
    dataset = _dataset(tmp_path / "dense.jsonl")
    base = tmp_path / "out"
    task = _init_task(base, "ab")
    group_a = task / layout.DIR_GROUP_BASELINE
    capsys.readouterr()

    # ① 部分完成：只跑了 s0
    assert _run_shard(group_a, dataset, shards=2, index=0) == 0
    capsys.readouterr()
    assert cli.main(["task", "status", str(task)]) == 0
    out = capsys.readouterr().out
    assert "数据集 dense: 分片 1/2 完成（shard_by=dialogue，limit=None）" in out
    assert "s0: done 2 对话 / 4 题 run_id=" in out
    assert "数据集级产物: 缺失" in out
    assert "还有 1 片未完成（s1）" in out
    assert "对话目录 2 个" in out

    # ② 分片跑齐但还没融合：提示补齐数据集级产物
    assert _run_shard(group_a, dataset, shards=2, index=1) == 0
    capsys.readouterr()
    assert cli.main(["task", "status", str(task)]) == 0
    out = capsys.readouterr().out
    assert "分片 2/2 完成" in out
    assert f"python -m membench.cli fuse --task-dir {group_a}" in out

    # ③ fuse --task-dir 就地融合：补齐数据集级与任务级产物，不新建目录
    assert cli.main(["fuse", "--task-dir", str(group_a)]) == 0
    fuse_out = capsys.readouterr().out
    assert "融合完成: 8 题" in fuse_out
    assert (group_a / layout.DIR_RUNS / "dense" / layout.DIR_REPORT / layout.FILE_METRICS).is_file()
    assert (group_a / layout.DIR_REPORTS / layout.FILE_REPORT_MD).is_file()
    assert [p.name for p in base.iterdir()] == [task.name], "就地融合不该新建实验目录"

    # ④ B 组跑一遍（不分片），status 显示两组齐备并给出 compare 提示
    assert (
        cli.main(
            [
                "run",
                "--adapter",
                "mock-strong",
                "--dataset",
                str(dataset),
                "--judge",
                "heuristic",
                "--task-dir",
                str(task / layout.DIR_GROUP_CANDIDATE),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert cli.main(["task", "status", str(task)]) == 0
    out = capsys.readouterr().out
    assert out.count("数据集级产物: 已齐备") == 2
    assert "未分片（无分片账本 shards.json）" in out
    # mock-strong 不声明快照能力：快照数 0、体积 0 B
    assert "快照 0 个 / 0 B" in out
    assert f"python -m membench.cli compare --task-dir {task}" in out


def test_task_status_reports_running_shard(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """分片仍在跑（账本 status=running）：显示未跑完并提示等它结束，不催融合。"""
    task = _init_task(tmp_path / "out", "solo", groups=1)
    dataset_dir = layout.dataset_dir(str(task), "dense")
    layout.update_shard_ledger(
        dataset_dir,
        layout.shard_ledger_header(
            dataset="dense", shards=2, shard_by="dialogue", limit=None, fingerprint=None
        ),
        layout.shard_key(0),
        {
            "status": "running",
            "dialogues": ["dd0", "dd2"],
            "run_id": "mock-strong_default_dense_dshard0of2_20260101_000000",
            "started_at": "2026-01-01T00:00:00",
        },
    )
    assert cli.main(["task", "status", str(task)]) == 0
    out = capsys.readouterr().out
    assert "分片 0/2 完成（shard_by=dialogue，limit=None），1 片仍在跑（running）" in out
    assert "s0: running 2 对话 / n/a 题" in out
    assert "仍在跑（running）：等它结束后再融合" in out
    assert "还有 1 片未完成（s1）" in out
    assert "数据集级产物: 缺失" in out


def test_task_status_rejects_dir_without_skeleton(tmp_path: Path) -> None:
    """既无 runs/ 也无 A/ B/ 的目录不是任务目录：直接报错并提示 task init。"""
    empty = tmp_path / "not-a-task"
    empty.mkdir()
    with pytest.raises(SystemExit, match="没有任务骨架"):
        cli.main(["task", "status", str(empty)])
    with pytest.raises(SystemExit, match="任务目录不存在"):
        cli.main(["task", "status", str(tmp_path / "nope")])


# ---------------------------------------------------------------------------
# run --task-dir
# ---------------------------------------------------------------------------


def test_run_task_dir_writes_into_given_dir(tmp_path: Path) -> None:
    """run --task-dir：产物直接落该目录（不分片时数据集级 + 任务级报告齐备）。"""
    task = _init_task(tmp_path / "out", "solo", groups=1)
    assert (
        cli.main(
            [
                "run",
                "--adapter",
                "mock-strong",
                "--dataset",
                str(SAMPLE_DATASET_PATH),
                "--judge",
                "heuristic",
                "--task-dir",
                str(task),
            ]
        )
        == 0
    )
    dataset_dir = task / layout.DIR_RUNS / "sample_dataset"
    assert (dataset_dir / layout.DIR_REPORT / layout.FILE_METRICS).is_file()
    assert (dataset_dir / "d001" / layout.DIR_REPORT / layout.FILE_RESULTS).is_file()
    assert (task / layout.DIR_REPORTS / layout.FILE_REPORT_MD).is_file()
    assert _metrics(dataset_dir)["dataset_dir"] == str(dataset_dir)
    # 没有另外新建 eval_* 实验目录
    assert [p.name for p in (tmp_path / "out").iterdir()] == [task.name]


def test_run_task_dir_missing_dir_errors(tmp_path: Path) -> None:
    """DIR 不存在时报错并提示先跑 task init。"""
    with pytest.raises(SystemExit, match="任务目录不存在") as excinfo:
        cli.main(
            [
                "run",
                "--adapter",
                "mock-strong",
                "--dataset",
                str(SAMPLE_DATASET_PATH),
                "--judge",
                "heuristic",
                "--task-dir",
                str(tmp_path / "nope"),
            ]
        )
    assert "task init" in str(excinfo.value)


def test_run_task_dir_question_shard_rejected(tmp_path: Path) -> None:
    """共享任务目录下题目级分片被拒（多片会写同一对话目录并互相覆盖）。"""
    dataset = _dataset(tmp_path / "dense.jsonl")
    task = _init_task(tmp_path / "out", "solo", groups=1)
    with pytest.raises(SystemExit, match="只支持 --shard-by dialogue"):
        cli.main(
            [
                "run",
                "--adapter",
                "mock-strong",
                "--dataset",
                str(dataset),
                "--judge",
                "heuristic",
                "--task-dir",
                str(task),
                "--shards",
                "2",
                "--shard-index",
                "0",
            ]
        )
    # 拒绝在建任何产物之前：runs/ 还是空的
    assert list((task / layout.DIR_RUNS).iterdir()) == []


def test_run_task_dir_and_report_out_are_mutually_exclusive(tmp_path: Path) -> None:
    """--task-dir 与 --report-out 二选一（argparse 拒绝，退出码 2）。"""
    task = _init_task(tmp_path / "out", "solo", groups=1)
    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "run",
                "--adapter",
                "mock-strong",
                "--dataset",
                str(SAMPLE_DATASET_PATH),
                "--judge",
                "heuristic",
                "--task-dir",
                str(task),
                "--report-out",
                str(tmp_path / "elsewhere"),
            ]
        )
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# fuse 两种模式
# ---------------------------------------------------------------------------


def test_fuse_cli_requires_exactly_one_mode(tmp_path: Path) -> None:
    """fuse 要么 --runs + --out，要么 --task-dir；两者混用 / 都不给都报错。"""
    with pytest.raises(SystemExit, match="fuse 需要"):
        cli.main(["fuse"])
    with pytest.raises(SystemExit, match="互斥"):
        cli.main(
            ["fuse", "--task-dir", str(tmp_path), "--runs", str(tmp_path), "--out", str(tmp_path)]
        )


def test_fuse_cross_dir_mode_still_works(tmp_path: Path) -> None:
    """保留的跨目录模式：各片写各自目录，fuse --runs ... --out ... 合成一份完整实验。"""
    dataset = _dataset(tmp_path / "dense.jsonl")
    shard_dirs: list[str] = []
    for index in (0, 1):
        out = tmp_path / f"out{index}"
        assert (
            cli.main(
                [
                    "run",
                    "--adapter",
                    "mock-strong",
                    "--dataset",
                    str(dataset),
                    "--judge",
                    "heuristic",
                    "--report-out",
                    str(out),
                    "--shards",
                    "2",
                    "--shard-index",
                    str(index),
                    "--shard-by",
                    "dialogue",
                ]
            )
            == 0
        )
        shard_dirs.append(str(_experiment_under(out)))

    merged = tmp_path / "merged"
    assert cli.main(["fuse", "--runs", ",".join(shard_dirs), "--out", str(merged)]) == 0
    dataset_dir = merged / layout.DIR_RUNS / "dense"
    assert (dataset_dir / layout.DIR_REPORT / layout.FILE_METRICS).is_file()
    assert (merged / layout.DIR_REPORTS / layout.FILE_REPORT_MD).is_file()
    assert len(list(dataset_dir.glob(f"*/{layout.DIR_REPORT}/{layout.FILE_RESULTS}"))) == 4


# ---------------------------------------------------------------------------
# 快照：默认导出 / --no-snapshot
# ---------------------------------------------------------------------------

#: 支持快照能力的最小适配器插件（模块顶层注册进 ADAPTERS，经 MEMBENCH_ADAPTER_MODULES 加载）。
_SNAP_PLUGIN = '''
from pathlib import Path

from membench.adapters import ADAPTERS
from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult


class SnapAdapter(BaseMemoryAdapter):
    """状态 = 已 ingest 的消息条数，导出到 target_dir/state.json。"""

    name = "snap-mock"

    def __init__(self) -> None:
        self._messages: list[dict] = []

    def reset(self) -> None:
        self._messages = []

    def ingest(self, messages):
        self._messages.extend(messages)
        return IngestStats()

    def query(self, question):
        return QueryResult(answer="周五", retrieved=[])

    def snapshot_capabilities(self):
        return {"supported": True, "restore": False, "rebuild": False, "format_version": 1}

    def snapshot(self, target_dir, tag):
        Path(target_dir, "state.json").write_text(
            '{"tag": "%s", "n": %d}' % (tag, len(self._messages)), encoding="utf-8"
        )
        return {"n_messages": len(self._messages)}


ADAPTERS["snap-mock"] = SnapAdapter
'''


@pytest.fixture()
def _snapshot_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """注册一个支持快照的适配器（snap-mock），用例结束后把注册项清掉。"""
    from membench.adapters import ADAPTERS

    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "snap_plugin.py").write_text(_SNAP_PLUGIN, encoding="utf-8")
    monkeypatch.syspath_prepend(str(plugin_dir))
    monkeypatch.setenv("MEMBENCH_ADAPTER_MODULES", "snap_plugin")
    sys.modules.pop("snap_plugin", None)
    yield "snap-mock"
    sys.modules.pop("snap_plugin", None)
    ADAPTERS.pop("snap-mock", None)


def _run_sample(out_dir: Path, *extra: str, adapter: str = "mock-strong") -> Path:
    """用内置 sample 数据集跑一次 run，返回实验目录。"""
    assert (
        cli.main(
            [
                "run",
                "--adapter",
                adapter,
                "--dataset",
                str(SAMPLE_DATASET_PATH),
                "--judge",
                "heuristic",
                "--report-out",
                str(out_dir),
                *extra,
            ]
        )
        == 0
    )
    return _experiment_under(out_dir)


def test_run_default_exports_snapshots_and_no_snapshot_skips(
    tmp_path: Path, _snapshot_adapter: str
) -> None:
    """默认导出快照（每对话 memory_snapshot/manifest.json）；--no-snapshot 一份都不导。"""
    exp = _run_sample(tmp_path / "out1", adapter=_snapshot_adapter)
    dataset_dir = exp / layout.DIR_RUNS / "sample_dataset"
    snapshots = sorted(
        p for p in dataset_dir.iterdir() if p.is_dir() and (p / layout.DIR_SNAPSHOT).is_dir()
    )
    assert snapshots, "默认应导出逐对话快照"
    for dlg_dir in snapshots:
        assert (dlg_dir / layout.DIR_SNAPSHOT / "manifest.json").is_file()
    assert _metrics(dataset_dir)["state_snapshot"]["role"] == "exported"

    exp2 = _run_sample(tmp_path / "out2", "--no-snapshot", adapter=_snapshot_adapter)
    dataset_dir2 = exp2 / layout.DIR_RUNS / "sample_dataset"
    assert not list(dataset_dir2.glob(f"*/{layout.DIR_SNAPSHOT}")), "--no-snapshot 不该导出快照"
    assert "state_snapshot" not in _metrics(dataset_dir2)


def test_run_default_snapshot_warns_once_for_unsupported_adapter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """适配器不支持快照：默认只警告一次并跳过（metrics 记 unsupported），--no-snapshot 则静默。"""
    exp = _run_sample(tmp_path / "out1")
    out = capsys.readouterr().out
    assert out.count("未声明快照能力") == 1
    dataset_dir = exp / layout.DIR_RUNS / "sample_dataset"
    assert not list(dataset_dir.glob(f"*/{layout.DIR_SNAPSHOT}"))
    assert _metrics(dataset_dir)["state_snapshot"]["role"] == "unsupported"

    _run_sample(tmp_path / "out2", "--no-snapshot")
    assert "未声明快照能力" not in capsys.readouterr().out
    quiet_dir = _experiment_under(tmp_path / "out2") / layout.DIR_RUNS / "sample_dataset"
    assert "state_snapshot" not in _metrics(quiet_dir)


def test_task_status_counts_snapshot_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], _snapshot_adapter: str
) -> None:
    """task status 统计快照目录数与总字节数（扫 <对话>/memory_snapshot/）。"""
    dataset = _dataset(tmp_path / "dense.jsonl")
    task = _init_task(tmp_path / "out", "solo", groups=1)
    capsys.readouterr()
    assert (
        cli.main(
            [
                "run",
                "--adapter",
                _snapshot_adapter,
                "--dataset",
                str(dataset),
                "--judge",
                "heuristic",
                "--task-dir",
                str(task),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert cli.main(["task", "status", str(task)]) == 0
    out = capsys.readouterr().out
    # 4 个对话各一份 state.json，体积逐字节计入（非 0 B）
    assert "快照 4 个 / " in out
    assert "快照 4 个 / 0 B" not in out


# ---------------------------------------------------------------------------
# compare / evaluate / evaluate-compare 的 --task-dir（依赖各模块的 task_dir 参数）
# ---------------------------------------------------------------------------


def _two_arm_task_with_products(tmp_path: Path) -> tuple[Path, Path]:
    """建一个双组任务目录并在两组各跑一次（不分片），返回 (任务目录, 产物根)。"""
    base = tmp_path / "out"
    task = _init_task(base, "ab")
    for group in (layout.DIR_GROUP_BASELINE, layout.DIR_GROUP_CANDIDATE):
        assert (
            cli.main(
                [
                    "run",
                    "--adapter",
                    "mock-strong",
                    "--dataset",
                    str(SAMPLE_DATASET_PATH),
                    "--judge",
                    "heuristic",
                    "--task-dir",
                    str(task / group),
                ]
            )
            == 0
        )
    return task, base


def test_compare_task_dir_in_place(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """compare --task-dir：就地读 A/ 与 B/（不必再传两组），comparison.md 落任务目录的 reports/。"""
    task, base = _two_arm_task_with_products(tmp_path)
    capsys.readouterr()
    assert cli.main(["compare", "--task-dir", str(task)]) == 0
    assert "compare_dir" in capsys.readouterr().out
    comparison = task / layout.DIR_REPORTS / layout.FILE_COMPARISON_MD
    assert comparison.is_file(), "对比报告应落任务目录的 reports/"
    assert "门禁结论： PASS" in comparison.read_text(encoding="utf-8")
    # 不新建 compare_* 目录、不复制两组：任务目录结构与内容保持原位
    assert [p.name for p in base.iterdir()] == [task.name]
    group_a_metrics = (
        task / layout.DIR_GROUP_BASELINE / layout.DIR_RUNS / "sample_dataset" / layout.DIR_REPORT / layout.FILE_METRICS
    )
    assert group_a_metrics.is_file()


def test_compare_argument_contract(tmp_path: Path) -> None:
    """两组入参校验：无 --task-dir 时必填；--task-dir 时缺组 / 目录不存在都当场报错。"""
    under_task = _init_task(tmp_path / "out", "ab")
    with pytest.raises(SystemExit, match="compare 需要 --baseline"):
        cli.main(["compare"])
    with pytest.raises(SystemExit, match="compare 需要 --baseline"):
        cli.main(["compare", "--baseline", str(under_task / "baseline")])
    with pytest.raises(SystemExit, match="任务目录不存在"):
        cli.main(["compare", "--task-dir", str(tmp_path / "nope")])

    shutil.rmtree(under_task / layout.DIR_GROUP_CANDIDATE)
    with pytest.raises(SystemExit, match="组目录不存在"):
        cli.main(["compare", "--task-dir", str(under_task)])


def test_evaluate_task_dir_writes_into_given_dir(tmp_path: Path) -> None:
    """evaluate --task-dir：产物直接写进该目录（各数据集 runs/<dataset>/ + reports/）。"""
    dataset = _dataset(tmp_path / "dense.jsonl")
    task = _init_task(tmp_path / "out", "solo", groups=1)
    assert (
        cli.main(
            [
                "evaluate",
                "--adapter",
                "mock-strong",
                "--judge",
                "heuristic",
                "--dataset",
                str(dataset),
                "--task-dir",
                str(task),
            ]
        )
        == 0
    )
    assert (task / layout.DIR_RUNS / "dense" / layout.DIR_REPORT / layout.FILE_METRICS).is_file()
    assert (task / layout.DIR_REPORTS / layout.FILE_SUMMARY).is_file()
    assert [p.name for p in (tmp_path / "out").iterdir()] == [task.name]


def test_evaluate_no_snapshot_passthrough(tmp_path: Path, _snapshot_adapter: str) -> None:
    """evaluate --no-snapshot：快照不导出（该开关要一路传到 Runner）。"""
    dataset = _dataset(tmp_path / "dense.jsonl")
    base = tmp_path / "out"
    assert (
        cli.main(
            [
                "evaluate",
                "--adapter",
                _snapshot_adapter,
                "--judge",
                "heuristic",
                "--dataset",
                str(dataset),
                "--no-snapshot",
                "--report-out",
                str(base),
            ]
        )
        == 0
    )
    dataset_dir = _experiment_under(base) / layout.DIR_RUNS / "dense"
    assert not list(dataset_dir.glob(f"*/{layout.DIR_SNAPSHOT}"))
    assert "state_snapshot" not in _metrics(dataset_dir)


def test_evaluate_compare_task_dir_writes_into_arms(tmp_path: Path) -> None:
    """evaluate-compare --task-dir：两组分别写 DIR/A、DIR/B，对比报告落 DIR/reports/。"""
    dataset = _dataset(tmp_path / "dense.jsonl", n_dialogues=2, per=1)
    base = tmp_path / "out"
    task = _init_task(base, "ab")
    assert (
        cli.main(
            [
                "evaluate-compare",
                "--baseline-adapter",
                "mock-strong",
                "--candidate-adapter",
                "mock-strong",
                "--judge",
                "heuristic",
                "--dataset",
                str(dataset),
                "--task-dir",
                str(task),
            ]
        )
        == 0
    )
    for group in (layout.DIR_GROUP_BASELINE, layout.DIR_GROUP_CANDIDATE):
        assert (
            task / group / layout.DIR_RUNS / "dense" / layout.DIR_REPORT / layout.FILE_METRICS
        ).is_file()
        assert (task / group / layout.DIR_REPORTS / layout.FILE_SUMMARY).is_file()
    assert (task / layout.DIR_REPORTS / layout.FILE_COMPARISON_MD).is_file()
    assert [p.name for p in base.iterdir()] == [task.name], "就地对比不该另建 compare_* 目录"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
