"""回放任务的任务级快照池：内容去重、逐对话条目、派生索引与并行分片回放。

回放（``--replay``）在恢复每个对话之后、第一道 query 之前把状态导出到**任务级**
内容寻址池 ``<任务>/snapshots/``：桶名 = 状态内容 sha256 前 12 位（内容哈希沿用
``snapshot.dir_sha256``，排除 manifest 自身），每个对话一个**自足条目文件**
``snapshots/entries/<臂>/<数据集>/<对话>.json``（真源），``snapshots/index.json``
由条目 + 桶清单重算（派生视图，fuse / task status 收尾刷新）。A/B 回放同一源时
两臂指向同一批桶（只存一份），内容不同则各存一份——差异本身就是"回放不保真"的证据。

条目一对话一文件，对话级分片的各进程只写自己的对话；桶落位只靠一次原子
``os.replace`` 判"新建 / 复用" ⇒ 并行回放无锁，这里用**真子进程**两片并跑来验证。

``task init --replay`` 建出 ``snapshots/``；``--no-snapshot`` 回放既不写桶也不写条目。
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from membench import cli, layout
from membench import snapshot as snapshot_mod
from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
from membench.runner import Runner

#: 数据集文件名 stem（= 数据集目录名与索引里的 dataset 段）。
DATASET = "ds"

#: 仓库里的 `membench/` 工程根（子进程用 ``python -m membench.cli`` 跑）。
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """产物根与持久化适配器目录都指向 tmp_path：不落仓库、不受本机适配器影响。"""
    root = tmp_path / "output-root"
    monkeypatch.setenv("MEMBENCH_OUTPUT_DIR", str(root))
    monkeypatch.setenv("MEMBENCH_ADAPTERS_DIR", str(tmp_path / "adapters"))
    monkeypatch.delenv("MEMBENCH_ADAPTER_MODULES", raising=False)
    return root


class _FileMemoryAdapter(BaseMemoryAdapter):
    """文件型记忆系统：状态写在 data_dir/memories.jsonl，支持 snapshot / restore。"""

    name = "mock-file"

    def __init__(self, data_dir: Path, *, answer_suffix: str = "") -> None:
        self.data_dir = Path(data_dir)
        self.answer_suffix = answer_suffix

    def reset(self) -> None:
        shutil.rmtree(self.data_dir, ignore_errors=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def ingest(self, messages: list[dict]) -> IngestStats:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with (self.data_dir / "memories.jsonl").open("a", encoding="utf-8") as handle:
            for msg in messages:
                handle.write(json.dumps(msg, ensure_ascii=False) + "\n")
        return IngestStats(input_tokens=7, llm_calls=1)

    def query(self, question: str) -> QueryResult:
        path = self.data_dir / "memories.jsonl"
        lines = (
            [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if path.is_file()
            else []
        )
        return QueryResult(answer="周五" + self.answer_suffix, retrieved=lines)

    def snapshot_capabilities(self) -> dict:
        return {
            "supported": True,
            "restore": True,
            "rebuild": False,
            "format_version": 1,
            "compatible_from": [1],
        }

    def snapshot(self, target_dir: str, tag: str) -> dict:
        dest = Path(target_dir) / "memory"
        shutil.rmtree(dest, ignore_errors=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.data_dir.is_dir():
            shutil.copytree(self.data_dir, dest)
        else:
            dest.mkdir(parents=True, exist_ok=True)
        return {"layers": ["raw"]}

    def restore(self, source_dir: str, tag: str) -> None:
        src = Path(source_dir) / "memory"
        shutil.rmtree(self.data_dir, ignore_errors=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, self.data_dir, dirs_exist_ok=True)


def _dataset(path: Path, marker: str = "") -> Path:
    """两个对话各一条记录 + 一道题；``marker`` 改内容以造出不同的状态。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in (1, 2):
        rows.append(
            {
                "dialogue_id": f"d{i}",
                "sessions": [
                    [
                        {
                            "role": "user",
                            "content": f"第{i}次记录：项目截止日是周五。{marker}",
                            "timestamp": None,
                        }
                    ]
                ],
                "questions": [
                    {
                        "qid": f"d{i}_q1",
                        "question": "项目截止日是哪天？",
                        "gold_answer": "周五",
                        "category": "single_hop",
                        "evidence": ["项目截止日是周五"],
                    }
                ],
            }
        )
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _metrics(dataset_dir: Path) -> dict:
    return json.loads(
        (dataset_dir / layout.DIR_REPORT / layout.FILE_METRICS).read_text(encoding="utf-8")
    )


def _dialogue_metrics(experiment: Path, dialogue_id: str) -> dict:
    return json.loads(
        (
            Path(layout.dialogue_report_dir(str(experiment), DATASET, dialogue_id))
            / layout.FILE_METRICS
        ).read_text(encoding="utf-8")
    )


def _full_run(tmp_path: Path, *, name: str, dataset: Path, answers: str = "") -> Path:
    """全量 run（含快照），返回数据集目录。"""
    out = Runner(
        _FileMemoryAdapter(tmp_path / f"mem-{name}", answer_suffix=answers),
        str(dataset),
        judge="heuristic",
        out_dir=str(tmp_path / name),
    ).run()
    assert out is not None
    return Path(out)


def _replay(
    tmp_path: Path,
    *,
    name: str,
    out_dir: Path,
    replay: Path,
    dataset: Path,
    snapshot: bool = True,
) -> Path:
    """回放 run，返回数据集目录。"""
    out = Runner(
        _FileMemoryAdapter(tmp_path / f"mem-{name}"),
        str(dataset),
        judge="heuristic",
        out_dir=str(out_dir),
        replay=str(replay),
        snapshot=snapshot,
    ).run()
    assert out is not None
    return Path(out)


def _init_task(base: Path, name: str, *, arms: int = 2, replay: bool = True) -> Path:
    """``task init``（或 layout 的同一入口），返回任务目录。stdout 只允许一行路径。"""
    if replay:
        task = layout.init_task_dir(str(base), name, "t", arms=arms, replay=True)
        return Path(task)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        assert cli.main(["task", "init", "--name", name, "--platform", "t",
                         "--report-out", str(base), "--arms", str(arms)]) == 0
    return Path(buffer.getvalue().strip())


def _index(task: Path) -> dict:
    """派生索引：从条目树 + 桶清单重算并物化（与 fuse / task status 的口径一致）。"""
    return layout.materialize_snapshot_index(str(task / layout.DIR_SNAPSHOTS))


def _entry_path(task: Path, arm: str, dialogue_id: str) -> Path:
    """逐对话条目文件路径（``snapshots/entries/<臂>/<数据集>/<对话>.json``）。"""
    return Path(
        layout.snapshot_entry_path(
            str(task / layout.DIR_SNAPSHOTS), arm, DATASET, dialogue_id
        )
    )


def _entries(task: Path, arm: str) -> dict[str, dict]:
    """某臂的全部条目：``{对话 id: 条目内容}``（按文件路径读，不经过索引）。"""
    root = task / layout.DIR_SNAPSHOTS / layout.DIR_SNAPSHOT_ENTRIES / (
        layout.snapshot_arm_segment(arm)
    )
    if not root.is_dir():
        return {}
    found: dict[str, dict] = {}
    for path in sorted(root.glob("*/*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        found[data["dialogue_id"]] = data
    return found


def _bucket_dirs(task: Path) -> list[Path]:
    pool = task / layout.DIR_SNAPSHOTS
    return [pool / digest for digest in layout.list_snapshot_buckets(str(pool))]


def _run_cli(argv: list[str]) -> tuple[str, str, int]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(argv)
    return out.getvalue(), err.getvalue(), code


# ---------------------------------------------------------------------------
# task init --replay
# ---------------------------------------------------------------------------


def test_task_init_replay_builds_snapshots_dir(tmp_path: Path) -> None:
    """双臂回放任务骨架 {A,B,reports,snapshots}；单臂 {runs,reports,snapshots}。"""
    base = tmp_path / "out"
    out, err, code = _run_cli(
        ["task", "init", "--name", "ab", "--platform", "t", "--report-out", str(base),
         "--arms", "2", "--replay"]
    )
    assert code == 0
    lines = [line for line in out.strip().splitlines() if line.strip()]
    assert len(lines) == 1, f"stdout 只能有路径一行，实际: {lines}"
    task = Path(lines[0])
    assert sorted(path.name for path in task.iterdir()) == ["A", "B", "reports", "snapshots"]
    assert "snapshots/" in err and "--replay" in err

    single = _init_task(base, "one", arms=1)
    assert sorted(path.name for path in single.iterdir()) == ["reports", "runs", "snapshots"]


def test_task_init_without_replay_has_no_snapshots_dir(tmp_path: Path) -> None:
    """不带 --replay 时行为不变：不建 snapshots/（普通任务不落池）。"""
    task = _init_task(tmp_path / "out", "plain", arms=2, replay=False)
    assert sorted(path.name for path in task.iterdir()) == ["A", "B", "reports"]


# ---------------------------------------------------------------------------
# 单臂回放：桶 + 索引
# ---------------------------------------------------------------------------


def test_single_arm_replay_exports_content_addressed_pool(tmp_path: Path) -> None:
    """单臂回放：逐对话落桶与条目、派生索引三段齐备、metrics 记 snapshots_dir 与计数。"""
    dataset = _dataset(tmp_path / "ds.jsonl")
    source = _full_run(tmp_path, name="src", dataset=dataset)
    task = _init_task(tmp_path / "out", "solo", arms=1)

    dataset_dir = _replay(
        tmp_path, name="replay", out_dir=task, replay=source, dataset=dataset
    )

    buckets = _bucket_dirs(task)
    index = _index(task)
    assert len(buckets) == 2
    assert index["schema"] == layout.SNAPSHOT_POOL_SCHEMA
    assert sorted(index["buckets"]) == sorted(path.name for path in buckets)
    assert all(len(digest) == 12 for digest in index["buckets"])

    # 逐对话条目（真源）：自足，且与派生索引 / 桶一一对上
    entries = _entries(task, "")
    assert sorted(entries) == ["d1", "d2"]
    for dialogue_id in ("d1", "d2"):
        entry = entries[dialogue_id]
        digest = index["entries"][layout.snapshot_entry_key("", DATASET, dialogue_id)]
        assert entry["digest"] == digest
        assert entry["arm"] == "" and entry["dataset"] == DATASET
        assert entry["state_sha256"] == index["buckets"][digest]["state_sha256"]
        assert entry["bytes"] == index["buckets"][digest]["bytes"]
        assert entry["n_files"] >= 1
        assert entry["reused"] is False
        assert entry["source"] == str(source)
        assert entry["source_manifest_sha256"]
        assert entry["run_id"] and entry["written_at"]

    # 桶内清单：entry 与普通快照一致，另记回放来源
    for dialogue_id in ("d1", "d2"):
        digest = index["entries"][layout.snapshot_entry_key("", DATASET, dialogue_id)]
        manifest_path = task / layout.DIR_SNAPSHOTS / digest / snapshot_mod.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["complete"] is True
        assert manifest["dialogue_id"] == dialogue_id
        assert manifest["entry"]["sha256"] == index["buckets"][digest]["state_sha256"]
        assert manifest["source"]["kind"] == "replay"
        assert manifest["source"]["replay"] == str(source)
        assert manifest["source"]["replay_mode"] == "restore"
        assert manifest["source"]["source_manifest_sha256"]

    arm = index["arms"][""]
    assert arm["source"] == str(source)
    assert (arm["n_dialogues"], arm["n_reused"], arm["n_new"]) == (2, 0, 2)

    state = _metrics(dataset_dir)["state_snapshot"]
    assert state["role"] == "replayed"
    assert state["snapshots_dir"] == str(task / layout.DIR_SNAPSHOTS)
    assert (state["n_buckets"], state["n_reused"], state["n_new"]) == (2, 0, 2)

    d1_state = _dialogue_metrics(task, "d1")["state_snapshot"]
    assert d1_state["bucket"] == index["entries"][layout.snapshot_entry_key("", DATASET, "d1")]
    assert d1_state["reused"] is False


# ---------------------------------------------------------------------------
# 两臂
# ---------------------------------------------------------------------------


def test_two_arms_same_source_share_one_bucket_per_dialogue(tmp_path: Path) -> None:
    """A/B 回放同一源：只产生一份桶，两臂在索引里指向同一 digest，status 报"共用"。"""
    dataset = _dataset(tmp_path / "ds.jsonl")
    source = _full_run(tmp_path, name="src", dataset=dataset)
    task = _init_task(tmp_path / "out", "ab", arms=2)

    _replay(tmp_path, name="a", out_dir=task / layout.DIR_ARM_A, replay=source, dataset=dataset)
    buckets_after_a = [path.name for path in _bucket_dirs(task)]
    _replay(tmp_path, name="b", out_dir=task / layout.DIR_ARM_B, replay=source, dataset=dataset)

    # B 臂没有新增桶：状态内容一致 → 复用 A 臂写下的桶
    assert [path.name for path in _bucket_dirs(task)] == buckets_after_a
    assert len(buckets_after_a) == 2

    index = _index(task)
    for dialogue_id in ("d1", "d2"):
        digest_a = index["entries"][layout.snapshot_entry_key("A", DATASET, dialogue_id)]
        digest_b = index["entries"][layout.snapshot_entry_key("B", DATASET, dialogue_id)]
        assert digest_a == digest_b
    assert index["arms"]["A"]["n_new"] == 2 and index["arms"]["A"]["n_reused"] == 0
    assert index["arms"]["B"]["n_new"] == 0 and index["arms"]["B"]["n_reused"] == 2

    # 条目也是两臂各一份：A 新写、B 复用同一个桶
    entries_a = _entries(task, layout.DIR_ARM_A)
    entries_b = _entries(task, layout.DIR_ARM_B)
    assert sorted(entries_a) == ["d1", "d2"] and sorted(entries_b) == ["d1", "d2"]
    assert all(entry["reused"] is False for entry in entries_a.values())
    assert all(entry["reused"] is True for entry in entries_b.values())
    for dialogue_id in ("d1", "d2"):
        assert entries_a[dialogue_id]["digest"] == entries_b[dialogue_id]["digest"]

    out, _err, code = _run_cli(["task", "status", str(task)])
    assert code == 0
    assert "2 个桶" in out
    assert "A 臂与 B 臂快照 2/2 逐对话一致（共用 2 个桶）" in out


def test_two_arms_different_sources_keep_separate_buckets(tmp_path: Path) -> None:
    """两臂回放不同源：内容不同 → 各存一份桶，status 列出差异对话数。"""
    dataset_a = _dataset(tmp_path / "a" / "ds.jsonl", marker="（甲）")
    dataset_b = _dataset(tmp_path / "b" / "ds.jsonl", marker="（乙）")
    source_a = _full_run(tmp_path, name="src-a", dataset=dataset_a)
    source_b = _full_run(tmp_path, name="src-b", dataset=dataset_b)
    task = _init_task(tmp_path / "out", "ab", arms=2)

    _replay(tmp_path, name="a", out_dir=task / layout.DIR_ARM_A, replay=source_a, dataset=dataset_a)
    _replay(tmp_path, name="b", out_dir=task / layout.DIR_ARM_B, replay=source_b, dataset=dataset_b)

    index = _index(task)
    assert len(_bucket_dirs(task)) == 4
    for dialogue_id in ("d1", "d2"):
        digest_a = index["entries"][layout.snapshot_entry_key("A", DATASET, dialogue_id)]
        digest_b = index["entries"][layout.snapshot_entry_key("B", DATASET, dialogue_id)]
        assert digest_a != digest_b

    out, _err, code = _run_cli(["task", "status", str(task)])
    assert code == 0
    assert "A 臂与 B 臂快照有差异" in out
    assert "内容不同 2" in out


# ---------------------------------------------------------------------------
# --no-snapshot / 链式回放 / 源解析
# ---------------------------------------------------------------------------


def test_no_snapshot_replay_writes_no_pool(tmp_path: Path) -> None:
    """--no-snapshot 回放：不写桶、不写索引，metrics 也不声称有池。"""
    dataset = _dataset(tmp_path / "ds.jsonl")
    source = _full_run(tmp_path, name="src", dataset=dataset)
    task = _init_task(tmp_path / "out", "solo", arms=1)

    dataset_dir = _replay(
        tmp_path, name="replay", out_dir=task, replay=source, dataset=dataset, snapshot=False
    )

    assert not (task / layout.DIR_SNAPSHOTS / layout.FILE_SNAPSHOT_INDEX).exists()
    assert _bucket_dirs(task) == []
    state = _metrics(dataset_dir)["state_snapshot"]
    assert state["role"] == "replayed"
    assert "snapshots_dir" not in state and "n_buckets" not in state


def test_chained_replay_from_replay_task(tmp_path: Path) -> None:
    """链式回放：run --task-dir <task2>/A --replay <task1>/A（源是上一轮回放任务）。"""
    dataset = _dataset(tmp_path / "ds.jsonl")
    source = _full_run(tmp_path, name="src", dataset=dataset)
    task1 = _init_task(tmp_path / "out", "first", arms=2)
    _replay(
        tmp_path, name="t1-a", out_dir=task1 / layout.DIR_ARM_A, replay=source, dataset=dataset
    )

    task2 = _init_task(tmp_path / "out", "second", arms=2)
    dataset_dir = _replay(
        tmp_path, name="t2-a", out_dir=task2 / layout.DIR_ARM_A,
        replay=task1 / layout.DIR_ARM_A, dataset=dataset,
    )

    # 源解析走了池：记录用的是哪个任务 / 哪个臂
    state = _metrics(dataset_dir)["state_snapshot"]
    assert state["root"] == str(task1 / layout.DIR_ARM_A)

    bucket_manifest = json.loads(
        (
            next(path for path in _bucket_dirs(task2))
            / snapshot_mod.MANIFEST_NAME
        ).read_text(encoding="utf-8")
    )
    assert bucket_manifest["source"]["source_pool"] == str(task1 / layout.DIR_SNAPSHOTS)
    assert bucket_manifest["source"]["source_arm"] == layout.DIR_ARM_A
    assert len(_bucket_dirs(task2)) == 2

    # 单臂任务根作源（索引里只有一个臂）也自动可用
    task3 = _init_task(tmp_path / "out", "third", arms=2)
    _replay(
        tmp_path, name="t3-a", out_dir=task3 / layout.DIR_ARM_A,
        replay=task1, dataset=dataset,
    )
    assert len(_bucket_dirs(task3)) == 2


def test_multi_arm_task_root_as_source_requires_arm_dir(tmp_path: Path) -> None:
    """任务根有多个臂时拒绝并提示改传具体臂目录（两臂状态可能不同）。"""
    dataset = _dataset(tmp_path / "ds.jsonl")
    source = _full_run(tmp_path, name="src", dataset=dataset)
    task = _init_task(tmp_path / "out", "ab", arms=2)
    for arm in (layout.DIR_ARM_A, layout.DIR_ARM_B):
        _replay(
            tmp_path, name=f"m-{arm}", out_dir=task / arm, replay=source, dataset=dataset
        )

    with pytest.raises(SystemExit, match="有 2 个臂") as excinfo:
        _replay(tmp_path, name="from-root", out_dir=tmp_path / "replay-root",
                replay=task, dataset=dataset)
    assert f"--replay 指向具体臂目录（如 {task / layout.DIR_ARM_A}）" in str(excinfo.value)


def test_missing_bucket_and_corrupt_entry_are_reported(tmp_path: Path) -> None:
    """池源不全时拒绝：条目指向的桶缺失 / 条目损坏都给出可操作的报错。"""
    dataset = _dataset(tmp_path / "ds.jsonl")
    source = _full_run(tmp_path, name="src", dataset=dataset)
    task = _init_task(tmp_path / "out", "first", arms=2)
    _replay(tmp_path, name="t1-a", out_dir=task / layout.DIR_ARM_A, replay=source, dataset=dataset)

    dropped = _bucket_dirs(task)[0]
    shutil.rmtree(dropped)
    with pytest.raises(SystemExit, match="条目指向的桶不存在"):
        _replay(tmp_path, name="t2-a", out_dir=tmp_path / "second" / "A",
                replay=task / layout.DIR_ARM_A, dataset=dataset)

    # 条目损坏：读到坏 JSON 直接报错，不猜
    _entry_path(task, layout.DIR_ARM_A, "d1").write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="快照池条目损坏"):
        _replay(tmp_path, name="t2-b", out_dir=tmp_path / "third" / "A",
                replay=task / layout.DIR_ARM_A, dataset=dataset)


# ---------------------------------------------------------------------------
# 桶冲突 / 损坏
# ---------------------------------------------------------------------------


def _copy_bucket(task: Path, bucket: Path, target: Path) -> Path:
    """把桶复制进另一个任务的池（模拟同名桶已存在），返回副本路径。"""
    dest = target / layout.DIR_SNAPSHOTS / bucket.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(bucket, dest)
    return dest


def test_bucket_hash_collision_is_rejected(tmp_path: Path) -> None:
    """同名桶内容哈希对不上（前缀撞车 / 桶被改）：拒绝，不静默复用。"""
    dataset = _dataset(tmp_path / "ds.jsonl")
    source = _full_run(tmp_path, name="src", dataset=dataset)
    first = _init_task(tmp_path / "out", "first", arms=1)
    _replay(tmp_path, name="first", out_dir=first, replay=source, dataset=dataset)

    bucket = _bucket_dirs(first)[0]
    second = _init_task(tmp_path / "out", "second", arms=1)
    copied = _copy_bucket(first, bucket, second)
    manifest = json.loads((copied / snapshot_mod.MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["entry"]["sha256"] = "0" * 64
    (copied / snapshot_mod.MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )

    with pytest.raises(SystemExit, match="内容哈希不一致"):
        _replay(tmp_path, name="second", out_dir=second, replay=source, dataset=dataset)


def test_bucket_corrupt_manifest_is_rejected(tmp_path: Path) -> None:
    """桶内清单损坏：报"快照清单损坏"，而不是当成可复用桶。"""
    dataset = _dataset(tmp_path / "ds.jsonl")
    source = _full_run(tmp_path, name="src", dataset=dataset)
    first = _init_task(tmp_path / "out", "first", arms=1)
    _replay(tmp_path, name="first", out_dir=first, replay=source, dataset=dataset)

    bucket = _bucket_dirs(first)[0]
    second = _init_task(tmp_path / "out", "second", arms=1)
    copied = _copy_bucket(first, bucket, second)
    (copied / snapshot_mod.MANIFEST_NAME).write_text("{broken", encoding="utf-8")

    with pytest.raises(SystemExit, match="快照清单损坏"):
        _replay(tmp_path, name="second", out_dir=second, replay=source, dataset=dataset)


# ---------------------------------------------------------------------------
# 逐对话条目的单文件语义：覆写校验 / 派生索引幂等 / 旧产物兼容
# ---------------------------------------------------------------------------


def test_entry_overwrite_must_keep_same_digest(tmp_path: Path) -> None:
    """同一对话重复写条目按"覆盖自己的条目"处理，但 digest 变了必须报错。"""
    dataset = _dataset(tmp_path / "ds.jsonl")
    source = _full_run(tmp_path, name="src", dataset=dataset)
    task = _init_task(tmp_path / "out", "solo", arms=1)
    _replay(tmp_path, name="replay", out_dir=task, replay=source, dataset=dataset)

    pool = str(task / layout.DIR_SNAPSHOTS)
    entry = layout.read_snapshot_entry(pool, "", DATASET, "d1")
    assert entry is not None

    # 原样重复写：允许（同片重跑的场景）
    layout.write_snapshot_entry(pool, entry)
    assert layout.read_snapshot_entry(pool, "", DATASET, "d1") == entry

    # digest 变了：状态与桶对不上，拒绝
    changed = dict(entry, digest="0" * layout.SNAPSHOT_BUCKET_CHARS)
    with pytest.raises(SystemExit, match="条目已存在且内容不同"):
        layout.write_snapshot_entry(pool, changed)
    assert layout.read_snapshot_entry(pool, "", DATASET, "d1") == entry


def test_materialize_snapshot_index_is_idempotent(tmp_path: Path) -> None:
    """派生索引连跑两次结果一致（同字节落盘），且等于 build 的纯计算结果。"""
    dataset = _dataset(tmp_path / "ds.jsonl")
    source = _full_run(tmp_path, name="src", dataset=dataset)
    task = _init_task(tmp_path / "out", "solo", arms=1)
    _replay(tmp_path, name="replay", out_dir=task, replay=source, dataset=dataset)

    pool = str(task / layout.DIR_SNAPSHOTS)
    first = layout.materialize_snapshot_index(pool)
    first_bytes = (task / layout.DIR_SNAPSHOTS / layout.FILE_SNAPSHOT_INDEX).read_bytes()
    second = layout.materialize_snapshot_index(pool)

    assert first == second == layout.build_snapshot_index(pool)
    assert (
        task / layout.DIR_SNAPSHOTS / layout.FILE_SNAPSHOT_INDEX
    ).read_bytes() == first_bytes
    assert sorted(first["entries"]) == [
        layout.snapshot_entry_key("", DATASET, dialogue_id) for dialogue_id in ("d1", "d2")
    ]


def test_legacy_pool_without_entries_still_replays(tmp_path: Path) -> None:
    """旧产物（只有 index.json、没有 entries/）仍可作 --replay 源。"""
    dataset = _dataset(tmp_path / "ds.jsonl")
    source = _full_run(tmp_path, name="src", dataset=dataset)
    first = _init_task(tmp_path / "out", "first", arms=2)
    _replay(
        tmp_path, name="t1-a", out_dir=first / layout.DIR_ARM_A,
        replay=source, dataset=dataset,
    )
    # 先把派生索引物化，再删掉条目树，模拟旧版回放任务留下的池
    layout.materialize_snapshot_index(str(first / layout.DIR_SNAPSHOTS))
    shutil.rmtree(first / layout.DIR_SNAPSHOTS / layout.DIR_SNAPSHOT_ENTRIES)
    legacy_index = layout.read_snapshot_index(str(first / layout.DIR_SNAPSHOTS))
    assert legacy_index is not None and len(legacy_index["entries"]) == 2

    # status 仍能展示旧池，且不得拿"空条目"重算覆盖唯一的索引记录
    out, _err, code = _run_cli(["task", "status", str(first)])
    assert code == 0
    assert "2 个桶" in out
    assert layout.read_snapshot_index(str(first / layout.DIR_SNAPSHOTS)) == legacy_index

    second = _init_task(tmp_path / "out", "second", arms=1)
    _replay(
        tmp_path, name="t2", out_dir=second,
        replay=first / layout.DIR_ARM_A, dataset=dataset,
    )
    assert sorted(_entries(second, "")) == ["d1", "d2"]
    assert len(_bucket_dirs(second)) == 2


def test_entry_tree_wins_over_stale_or_corrupt_index(tmp_path: Path) -> None:
    """条目齐全时完全不读 index.json：索引坏了也能回放，status 还会顺手重建它。"""
    dataset = _dataset(tmp_path / "ds.jsonl")
    source = _full_run(tmp_path, name="src", dataset=dataset)
    task = _init_task(tmp_path / "out", "first", arms=2)
    _replay(
        tmp_path, name="t1-a", out_dir=task / layout.DIR_ARM_A,
        replay=source, dataset=dataset,
    )
    index_path = task / layout.DIR_SNAPSHOTS / layout.FILE_SNAPSHOT_INDEX
    index_path.write_text("{not json", encoding="utf-8")

    # 条目齐全：回放成功（不碰坏索引）
    second = _init_task(tmp_path / "out", "second", arms=1)
    _replay(
        tmp_path, name="t2", out_dir=second,
        replay=task / layout.DIR_ARM_A, dataset=dataset,
    )
    assert sorted(_entries(second, "")) == ["d1", "d2"]

    # status 从条目 + 桶重算展示，并 best-effort 重建索引
    out, _err, code = _run_cli(["task", "status", str(task)])
    assert code == 0
    assert "2 个桶" in out
    rebuilt = layout.read_snapshot_index(str(task / layout.DIR_SNAPSHOTS))
    assert rebuilt is not None and len(rebuilt["entries"]) == 2

    # 条目缺一个时才会回退读索引——把索引再弄坏，这时如实报错
    _entry_path(task, layout.DIR_ARM_A, "d1").unlink()
    index_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="快照池索引损坏"):
        _replay(
            tmp_path, name="third", out_dir=tmp_path / "third" / "A",
            replay=task / layout.DIR_ARM_A, dataset=dataset,
        )


# ---------------------------------------------------------------------------
# 真并发：两个分片进程同时回放到同一任务目录的同一臂
# ---------------------------------------------------------------------------

#: 子进程用的文件型适配器（放进 MEMBENCH_ADAPTERS_DIR，经持久化适配器机制加载）。
_PLUGIN = '''\
"""测试用文件型记忆适配器：状态 = data_dir/memories.jsonl，支持快照 / 恢复。"""
import json
import shutil
from pathlib import Path

from membench.adapters import ADAPTERS
from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult


class ReplayFileAdapter(BaseMemoryAdapter):
    name = "replay-file"

    def __init__(self, data_dir: str = "") -> None:
        self.data_dir = Path(data_dir)

    def reset(self) -> None:
        shutil.rmtree(self.data_dir, ignore_errors=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def ingest(self, messages):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with (self.data_dir / "memories.jsonl").open("a", encoding="utf-8") as handle:
            for message in messages:
                handle.write(json.dumps(message, ensure_ascii=False) + "\\n")
        return IngestStats(input_tokens=1, llm_calls=1)

    def query(self, question):
        path = self.data_dir / "memories.jsonl"
        lines = (
            [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if path.is_file()
            else []
        )
        return QueryResult(answer="周五", retrieved=lines)

    def snapshot_capabilities(self):
        return {"supported": True, "restore": True, "rebuild": False,
                "format_version": 1, "compatible_from": [1]}

    def snapshot(self, target_dir, tag):
        dest = Path(target_dir) / "memory"
        shutil.rmtree(dest, ignore_errors=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.data_dir.is_dir():
            shutil.copytree(self.data_dir, dest)
        else:
            dest.mkdir(parents=True, exist_ok=True)
        return {"layers": ["raw"]}

    def restore(self, source_dir, tag):
        source = Path(source_dir) / "memory"
        shutil.rmtree(self.data_dir, ignore_errors=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, self.data_dir, dirs_exist_ok=True)


def register(registry):
    registry["replay-file"] = ReplayFileAdapter
'''


def _install_plugin() -> None:
    """把子进程适配器写进 MEMBENCH_ADAPTERS_DIR（fixture 已指向 tmp_path）。"""
    plugin_dir = Path(os.environ["MEMBENCH_ADAPTERS_DIR"])
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "replay_file_plugin.py").write_text(_PLUGIN, encoding="utf-8")


def _spawn_shard(
    tmp_path: Path,
    *,
    dataset: Path,
    task: Path,
    source: Path,
    shard_index: int,
    shards: int = 2,
) -> subprocess.Popen:
    """起一个分片回放子进程（真进程，不是线程 / 同进程调用）。"""
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "membench.cli",
            "run",
            "--adapter",
            "replay-file",
            "--adapter-arg",
            f"data_dir={tmp_path / f'mem-s{shard_index}'}",
            "--dataset",
            str(dataset),
            "--judge",
            "heuristic",
            "--task-dir",
            str(task),
            "--shards",
            str(shards),
            "--shard-index",
            str(shard_index),
            "--shard-by",
            "dialogue",
            "--replay",
            str(source),
        ],
        cwd=str(_PROJECT_ROOT),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_shards(processes: list[subprocess.Popen]) -> None:
    """等全部子进程结束；任一个非零退出就把它的 stderr 带进断言。"""
    failures: list[str] = []
    for index, process in enumerate(processes):
        out, err = process.communicate(timeout=180)
        if process.returncode != 0:
            failures.append(f"分片进程 {index} 退出码 {process.returncode}\nstdout:\n{out}\nstderr:\n{err}")
    assert not failures, "\n\n".join(failures)


def test_parallel_sharded_replay_writes_one_entry_per_dialogue(tmp_path: Path) -> None:
    """两个分片进程并行回放：桶数 / 条目数正确，fuse 物化索引后两片一致。"""
    dataset = _dataset(tmp_path / "ds.jsonl")
    source = _full_run(tmp_path, name="src", dataset=dataset)
    task = _init_task(tmp_path / "out", "par", arms=2)
    _install_plugin()

    arm_a = task / layout.DIR_ARM_A
    processes = [
        _spawn_shard(tmp_path, dataset=dataset, task=arm_a, source=source, shard_index=index)
        for index in (0, 1)
    ]
    _wait_shards(processes)

    # 分片账本必须两片齐全（并行写账本不能丢条目）
    ledger = layout.read_shard_ledger(str(arm_a / layout.DIR_RUNS / DATASET))
    assert ledger is not None
    assert sorted(ledger["entries"]) == ["s0", "s1"]
    assert all(entry["status"] == "done" for entry in ledger["entries"].values())

    # 条目：每个对话一个文件，各片只写自己负责的对话
    entries = _entries(task, layout.DIR_ARM_A)
    assert sorted(entries) == ["d1", "d2"]
    assert all(entry["reused"] is False for entry in entries.values())

    # fuse 是分片流程的收尾：物化派生索引，两片信息合到同一份视图
    _out, err, code = _run_cli(["fuse", "--task-dir", str(arm_a)])
    assert code == 0, err
    index = json.loads(
        (task / layout.DIR_SNAPSHOTS / layout.FILE_SNAPSHOT_INDEX).read_text(encoding="utf-8")
    )
    assert len(index["buckets"]) == 2
    assert index["arms"][layout.DIR_ARM_A]["n_dialogues"] == 2
    assert index["entries"] == {
        layout.snapshot_entry_key(layout.DIR_ARM_A, DATASET, dialogue_id): entries[dialogue_id][
            "digest"
        ]
        for dialogue_id in ("d1", "d2")
    }

    out, _err, code = _run_cli(["task", "status", str(task)])
    assert code == 0
    assert "2 个桶" in out
    assert "臂 A: 2 对话（复用 0 / 新写 2）" in out


def _dataset_same_state(path: Path) -> Path:
    """两个对话的 ingest 内容完全相同：回放导出后状态哈希一致（抢同一个桶）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in (1, 2):
        rows.append(
            {
                "dialogue_id": f"d{index}",
                "sessions": [
                    [
                        {
                            "role": "user",
                            "content": "项目截止日是周五。",
                            "timestamp": None,
                        }
                    ]
                ],
                "questions": [
                    {
                        "qid": f"d{index}_q1",
                        "question": "项目截止日是哪天？",
                        "gold_answer": "周五",
                        "category": "single_hop",
                        "evidence": ["项目截止日是周五"],
                    }
                ],
            }
        )
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_parallel_shards_racing_on_same_bucket_reuse_it(tmp_path: Path) -> None:
    """两片恰好导出相同内容：只产生一个桶，另一片经 os.replace 失败路径记复用。"""
    dataset = _dataset_same_state(tmp_path / "ds.jsonl")
    source = _full_run(tmp_path, name="src", dataset=dataset)
    task = _init_task(tmp_path / "out", "race", arms=2)
    _install_plugin()

    arm_a = task / layout.DIR_ARM_A
    processes = [
        _spawn_shard(tmp_path, dataset=dataset, task=arm_a, source=source, shard_index=index)
        for index in (0, 1)
    ]
    _wait_shards(processes)

    assert len(_bucket_dirs(task)) == 1
    entries = _entries(task, layout.DIR_ARM_A)
    assert sorted(entries) == ["d1", "d2"]
    digests = {entry["digest"] for entry in entries.values()}
    assert len(digests) == 1
    # 恰好一片是真创建（reused=false），另一片在竞态中读到已存在的桶（reused=true）
    assert sorted(entry["reused"] for entry in entries.values()) == [False, True]

    index = layout.materialize_snapshot_index(str(task / layout.DIR_SNAPSHOTS))
    assert len(index["buckets"]) == 1
    arm = index["arms"][layout.DIR_ARM_A]
    assert (arm["n_dialogues"], arm["n_reused"], arm["n_new"]) == (2, 1, 1)
