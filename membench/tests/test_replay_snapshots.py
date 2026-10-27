"""回放任务的任务级快照池：内容去重、索引、两臂共用与链式回放。

回放（``--replay``）在恢复每个对话之后、第一道 query 之前把状态导出到**任务级**
内容寻址池 ``<任务>/snapshots/``：桶名 = 状态内容 sha256 前 12 位（内容哈希沿用
``snapshot.dir_sha256``，排除 manifest 自身），``snapshots/index.json`` 记
``臂|数据集|对话 -> 桶`` 与各臂来源。A/B 回放同一源时两臂指向同一批桶（只存一份），
内容不同则各存一份——差异本身就是"回放不保真"的证据。

``task init --replay`` 建出 ``snapshots/``；``--no-snapshot`` 回放既不写桶也不写索引。
"""

from __future__ import annotations

import io
import json
import shutil
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from membench import cli, layout
from membench import snapshot as snapshot_mod
from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
from membench.runner import Runner

#: 数据集文件名 stem（= 数据集目录名与索引里的 dataset 段）。
DATASET = "ds"


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
    return json.loads(
        (task / layout.DIR_SNAPSHOTS / layout.FILE_SNAPSHOT_INDEX).read_text(encoding="utf-8")
    )


def _bucket_dirs(task: Path) -> list[Path]:
    pool = task / layout.DIR_SNAPSHOTS
    return sorted(
        path
        for path in pool.iterdir()
        if path.is_dir() and not path.name.startswith(layout.SNAPSHOT_TMP_PREFIX)
    )


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
    """单臂回放：逐对话落桶、索引三段齐备、metrics 记 snapshots_dir 与计数。"""
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


def test_missing_bucket_and_corrupt_index_are_reported(tmp_path: Path) -> None:
    """池源不全时拒绝：索引指向的桶缺失 / 索引损坏都给出可操作的报错。"""
    dataset = _dataset(tmp_path / "ds.jsonl")
    source = _full_run(tmp_path, name="src", dataset=dataset)
    task = _init_task(tmp_path / "out", "first", arms=2)
    _replay(tmp_path, name="t1-a", out_dir=task / layout.DIR_ARM_A, replay=source, dataset=dataset)

    dropped = _bucket_dirs(task)[0]
    shutil.rmtree(dropped)
    with pytest.raises(SystemExit, match="索引指向的桶不存在"):
        _replay(tmp_path, name="t2-a", out_dir=tmp_path / "second" / "A",
                replay=task / layout.DIR_ARM_A, dataset=dataset)

    # 索引损坏：读到坏 JSON 直接报错，不猜
    index_path = task / layout.DIR_SNAPSHOTS / layout.FILE_SNAPSHOT_INDEX
    index_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="快照池索引损坏"):
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
