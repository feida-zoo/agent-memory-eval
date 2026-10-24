"""任务目录（``layout.init_task_dir`` / ``task_subdir`` / ``arm_dir``）与共享目录下的分片写入。

契约（见 :mod:`membench.layout` 与 :class:`membench.runner.Runner`）：

- 一次评测 = 一个任务目录：单臂 ``eval_<name>_<platform>_<ts>``，双臂
  ``compare_<name>_<platform>_<ts>``（``A/`` ``B/`` 各是一个完整评测产物根）；
- 多个分片可以写进**同一个**任务目录：共享目录下只允许 ``shard_by="dialogue"``
  （题目级分片会让多片写同一对话目录），各片进度与归属记在分片账本
  ``runs/<dataset>/shards.json``；
- 分片只写本片负责的对话目录与账本，数据集级 ``report/`` 与任务级 ``reports/``
  由 ``fuse`` 补齐；同一分片重复跑、账本口径不一致一律在写盘前拒绝。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from membench import layout
from membench.adapters import MockStrongAdapter
from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
from membench.runner import Runner

DATASET = "taskdir"


@pytest.fixture(autouse=True)
def _isolated_output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """默认产物根指向 tmp_path：任何走 output_root() 的路径都不落进仓库。"""
    root = tmp_path / "output-root"
    monkeypatch.setenv("MEMBENCH_OUTPUT_DIR", str(root))
    return root


def _dataset(path: Path, *, n_dialogues: int = 4, per: int = 2, marker: str = "") -> Path:
    """n 个对话 × 每对话 per 题的密集数据集（每个对话都有题）。"""
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
                                    "content": f"dd{d} 的记录：项目截止日是周五。{marker}",
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


class _BoomAdapter(BaseMemoryAdapter):
    """第 2 个对话的 query 抛错：验证分片中途失败时账本停在 running。"""

    name = "mock-boom"

    def reset(self) -> None:
        pass

    def ingest(self, messages: list[dict]) -> IngestStats:
        return IngestStats()

    def query(self, question: str) -> QueryResult:
        if os.environ.get("MEMBENCH_DIALOGUE_ID") == "dd2":
            raise RuntimeError("boom")
        return QueryResult(answer="周五", retrieved=[])


def _dialogues_of(dataset_dir: Path) -> list[str]:
    """数据集目录下的对话目录名（账本文件不计）。"""
    return sorted(p.name for p in Path(dataset_dir).iterdir() if p.is_dir())


def _ledger(dataset_dir: Path) -> dict:
    """读分片账本。"""
    return json.loads((Path(dataset_dir) / layout.FILE_SHARDS).read_text(encoding="utf-8"))


def _run_shard(experiment_dir: str, dataset: Path, *, index: int, **kwargs) -> Path:
    """在给定实验目录（任务目录或某个臂目录）里跑一个分片，返回数据集目录。"""
    return Path(
        Runner(
            MockStrongAdapter(),
            str(dataset),
            judge="heuristic",
            out_dir=experiment_dir,
            shard_index=index,
            **kwargs,
        ).run()
    )


def test_shared_task_dir_accepts_dialogue_shards_from_several_processes(tmp_path: Path) -> None:
    """两片写同一任务目录的同一臂：对话目录互补、账本累积两条 done、无数据集级产物。"""
    dataset = _dataset(tmp_path / f"{DATASET}.jsonl")
    task = Path(layout.init_task_dir(str(tmp_path), "a_vs_b", "t", arms=2, ts="20260101-120000"))
    arm_a = Path(layout.arm_dir(str(task), layout.DIR_ARM_A))

    for index in (0, 1):
        _run_shard(str(arm_a), dataset, index=index, shards=2, shard_by="dialogue", shared=True)

    dataset_dir = Path(layout.dataset_dir(str(arm_a), DATASET))
    ledger = _ledger(dataset_dir)
    assert ledger["schema"] == 1
    assert ledger["dataset"] == DATASET and ledger["shards"] == 2
    assert ledger["shard_by"] == "dialogue" and ledger["limit"] is None
    assert ledger["dataset_fingerprint"]["sha256"]
    assert sorted(ledger["entries"]) == ["s0", "s1"]
    assert {entry["status"] for entry in ledger["entries"].values()} == {"done"}
    assert ledger["entries"]["s0"]["dialogues"] == ["dd0", "dd2"]
    assert ledger["entries"]["s1"]["dialogues"] == ["dd1", "dd3"]
    assert sum(entry["n_questions"] for entry in ledger["entries"].values()) == 8
    for entry in ledger["entries"].values():
        assert entry["run_id"] and entry["started_at"] and entry["finished_at"]

    # 分片不写数据集级 report/，也不写任何一层的任务级 reports/
    assert not (dataset_dir / layout.DIR_REPORT).exists()
    assert not (arm_a / layout.DIR_REPORTS).exists()
    assert list((task / layout.DIR_REPORTS).iterdir()) == []
    # 另一条臂没被本次评测碰过
    assert list((task / layout.DIR_ARM_B).iterdir()) == []

    # 共享树完整（两片写完之后四个对话目录都在），且每个对话目录只由一片写过：
    # 对话级 metrics 记下写它的那个分片，这正是"两片不互相覆盖"的证据
    assert _dialogues_of(dataset_dir) == ["dd0", "dd1", "dd2", "dd3"]
    for dialogue_id, shard_index in (("dd0", 0), ("dd1", 1), ("dd2", 0), ("dd3", 1)):
        dlg_metrics = json.loads(
            (dataset_dir / dialogue_id / layout.DIR_REPORT / layout.FILE_METRICS).read_text(
                encoding="utf-8"
            )
        )
        assert dlg_metrics["shard_index"] == shard_index
        assert dlg_metrics["shard_by"] == "dialogue"
        assert f"_dshard{shard_index}of2_" in dlg_metrics["run_id"]
        assert dlg_metrics["dataset_dir"] == str(dataset_dir)


def test_shared_task_dir_rejects_repeat_shard(tmp_path: Path) -> None:
    """同一分片重复跑：按账本拒绝，且拒绝发生在写盘之前（现场不被改动）。"""
    dataset = _dataset(tmp_path / f"{DATASET}.jsonl")
    task = layout.init_task_dir(str(tmp_path), "one", "t", arms=1, ts="20260101-120000")
    dataset_dir = _run_shard(
        task, dataset, index=0, shards=2, shard_by="dialogue", shared=True
    )
    ledger_before = _ledger(dataset_dir)

    with pytest.raises(SystemExit, match="已经在本实验目录下跑过"):
        _run_shard(task, dataset, index=0, shards=2, shard_by="dialogue", shared=True)
    assert _ledger(dataset_dir) == ledger_before
    assert _dialogues_of(dataset_dir) == ["dd0", "dd2"]


def test_shared_task_dir_rejects_question_shard(tmp_path: Path) -> None:
    """共享任务目录只收 dialogue 分片：题目级分片会让多片写同一对话目录。"""
    dataset = _dataset(tmp_path / f"{DATASET}.jsonl")
    task = layout.init_task_dir(str(tmp_path), "one", "t", arms=1, ts="20260101-120000")
    with pytest.raises(SystemExit, match="只支持 --shard-by dialogue"):
        _run_shard(task, dataset, index=0, shards=2, shard_by="question", shared=True)
    # 拒绝在建任何产物之前（任务目录里只有 init_task_dir 建出的空骨架）
    assert list(Path(layout.runs_root(task)).iterdir()) == []


def test_shared_task_dir_rejects_ledger_mismatch(tmp_path: Path) -> None:
    """账本与本次不一致（shards / shard_by / limit / 数据集指纹）逐项拒绝：混片不可归因。"""
    dataset = _dataset(tmp_path / f"{DATASET}.jsonl")
    task = layout.init_task_dir(str(tmp_path), "one", "t", arms=1, ts="20260101-120000")
    dataset_dir = _run_shard(
        task, dataset, index=0, shards=2, shard_by="dialogue", shared=True
    )

    with pytest.raises(SystemExit, match=r"不一致（shards）"):
        _run_shard(task, dataset, index=1, shards=3, shard_by="dialogue", shared=True)
    # 非 shared 下 shard_by=question 本身合法，但与账本记录的分片维度冲突
    with pytest.raises(SystemExit, match=r"不一致（shard_by）"):
        _run_shard(task, dataset, index=1, shards=2, shard_by="question")
    with pytest.raises(SystemExit, match=r"不一致（limit）"):
        _run_shard(task, dataset, index=1, shards=2, shard_by="dialogue", shared=True, limit=2)
    # 同名数据集、内容变过 -> 指纹不一致
    drifted = _dataset(tmp_path / "elsewhere" / f"{DATASET}.jsonl", marker="（另一份数据集）")
    with pytest.raises(SystemExit, match="数据集指纹不一致"):
        _run_shard(task, drifted, index=1, shards=2, shard_by="dialogue", shared=True)

    # 四种拒绝都发生在写盘之前：账本只有第一片，也没有多出对话目录
    assert sorted(_ledger(dataset_dir)["entries"]) == ["s0"]
    assert _dialogues_of(dataset_dir) == ["dd0", "dd2"]


def test_shard_ledger_marks_running_before_finish(tmp_path: Path) -> None:
    """分片开跑即落 status=running：中途失败也能看出哪片在跑、覆盖哪些对话。"""
    dataset = _dataset(tmp_path / f"{DATASET}.jsonl")
    task = layout.init_task_dir(str(tmp_path), "one", "t", arms=1, ts="20260101-120000")
    with pytest.raises(RuntimeError, match="boom"):
        Runner(
            _BoomAdapter(),
            str(dataset),
            judge="heuristic",
            out_dir=task,
            shards=2,
            shard_index=0,
            shard_by="dialogue",
            shared=True,
        ).run()

    entry = _ledger(Path(layout.dataset_dir(task, DATASET)))["entries"]["s0"]
    assert entry["status"] == "running"
    assert entry["dialogues"] == ["dd0", "dd2"]
    assert entry["run_id"] and entry["started_at"]
    assert "finished_at" not in entry


def test_non_shared_shard_also_writes_ledger_but_no_dataset_product(tmp_path: Path) -> None:
    """非共享的独立分片目录同样只落对话目录 + 账本（分片产物不是"完整实验"）。"""
    dataset = _dataset(tmp_path / f"{DATASET}.jsonl")
    out_dir = tmp_path / "eval_shard_s0"
    dataset_dir = _run_shard(
        str(out_dir), dataset, index=0, shards=2, shard_by="dialogue"
    )
    assert _dialogues_of(dataset_dir) == ["dd0", "dd2"]
    assert _ledger(dataset_dir)["entries"]["s0"]["status"] == "done"
    assert not (dataset_dir / layout.DIR_REPORT).exists()
    assert not (out_dir / layout.DIR_REPORTS).exists()
