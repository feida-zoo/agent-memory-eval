"""fuse 测试：分片账本 + 对话目录 → 完整数据集树。

分片产物只落本片负责的对话目录与 ``runs/<dataset>/shards.json`` 账本（见
:mod:`membench.layout`）；``fuse`` 按账本与对话目录把它们拼成一份完整实验：
``<out>/runs/<dataset>/{report/,<对话>/report/}`` + ``<out>/reports/``。

两种入参形态都在这里覆盖：

- 跨目录 ``fuse_runs([dir1, dir2, ...], out)``——各片各写自己的实验目录；
- 就地 ``fuse_runs([], task_dir)``（等价写法 ``fuse_runs([task_dir], task_dir)``）——
  各片本来就写在同一棵树上，只补齐数据集级 / 任务级产物。

校验失败必须发生在写盘之前（不留半成品），故各拒绝用例末尾都断言目标目录不存在。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from membench import layout
from membench.adapters import MockStrongAdapter, MockWeakAdapter
from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
from membench.data import SAMPLE_DATASET_PATH
from membench.fuse import fuse_runs
from membench.runner import Runner

_DATASET = "sample_dataset"


@pytest.fixture(autouse=True)
def _isolated_output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """兜底：任何走默认产物目录的调用都落 tmp，不往仓库写产物。"""
    monkeypatch.setenv("MEMBENCH_OUTPUT_DIR", str(tmp_path / "output"))


class _StaleAdapter(BaseMemoryAdapter):
    """永远答旧值的适配器：knowledge_update 题必 stale（新值未命中）。"""

    name = "mock-stale"

    def reset(self) -> None:
        pass

    def ingest(self, messages: list[dict]) -> IngestStats:
        return IngestStats()

    def query(self, question: str) -> QueryResult:
        return QueryResult(answer="截止日是周三", retrieved=[])


class _SnapshotAdapter(BaseMemoryAdapter):
    """最小快照型适配器：只在快照目录里落一份能读回的文件，用于验证对话目录随对话并入。"""

    name = "mock-snapshot"

    def reset(self) -> None:
        pass

    def ingest(self, messages: list[dict]) -> IngestStats:
        return IngestStats(input_tokens=5, llm_calls=1)

    def query(self, question: str) -> QueryResult:
        return QueryResult(answer="周五", retrieved=["项目截止日是周五"])

    def snapshot_capabilities(self) -> dict:
        return {
            "supported": True,
            "restore": True,
            "format_version": 1,
            "compatible_from": [1],
        }

    def snapshot(self, target_dir: str, tag: str) -> dict:
        Path(target_dir, "memory.json").write_text(
            json.dumps({"tag": tag}), encoding="utf-8"
        )
        return {"items": 1}

    def restore(self, source_dir: str, tag: str) -> None:
        """无状态可恢复的最小实现：声明了 restore 能力就得真的能调起来。"""
        assert Path(source_dir).is_dir()


def _stale_dataset(path: Path) -> Path:
    """单 dialogue 两道 knowledge_update 题（gold=周五, gold_stale=周三）。"""
    dialogue = {
        "dialogue_id": "stale_d1",
        "sessions": [[{"role": "user", "content": "截止日改到周五了。", "timestamp": None}]],
        "questions": [
            {
                "qid": f"stale_q{i}",
                "question": "现在项目截止日是哪天？",
                "gold_answer": "周五",
                "category": "knowledge_update",
                "evidence": [],
                "gold_stale": "周三",
            }
            for i in (1, 2)
        ],
    }
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(dialogue, ensure_ascii=False) + "\n")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _report_path(dataset_dir: str, name: str) -> Path:
    return Path(dataset_dir) / layout.DIR_REPORT / name


def _read_metrics(dataset_dir: str) -> dict:
    """读数据集级 metrics（融合产物才有；分片产物没有数据集级报告）。"""
    return json.loads(
        _report_path(dataset_dir, layout.FILE_METRICS).read_text(encoding="utf-8")
    )


def _dialogue_dirs(dataset_dir: str) -> list[Path]:
    """数据集目录下的对话目录（``report/`` 与账本文件都不算）。"""
    return sorted(
        p
        for p in Path(dataset_dir).iterdir()
        if p.is_dir() and p.name != layout.DIR_REPORT
    )


def _dialogue_metrics(dataset_dir: str, name: str) -> dict:
    return json.loads(
        (Path(dataset_dir) / name / layout.DIR_REPORT / layout.FILE_METRICS).read_text(
            encoding="utf-8"
        )
    )


def _ledger(dataset_dir: str) -> dict:
    return json.loads(
        (Path(dataset_dir) / layout.FILE_SHARDS).read_text(encoding="utf-8")
    )


def _write_ledger(dataset_dir: str, ledger: dict[str, Any]) -> None:
    (Path(dataset_dir) / layout.FILE_SHARDS).write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _drop_last_record(dataset_dir: str, name: str) -> None:
    """从某个对话的逐题明细里删掉最后一条（模拟产物被削掉一题）。"""
    path = Path(dataset_dir) / name / layout.DIR_REPORT / layout.FILE_RESULTS
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")


def _set_all_dialogue_metrics(dataset_dir: str, key: str, value: Any) -> None:
    """把某一片下所有对话级 metrics 的某个字段改掉（模拟各片口径/元数据混用）。"""
    for dialogue_dir in _dialogue_dirs(dataset_dir):
        path = dialogue_dir / layout.DIR_REPORT / layout.FILE_METRICS
        metrics = json.loads(path.read_text(encoding="utf-8"))
        metrics[key] = value
        path.write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")


def _sum_of_shard_usage(*dataset_dirs: str) -> dict[str, int]:
    """各片各对话的 ``token_usage`` 独立求和（融合口径的独立复算）。"""
    total: dict[str, int] = {}
    for dataset_dir in dataset_dirs:
        for dialogue_dir in _dialogue_dirs(dataset_dir):
            usage = _dialogue_metrics(dataset_dir, dialogue_dir.name)["token_usage"]
            for field, value in usage.items():
                total[field] = total.get(field, 0) + int(value or 0)
    return total


def _shard_run(
    tmp_path: Path,
    name: str,
    adapter: Any,
    shards: int,
    shard_index: int,
    shard_by: str = "question",
    out_dir: str | None = None,
    dataset: Path = Path(SAMPLE_DATASET_PATH),
    **kwargs: Any,
) -> str:
    """跑一个分片 run，返回其**数据集目录**（``<实验目录>/runs/<dataset>``）。"""
    runner = Runner(
        adapter,
        str(dataset),
        judge="heuristic",
        out_dir=out_dir or str(tmp_path / name),
        shards=shards,
        shard_index=shard_index,
        shard_by=shard_by,
        **kwargs,
    )
    return runner.run()


def _full_run(tmp_path: Path, name: str, adapter: Any) -> tuple[str, dict, list[dict]]:
    """跑全量 run，返回 ``(数据集目录, 数据集级 metrics, 全量逐题记录)``。"""
    runner = Runner(
        adapter,
        str(SAMPLE_DATASET_PATH),
        judge="heuristic",
        out_dir=str(tmp_path / name),
    )
    dataset_dir = runner.run()
    return (
        dataset_dir,
        _read_metrics(dataset_dir),
        _read_jsonl(_report_path(dataset_dir, layout.FILE_RESULTS)),
    )


def test_fuse_two_shards_matches_full_run(tmp_path: Path) -> None:
    """两个题目级分片融合后的题集与指标等同于全量 run，产物落完整实验树。"""
    s0 = _shard_run(tmp_path, "s0", MockStrongAdapter(), 2, 0)
    s1 = _shard_run(tmp_path, "s1", MockStrongAdapter(), 2, 1)
    _, full_metrics, full_records = _full_run(tmp_path, "full", MockStrongAdapter())

    out_dir = tmp_path / "fused"
    result = fuse_runs([s0, s1], str(out_dir))

    dataset_dir = Path(result["dataset_dir"])
    assert result["out_dir"] == str(out_dir)
    assert result["mode"] == "merge"
    assert result["experiment_dir"] == str(out_dir)
    assert dataset_dir == out_dir / layout.DIR_RUNS / _DATASET
    for name in (
        layout.FILE_RESULTS,
        layout.FILE_METRICS,
        layout.FILE_REPORT_MD,
        layout.FIG_CATEGORY,
        layout.FIG_LATENCY,
    ):
        assert (dataset_dir / layout.DIR_REPORT / name).is_file(), f"缺少 report/{name}"
    # 任务级 reports/：与 Runner 的 write_task_reports 同形态（md + metrics）
    task_dir = out_dir / layout.DIR_REPORTS
    assert (task_dir / layout.FILE_REPORT_MD).is_file()
    assert (task_dir / layout.FILE_METRICS).is_file()

    fused_records = _read_jsonl(dataset_dir / layout.DIR_REPORT / layout.FILE_RESULTS)
    assert [r["qid"] for r in fused_records] == [r["qid"] for r in full_records]
    assert result["n_questions"] == len(full_records)
    metrics = result["metrics"]

    # 逐对话报告：融合时按对话粒度重建（每条只含自己的题，且不出图）
    dialogue_dirs = _dialogue_dirs(str(dataset_dir))
    assert dialogue_dirs
    fused_qids: list[str] = []
    for dialogue_dir in dialogue_dirs:
        dialogue_report = dialogue_dir / layout.DIR_REPORT
        assert (dialogue_report / layout.FILE_METRICS).is_file()
        assert (dialogue_report / layout.FILE_REPORT_MD).is_file()
        assert not list(dialogue_report.glob("fig_*.png"))
        dialogue_metrics = json.loads(
            (dialogue_report / layout.FILE_METRICS).read_text(encoding="utf-8")
        )
        # 对话目录里的 runtime/ 始终在（适配器要把运行时数据落这里）
        assert (dialogue_dir / layout.DIR_RUNTIME).is_dir()
        # 跨目录融合重建的对话级 metrics 属融合身份（不再是某一分片的分片身份）
        assert dialogue_metrics["shard_index"] is None
        assert dialogue_metrics["shards"] == 2
        assert dialogue_metrics["run_id"] == metrics["run_id"]
        fused_qids.extend(r["qid"] for r in _read_jsonl(dialogue_report / layout.FILE_RESULTS))
    assert sorted(fused_qids) == sorted(r["qid"] for r in full_records)

    # 与普通 run 同构：字段集合逐项对齐——融合产物只多 fused / fused_from / stale_gate
    # （门禁结论），少 state_snapshot（各片快照按对话归属片目录，融一次不会凭空产生
    # "整场"快照）
    assert set(metrics) - {"fused", "fused_from", "stale_gate"} == set(full_metrics) - {
        "state_snapshot"
    }
    assert "state_snapshot" not in metrics
    # 非计时指标逐项对齐（latency / ingest 耗时不参与，机器抖动不属于口径）
    timing = ("latency_p50_ms", "latency_p95_ms", "avg_ingest_latency_ms")
    assert {k: v for k, v in metrics["overall"].items() if k not in timing} == {
        k: v for k, v in full_metrics["overall"].items() if k not in timing
    }
    for key in (
        "by_category",
        "has_evidence",
        "has_stale",
        "has_abstention",
        "n_ignored",
    ):
        assert metrics[key] == full_metrics[key], key
    # token 记账：逐片逐对话求和（题目级分片下同一对话被两片各 ingest 一次，不去重——
    # 各片自己的计数才是真实成本），题目与 query 侧不重复
    expected_usage = _sum_of_shard_usage(s0, s1)
    assert metrics["token_usage"] == expected_usage
    assert metrics["token_usage"]["n_questions"] == full_metrics["token_usage"]["n_questions"]
    assert expected_usage["query_input_tokens"] == full_metrics["token_usage"]["query_input_tokens"]
    assert expected_usage["query_output_tokens"] == full_metrics["token_usage"]["query_output_tokens"]
    assert expected_usage["ingest_input_tokens"] > full_metrics["token_usage"]["ingest_input_tokens"]
    assert metrics["fused"] is True
    assert len(metrics["fused_from"]) == 2
    assert metrics["shards"] == 2 and metrics["shard_by"] == "question"
    assert metrics["run_mode"] == "full" and metrics["limit"] is None
    # 落盘的数据集级 metrics 与返回值一致
    assert _read_metrics(result["dataset_dir"]) == metrics
    assert result["gate"]["status"] == "PASS"
    assert metrics["stale_gate"]["status"] == "PASS"


def test_fuse_in_place_after_dialogue_shards_in_one_task_dir(tmp_path: Path) -> None:
    """就地模式：两片对话级分片写进同一任务目录 → fuse 补齐数据集级与任务级产物。

    就地融合不新建目录：融合的输入树与输出树是同一棵（对话目录本来就各归各片），
    只补数据集级 ``report/`` 与任务级 ``reports/``；``fuse_runs([task], task)``
    与 ``fuse_runs([], task)`` 等价。
    """
    task = Path(layout.init_task_dir(str(tmp_path), "inplace", "t", arms=1, ts="20260101-120000"))
    for index in (0, 1):
        _shard_run(
            tmp_path,
            "",
            MockStrongAdapter(),
            2,
            index,
            shard_by="dialogue",
            out_dir=str(task),
            shared=True,
        )
    dataset_dir = Path(layout.dataset_dir(str(task), _DATASET))
    _, _, full_records = _full_run(tmp_path, "full", MockStrongAdapter())
    assert not (dataset_dir / layout.DIR_REPORT).exists()
    assert _dialogue_dirs(str(dataset_dir))

    result = fuse_runs([], str(task))
    assert result["mode"] == "in_place"
    assert result["experiment_dir"] == str(task)
    assert result["dataset_dir"] == str(dataset_dir)
    # 数据集级与任务级产物补齐，逐题顺序与不分片直跑一致
    fused_records = _read_jsonl(dataset_dir / layout.DIR_REPORT / layout.FILE_RESULTS)
    assert [r["qid"] for r in fused_records] == [r["qid"] for r in full_records]
    assert (task / layout.DIR_REPORTS / layout.FILE_REPORT_MD).is_file()
    assert (task / layout.DIR_REPORTS / layout.FILE_METRICS).is_file()
    metrics = _read_metrics(str(dataset_dir))
    assert result["n_questions"] == len(full_records)
    assert metrics["shard_by"] == "dialogue" and metrics["shards"] == 2
    assert "_fused" in metrics["run_id"] and "_dshard2" in metrics["run_id"]
    # 就地融合不重写对话级产物：账本仍是"哪个对话归哪一片"的记录，各对话的
    # metrics 保持自己那一片的分片身份（这也让 fuse 可以重复执行）
    assert (dataset_dir / layout.FILE_SHARDS).is_file()
    ledger = _ledger(str(dataset_dir))
    owned = {
        name: index
        for key, entry in ledger["entries"].items()
        for index in [layout.shard_key_index(key)]
        for name in entry["dialogues"]
    }
    assert owned
    for name, index in owned.items():
        dialogue_metrics = _dialogue_metrics(str(dataset_dir), name)
        assert dialogue_metrics["shard_index"] == index
        assert f"_dshard{index}of2_" in dialogue_metrics["run_id"]

    # 等价写法：把同一个数据集目录写给 --runs，仍识别为就地融合（可重复执行）
    again = fuse_runs([str(task)], str(task))
    assert again["mode"] == "in_place"
    assert [r["qid"] for r in _read_jsonl(
        dataset_dir / layout.DIR_REPORT / layout.FILE_RESULTS
    )] == [r["qid"] for r in full_records]
    assert again["n_questions"] == len(full_records)


def test_fuse_merges_dialogue_dirs_with_runtime_and_snapshot(tmp_path: Path) -> None:
    """跨目录融合：对话目录随对话并入融合树，快照与运行时数据一并带过来。"""
    s0 = _shard_run(tmp_path, "s0", _SnapshotAdapter(), 2, 0)
    s1 = _shard_run(tmp_path, "s1", _SnapshotAdapter(), 2, 1)
    # 题目级分片下两片都可能碰到同一个对话：并入的是**序号最小**那一片的那一份
    expected_src = {
        d.name: _dialogue_metrics(s0, d.name)["run_id"] for d in _dialogue_dirs(s0)
    }
    for shard_dir in (s1,):
        for dialogue_dir in _dialogue_dirs(shard_dir):
            expected_src.setdefault(
                dialogue_dir.name, _dialogue_metrics(shard_dir, dialogue_dir.name)["run_id"]
            )

    result = fuse_runs([s0, s1], str(tmp_path / "fused"))
    dataset_dir = Path(result["dataset_dir"])
    assert {d.name for d in _dialogue_dirs(str(dataset_dir))} == set(expected_src)
    for dialogue_dir in _dialogue_dirs(str(dataset_dir)):
        assert (dialogue_dir / layout.DIR_RUNTIME).is_dir()
        manifest_path = dialogue_dir / layout.DIR_SNAPSHOT / "manifest.json"
        assert manifest_path.is_file(), "快照应随对话目录并入"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["dialogue_id"] == dialogue_dir.name
        assert manifest["complete"] is True
        # 并入的是那一片自己的快照（清单里的 run_id 就是它的身份）
        assert manifest["run_id"] == expected_src[dialogue_dir.name]

    # 融合树是自包含的：可以直接拿它做 --replay（清单齐备、身份与指纹自洽）
    replayed = Runner(
        _SnapshotAdapter(),
        str(SAMPLE_DATASET_PATH),
        judge="heuristic",
        out_dir=str(tmp_path / "replay"),
        replay=str(dataset_dir),
        replay_mode="restore",
    ).run()
    replayed_metrics = _read_metrics(replayed)
    assert replayed_metrics["run_mode"] == "replay-restore"
    assert replayed_metrics["overall"]["n_questions"] == result["n_questions"]


def test_fuse_dialogue_shards_matches_full_run(tmp_path: Path) -> None:
    """dialogue 分片的两个分片可融合为与全量一致的完整 run。"""
    d0 = _shard_run(tmp_path, "d0", MockStrongAdapter(), 2, 0, shard_by="dialogue")
    d1 = _shard_run(tmp_path, "d1", MockStrongAdapter(), 2, 1, shard_by="dialogue")
    _, full_metrics, full_records = _full_run(tmp_path, "full", MockStrongAdapter())

    result = fuse_runs([d0, d1], str(tmp_path / "fused"))
    assert result["n_questions"] == len(full_records)
    assert result["metrics"]["shard_by"] == "dialogue"
    assert result["metrics"]["shards"] == 2
    assert result["metrics"]["overall"]["accuracy"] == pytest.approx(
        full_metrics["overall"]["accuracy"]
    )
    # dialogue 分片间不重复 ingest，融合后与全量 run 的题集一致
    fused_records = _read_jsonl(
        Path(result["dataset_dir"]) / layout.DIR_REPORT / layout.FILE_RESULTS
    )
    assert [r["qid"] for r in fused_records] == [r["qid"] for r in full_records]
    # 每段对话只被一片 ingest 一次：融合的 token 记账与全量直跑逐字段相等
    assert result["metrics"]["token_usage"] == full_metrics["token_usage"]


def test_fuse_requires_full_shard_coverage(tmp_path: Path) -> None:
    """账本条目没覆盖 0..N-1（少片）时拒绝融合，并点名缺哪片。"""
    s0 = _shard_run(tmp_path, "s0", MockStrongAdapter(), 3, 0)
    s1 = _shard_run(tmp_path, "s1", MockStrongAdapter(), 3, 1)
    with pytest.raises(ValueError, match=r"未完整覆盖 0\.\.2：缺 s2"):
        fuse_runs([s0, s1], str(tmp_path / "fused"))
    assert not (tmp_path / "fused").exists()


def test_fuse_rejects_unfinished_shard(tmp_path: Path) -> None:
    """账本里还有 running 的片（跑到一半）时拒绝：半成品会被当成完整的一片。"""
    s0 = _shard_run(tmp_path, "s0", MockStrongAdapter(), 2, 0)
    s1 = _shard_run(tmp_path, "s1", MockStrongAdapter(), 2, 1)
    ledger = _ledger(s1)
    ledger["entries"]["s1"]["status"] = "running"
    _write_ledger(s1, ledger)

    with pytest.raises(ValueError, match="还没跑完"):
        fuse_runs([s0, s1], str(tmp_path / "fused"))
    assert not (tmp_path / "fused").exists()


def test_fuse_rejects_duplicate_shards(tmp_path: Path) -> None:
    """同一个分片目录给多次 / 同片出现在两个目录都会拒绝。"""
    s0 = _shard_run(tmp_path, "s0", MockStrongAdapter(), 2, 0)
    s1 = _shard_run(tmp_path, "s1", MockStrongAdapter(), 2, 1)
    with pytest.raises(ValueError, match="被给了多次"):
        fuse_runs([s0, s0, s1], str(tmp_path / "fused"))
    with pytest.raises(ValueError, match="被给了多次"):
        fuse_runs([s0, s0], str(tmp_path / "fused2"))

    # 另一份实验目录里再跑一次同一片：同片出现在两处 -> 也无法判断该用哪份
    clone = _shard_run(tmp_path, "clone", MockStrongAdapter(), 2, 0)
    with pytest.raises(ValueError, match="同时出现在两个目录"):
        fuse_runs([s0, clone, s1], str(tmp_path / "fused3"))
    assert not (tmp_path / "fused").exists()
    assert not (tmp_path / "fused2").exists()
    assert not (tmp_path / "fused3").exists()


def test_fuse_rejects_unsharded_run_and_fused_product(tmp_path: Path) -> None:
    """完整 run（未分片）与已是融合结果的产物都明确拒绝。"""
    dataset_dir, _, _ = _full_run(tmp_path, "full", MockStrongAdapter())
    with pytest.raises(ValueError, match="未分片"):
        fuse_runs([dataset_dir], str(tmp_path / "fused"))

    s0 = _shard_run(tmp_path, "s0", MockStrongAdapter(), 2, 0)
    s1 = _shard_run(tmp_path, "s1", MockStrongAdapter(), 2, 1)
    fused = fuse_runs([s0, s1], str(tmp_path / "fused2"))["dataset_dir"]
    with pytest.raises(ValueError, match="已经是融合结果"):
        fuse_runs([fused], str(tmp_path / "fused3"))
    assert not (tmp_path / "fused3").exists()


def test_fuse_rejects_adapter_mismatch(tmp_path: Path) -> None:
    s0 = _shard_run(tmp_path, "s0", MockStrongAdapter(), 2, 0)
    s1 = _shard_run(tmp_path, "s1", MockWeakAdapter(), 2, 1)
    with pytest.raises(ValueError, match="adapter 不一致"):
        fuse_runs([s0, s1], str(tmp_path / "fused"))


def test_fuse_rejects_ledger_scope_mismatch(tmp_path: Path) -> None:
    """账本口径不一致（shards / shard_by / limit / 分片数）逐项拒绝。"""
    s0 = _shard_run(tmp_path, "s0", MockStrongAdapter(), 2, 0)
    more = _shard_run(tmp_path, "more", MockStrongAdapter(), 4, 1)
    with pytest.raises(ValueError, match="shards 不一致"):
        fuse_runs([s0, more], str(tmp_path / "fused"))

    dialogue = _shard_run(tmp_path, "d1", MockStrongAdapter(), 2, 1, shard_by="dialogue")
    with pytest.raises(ValueError, match="shard_by 不一致"):
        fuse_runs([s0, dialogue], str(tmp_path / "fused2"))
    assert not (tmp_path / "fused").exists()
    assert not (tmp_path / "fused2").exists()


def test_fuse_rejects_mixed_shard_by(tmp_path: Path) -> None:
    """题目级分片与 dialogue 分片混合融合 → 明确报错（不允许混维）。"""
    q0 = _shard_run(tmp_path, "q0", MockStrongAdapter(), 2, 0, shard_by="question")
    d1 = _shard_run(tmp_path, "d1", MockStrongAdapter(), 2, 1, shard_by="dialogue")
    with pytest.raises(ValueError, match="shard_by 不一致"):
        fuse_runs([q0, d1], str(tmp_path / "fused"))


def test_fuse_rejects_incomplete_question_coverage(tmp_path: Path) -> None:
    """题目级分片少一道题的明细 → 覆盖完整性校验拒绝（不做"看起来完整"的融合）。"""
    s0 = _shard_run(tmp_path, "s0", MockStrongAdapter(), 2, 0)
    s1 = _shard_run(tmp_path, "s1", MockStrongAdapter(), 2, 1)
    name = _dialogue_dirs(s0)[0].name
    _drop_last_record(s0, name)

    with pytest.raises(ValueError, match="题集与该片口径不符（缺 "):
        fuse_runs([s0, s1], str(tmp_path / "fused"))
    assert not (tmp_path / "fused").exists()


def test_fuse_rejects_missing_or_extra_dialogue_dirs(tmp_path: Path) -> None:
    """账本记的对话与树里的对话目录必须对得上（少一个 / 多一个都拒绝）。"""
    s0 = _shard_run(tmp_path, "s0", MockStrongAdapter(), 2, 0)
    s1 = _shard_run(tmp_path, "s1", MockStrongAdapter(), 2, 1)

    # 少：删掉本片一个对话目录（账本仍记着它）
    victim = _dialogue_dirs(s0)[0]
    shutil.move(str(victim), str(tmp_path / "moved-away"))
    with pytest.raises(ValueError, match="树里却没有它的产物"):
        fuse_runs([s0, s1], str(tmp_path / "fused"))
    assert not (tmp_path / "fused").exists()
    shutil.move(str(tmp_path / "moved-away"), str(victim))

    # 多：把另一片的对话目录挪进本片（账本里没有它）
    extra = _dialogue_dirs(s1)[0]
    shutil.move(str(extra), str(Path(s0) / "intruder"))
    with pytest.raises(ValueError, match="不在分片账本里"):
        fuse_runs([s0, s1], str(tmp_path / "fused2"))
    assert not (tmp_path / "fused2").exists()


def test_resolve_dataset_dir_reports_unfused_shard_product(tmp_path: Path) -> None:
    """未融合的分片产物有专门文案（不是"没有任何数据集产物"，那是另一回事）。"""
    from membench.layout import resolve_dataset_dir

    s0 = _shard_run(tmp_path, "s0", MockStrongAdapter(), 2, 0)
    for target in (s0, str(Path(s0).parent.parent)):
        with pytest.raises(SystemExit, match="未融合的分片产物"):
            resolve_dataset_dir(target)

    # 真正没有产物的目录仍是"没有任何数据集产物"（两者不能混为一谈）
    empty = tmp_path / "empty_exp"
    (empty / layout.DIR_RUNS).mkdir(parents=True)
    with pytest.raises(SystemExit, match="没有任何数据集产物"):
        resolve_dataset_dir(str(empty))


def test_fuse_stale_gate_fail(tmp_path: Path) -> None:
    """融合结果 stale_rate 超阈值 → 门禁 FAIL（一票否决，复用 evaluate 阈值）。"""
    ds = _stale_dataset(tmp_path / "stale.jsonl")
    s0 = _shard_run(tmp_path, "s0", _StaleAdapter(), 2, 0, dataset=ds)
    s1 = _shard_run(tmp_path, "s1", _StaleAdapter(), 2, 1, dataset=ds)
    result = fuse_runs([s0, s1], str(tmp_path / "fused"))
    assert result["metrics"]["overall"]["stale_rate"] == pytest.approx(1.0)
    assert result["gate"]["status"] == "FAIL"
    assert result["metrics"]["stale_gate"]["status"] == "FAIL"
    # 门禁结论落盘：数据集级 metrics.json 里同样标 FAIL（读产物即可复核结论）
    assert _read_metrics(result["dataset_dir"])["stale_gate"]["status"] == "FAIL"


def test_fuse_missing_input_or_incomplete_product_rejected(tmp_path: Path) -> None:
    """入参不存在 -> SystemExit；产物有 report/ 层级但缺 metrics.json -> SystemExit。"""
    with pytest.raises(SystemExit, match="产物目录不存在"):
        fuse_runs([str(tmp_path / "nonexistent")], str(tmp_path / "fused"))

    # 有 report/ 层级但没落 metrics.json：产物不完整，宁可报错也别拿半成品融合
    incomplete = Path(layout.dataset_report_dir(str(tmp_path / "exp"), _DATASET))
    incomplete.mkdir(parents=True)
    with pytest.raises(SystemExit, match="metrics.json"):
        fuse_runs([str(incomplete.parent)], str(tmp_path / "fused2"))


def test_fuse_reports_missing_ledger_for_old_shard_products(tmp_path: Path) -> None:
    """分片产物没有账本（旧版写法）时明确拒绝，而不是猜覆盖范围。"""
    dataset_dir = Path(layout.dataset_dir(str(tmp_path / "legacy_shard"), _DATASET))
    (dataset_dir / layout.DIR_REPORT).mkdir(parents=True)
    (dataset_dir / layout.DIR_REPORT / layout.FILE_METRICS).write_text(
        json.dumps({"dataset": _DATASET, "shards": 2, "shard_index": 0}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="缺少账本"):
        fuse_runs([str(dataset_dir)], str(tmp_path / "fused"))
    assert not (tmp_path / "fused").exists()


def test_fuse_passes_through_comparability_metadata(tmp_path: Path) -> None:
    """融合产物透传可比性元数据：fused run 之间必须能过 strict 可比性（回归铁证）。"""

    def _shard(name: str, index: int) -> str:
        return _shard_run(
            tmp_path,
            name,
            MockStrongAdapter(),
            2,
            index,
            driver_llm="gpt-4o",
            adapter_version="v1",
            adapter_meta="k=v",
        )

    s0 = _shard("m0", 0)
    s1 = _shard("m1", 1)
    fused = fuse_runs([s0, s1], str(tmp_path / "fused_meta"))["dataset_dir"]

    from membench.compare import compare_runs

    src = _dialogue_metrics(s0, _dialogue_dirs(s0)[0].name)
    dst = _read_metrics(fused)
    for key in (
        "driver_llm",
        "adapter_version",
        "adapter_meta",
        "dataset_fingerprint",
        "retrieval_match",
        "schema_version",
    ):
        assert dst.get(key) == src.get(key), key
    assert dst["fused"] is True and dst["fused_from"]

    # 产物归属字段指向融合实验目录内的真实路径
    assert dst["experiment_dir"] == str(tmp_path / "fused_meta")
    assert dst["dataset_dir"] == fused

    # 回归：此前关键维度全缺失，require_comparable 必然 FAIL
    result = compare_runs(
        fused, fused, require_comparable=True, out_dir=str(tmp_path / "cmp")
    )
    assert result["comparability"]["unverifiable"] == []
    assert result["comparability"]["comparable"] is True
    assert result["verdict"] != "FAIL"


def test_fuse_rejects_mixed_metadata(tmp_path: Path) -> None:
    """判分/驱动等元数据不一致的分片拒绝融合（防止混片）。"""
    s0 = _shard_run(tmp_path, "x0", MockStrongAdapter(), 2, 0)
    s1 = _shard_run(tmp_path, "x1", MockStrongAdapter(), 2, 1)
    _set_all_dialogue_metrics(s1, "driver_llm", "different-model")

    with pytest.raises(ValueError, match="driver_llm 不一致"):
        fuse_runs([s0, s1], str(tmp_path / "bad"))
    assert not (tmp_path / "bad").exists()


def test_fuse_rejects_dialogue_metadata_off_ledger(tmp_path: Path) -> None:
    """对话级 metrics 与账本不一致（分片序号 / 身份 / 口径）时拒绝。"""
    s0 = _shard_run(tmp_path, "s0", MockStrongAdapter(), 2, 0)
    s1 = _shard_run(tmp_path, "s1", MockStrongAdapter(), 2, 1)
    _set_all_dialogue_metrics(s0, "shard_index", 1)
    with pytest.raises(ValueError, match="对话级元数据与分片账本不一致"):
        fuse_runs([s0, s1], str(tmp_path / "bad"))
    assert not (tmp_path / "bad").exists()


def _limit_shard(tmp_path: Path, name: str, index: int, limit: int | None) -> str:
    """跑一个带 ``limit`` 的分片 run，返回其数据集目录。"""
    return _shard_run(tmp_path, name, MockStrongAdapter(), 2, index, limit=limit)


def test_fuse_rejects_mixed_limit(tmp_path: Path) -> None:
    """全量分片与 --limit 冒烟分片混融 -> 拒绝（否则得到"部分全量"的假完整结果）。"""
    smoke = _limit_shard(tmp_path, "lim0", 0, 3)
    full = _limit_shard(tmp_path, "lim1", 1, None)

    with pytest.raises(ValueError, match="limit 不一致"):
        fuse_runs([smoke, full], str(tmp_path / "fused_bad"))
    assert not (tmp_path / "fused_bad").exists()


def test_fuse_passes_limit_through_and_marks_run_id(tmp_path: Path) -> None:
    """融合产物透传 limit，run_id 带 fused / 分片 / limit（与 commit）标记。

    回归：limit 曾被硬编码为 None、run_id 由 fuse 手写（漏 commit 与分片标记），
    冒烟融合产物与全量融合产物因此完全不可区分。
    """
    smoke0 = _limit_shard(tmp_path, "m0", 0, 3)
    smoke1 = _limit_shard(tmp_path, "m1", 1, 3)
    full0 = _limit_shard(tmp_path, "f0", 0, None)
    full1 = _limit_shard(tmp_path, "f1", 1, None)

    smoke = fuse_runs([smoke0, smoke1], str(tmp_path / "fused_smoke"))
    full = fuse_runs([full0, full1], str(tmp_path / "fused_full"))

    assert smoke["metrics"]["limit"] == 3
    assert full["metrics"]["limit"] is None
    # 冒烟融合产物的题集确实只是全量的一部分（不是"假全量"）
    assert 0 < smoke["n_questions"] < full["n_questions"]

    smoke_id = smoke["metrics"]["run_id"]
    assert "_fused" in smoke_id
    assert "_shard2" in smoke_id  # 分片数进身份（无具体分片序号：这是整体融合产物）
    assert "_limit3" in smoke_id
    full_id = full["metrics"]["run_id"]
    assert "_fused" in full_id and "_shard2" in full_id
    assert "limit" not in full_id
    assert smoke_id != full_id


def test_fused_smoke_and_full_products_are_incomparable(tmp_path: Path) -> None:
    """核心回归：--limit 冒烟融合产物 vs 全量融合产物，strict 下判不可比（FAIL）。

    两侧只有题目集不同（6 题 vs 27 题），此前 comparability 记 0 项不一致、
    verdict PASS、退出码 0；现在题目数（关键维度）不一致 -> FAIL。
    """

    def _shard(name: str, index: int, limit: int | None) -> str:
        return _shard_run(
            tmp_path,
            name,
            MockStrongAdapter(),
            2,
            index,
            limit=limit,
            driver_llm="gpt-4o",
            adapter_version="v1",
        )

    smoke = fuse_runs(
        [_shard("s0", 0, 3), _shard("s1", 1, 3)], str(tmp_path / "fused_smoke")
    )["dataset_dir"]
    full = fuse_runs(
        [_shard("t0", 0, None), _shard("t1", 1, None)], str(tmp_path / "fused_full")
    )["dataset_dir"]

    from membench.compare import compare_runs

    result = compare_runs(
        smoke, full, require_comparable=True, out_dir=str(tmp_path / "cmp")
    )
    comparability = result["comparability"]
    # 除题目集之外的维度都已一致：不可比必须来自题目数本身
    assert comparability["unverifiable"] == []
    assert comparability["comparable"] is False
    assert result["verdict"] == "FAIL"
    dims = {c["dimension"]: c for c in comparability["checks"]}
    assert dims["n_questions"]["same"] is False
    assert dims["n_questions"]["a"] != dims["n_questions"]["b"]
    assert dims["n_questions"]["severity"] == "critical"
    assert any("n_questions" in reason for reason in comparability["reasons"])
