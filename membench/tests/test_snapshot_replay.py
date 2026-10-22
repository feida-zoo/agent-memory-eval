"""状态快照与回放测试：导出 → 回放 → 保真校验 → 拒绝与可比性。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
from membench.compare import check_comparability
from membench.data.schema import dataset_fingerprint
from membench.runner import Runner
from membench import snapshot as snapshot_mod


class _FileMemoryAdapter(BaseMemoryAdapter):
    """文件型记忆系统：记忆写在 data_dir/memories.jsonl，支持快照 / 恢复。"""

    name = "mock-file"

    def __init__(
        self,
        data_dir: Path,
        *,
        format_version: int = 1,
        support_restore: bool = True,
        support_rebuild: bool = False,
        answer_suffix: str = "",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.format_version = format_version
        self.support_restore = support_restore
        self.support_rebuild = support_rebuild
        self.answer_suffix = answer_suffix  # 改读取侧：答案加后缀（模拟代码变更）

    # ---- 三动词契约 ----
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
        lines = []
        path = self.data_dir / "memories.jsonl"
        if path.is_file():
            lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return QueryResult(
            answer="周五" + self.answer_suffix,
            retrieved=lines,
            input_tokens=11,
            output_tokens=3,
        )

    # ---- 可选快照能力 ----
    def snapshot_capabilities(self) -> dict:
        return {
            "supported": True,
            "restore": self.support_restore,
            "rebuild": self.support_rebuild,
            "format_version": self.format_version,
            "compatible_from": [self.format_version],
        }

    def snapshot(self, target_dir: str, tag: str) -> dict:
        dest = Path(target_dir) / "memory"
        shutil.rmtree(dest, ignore_errors=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.data_dir.is_dir():
            shutil.copytree(self.data_dir, dest)
        else:
            dest.mkdir(parents=True, exist_ok=True)
        return {"layers": ["raw"], "items": sum(1 for _ in dest.rglob("*"))}

    def restore(self, source_dir: str, tag: str) -> None:
        src = Path(source_dir) / "memory"
        shutil.rmtree(self.data_dir, ignore_errors=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, self.data_dir, dirs_exist_ok=True)

    def rebuild(self, source_dir: str, tag: str) -> None:
        self.restore(source_dir, tag)  # 文件级重建 = 恢复后由系统自行重建派生结构


def _dataset(path: Path, marker: str = "") -> Path:
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
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    return path


def _full_run(tmp_path: Path, **kwargs) -> str:
    ds = _dataset(tmp_path / "ds.jsonl")
    return Runner(
        _FileMemoryAdapter(tmp_path / "mem"),
        str(ds),
        judge="heuristic",
        out_dir=str(tmp_path / "runs"),
        **kwargs,
    ).run()


def test_snapshot_export_writes_manifest_and_state(tmp_path: Path) -> None:
    """全量 run + --snapshot-out：清单落账（complete/指纹/每对话哈希）与指标元数据。"""
    snap_root = tmp_path / "snap"
    run_dir = _full_run(tmp_path, snapshot_out=str(snap_root))
    manifest = json.loads((snap_root / snapshot_mod.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert set(manifest["dialogues"]) == {"d1", "d2"}
    assert manifest["dataset_fingerprint"]["sha256"] == dataset_fingerprint(
        str(tmp_path / "ds.jsonl")
    )["sha256"]
    for entry in manifest["dialogues"].values():
        assert len(entry["sha256"]) == 64 and entry["bytes"] > 0
        assert entry["ingest_payload_sha256"]
    metrics = json.loads((Path(run_dir) / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["run_mode"] == "full"
    assert metrics["state_snapshot"]["role"] == "exported"
    assert metrics["state_snapshot"]["n_dialogues"] == 2
    assert "token_usage" in metrics  # 全量 run 照常记账


def test_replay_same_code_is_faithful(tmp_path: Path) -> None:
    """未改动系统时回放应与源 run 逐题一致；[必跑] ingest 被跳过、口径已标注。"""
    snap_root = tmp_path / "snap"
    source_run = _full_run(tmp_path, snapshot_out=str(snap_root))

    replay_run = Runner(
        _FileMemoryAdapter(tmp_path / "mem2"),  # 新进程/空目录：状态只能来自快照
        str(tmp_path / "ds.jsonl"),
        judge="heuristic",
        out_dir=str(tmp_path / "runs2"),
        replay=str(snap_root),
        replay_verify_source=source_run,
    ).run()
    metrics = json.loads((Path(replay_run) / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["run_mode"] == "replay-restore"
    assert metrics["state_snapshot"]["role"] == "replayed"
    assert metrics["state_snapshot"]["source_run"] == json.loads(
        (Path(source_run) / "metrics.json").read_text(encoding="utf-8")
    )["run_id"]
    assert metrics["token_usage"]["ingest_input_tokens"] == 0
    assert metrics["token_usage"]["ingest_skipped"] == 1
    assert metrics["replay_verify"]["n_answer_diff"] == 0
    assert metrics["replay_verify"]["n_retrieved_diff"] == 0
    assert metrics["replay_verify"]["answer_match_rate"] == 1.0

    md = (Path(replay_run) / "report.md").read_text(encoding="utf-8")
    assert "快照回放（写入侧固定）" in md
    assert "回放保真校验" in md
    assert "ingest 被跳过" in md


def test_replay_detects_read_side_change(tmp_path: Path) -> None:
    """回放能测出读取侧改动：改答案后保真校验报不一致（A/B 用法的地基）。"""
    snap_root = tmp_path / "snap"
    source_run = _full_run(tmp_path, snapshot_out=str(snap_root))
    replay_run = Runner(
        _FileMemoryAdapter(tmp_path / "mem2", answer_suffix="（改版）"),
        str(tmp_path / "ds.jsonl"),
        judge="heuristic",
        out_dir=str(tmp_path / "runs2"),
        replay=str(snap_root),
        replay_verify_source=source_run,
    ).run()
    verify = json.loads((Path(replay_run) / "metrics.json").read_text(encoding="utf-8"))[
        "replay_verify"
    ]
    assert verify["n_answer_diff"] == verify["n_common"] == 2
    assert verify["answer_match_rate"] == 0.0


class _NoSnapshotAdapter(_FileMemoryAdapter):
    def snapshot_capabilities(self) -> dict:
        return {"supported": False}


def test_replay_refuses_without_snapshot_support(tmp_path: Path) -> None:
    snap_root = tmp_path / "snap"
    _full_run(tmp_path, snapshot_out=str(snap_root))
    with pytest.raises(SystemExit, match="未声明快照能力"):
        Runner(
            _NoSnapshotAdapter(tmp_path / "mem2"),
            str(tmp_path / "ds.jsonl"),
            judge="heuristic",
            out_dir=str(tmp_path / "runs2"),
            replay=str(snap_root),
        ).run()


def test_snapshot_out_refuses_without_capability(tmp_path: Path) -> None:
    _dataset(tmp_path / "ds.jsonl")
    with pytest.raises(SystemExit, match="无法 --snapshot-out"):
        Runner(
            _NoSnapshotAdapter(tmp_path / "mem"),
            str(tmp_path / "ds.jsonl"),
            judge="heuristic",
            out_dir=str(tmp_path / "runs"),
            snapshot_out=str(tmp_path / "snap"),
        ).run()


def test_replay_refuses_on_dataset_fingerprint_mismatch(tmp_path: Path) -> None:
    snap_root = tmp_path / "snap"
    _full_run(tmp_path, snapshot_out=str(snap_root))
    other = _dataset(tmp_path / "other.jsonl", marker="（另一份数据集）")
    with pytest.raises(SystemExit, match="数据集指纹与快照不一致"):
        Runner(
            _FileMemoryAdapter(tmp_path / "mem2"),
            str(other),
            judge="heuristic",
            out_dir=str(tmp_path / "runs2"),
            replay=str(snap_root),
        ).run()


def test_replay_refuses_when_dialogue_missing(tmp_path: Path) -> None:
    """快照只覆盖一个对话（limit 冒烟导出）→ 全量回放被拒绝。"""
    snap_root = tmp_path / "snap"
    _full_run(tmp_path, snapshot_out=str(snap_root), limit=1)
    with pytest.raises(SystemExit, match="快照缺少本 run 需要的对话"):
        Runner(
            _FileMemoryAdapter(tmp_path / "mem2"),
            str(tmp_path / "ds.jsonl"),
            judge="heuristic",
            out_dir=str(tmp_path / "runs2"),
            replay=str(snap_root),
        ).run()


def test_replay_incompatible_format_needs_rebuild(tmp_path: Path) -> None:
    """格式不兼容且未声明 rebuild → 拒绝（提示必须重 ingest）。"""
    snap_root = tmp_path / "snap"
    _full_run(tmp_path, snapshot_out=str(snap_root))
    with pytest.raises(SystemExit, match="该改动必须重新 ingest"):
        Runner(
            _FileMemoryAdapter(tmp_path / "mem2", format_version=2),
            str(tmp_path / "ds.jsonl"),
            judge="heuristic",
            out_dir=str(tmp_path / "runs2"),
            replay=str(snap_root),
        ).run()


def test_replay_incompatible_format_with_rebuild(tmp_path: Path) -> None:
    """声明了 rebuild → 自动走 replay-rebuild（派生/结构变更的通道）。"""
    snap_root = tmp_path / "snap"
    source_run = _full_run(tmp_path, snapshot_out=str(snap_root))
    replay_run = Runner(
        _FileMemoryAdapter(tmp_path / "mem2", format_version=2, support_rebuild=True),
        str(tmp_path / "ds.jsonl"),
        judge="heuristic",
        out_dir=str(tmp_path / "runs2"),
        replay=str(snap_root),
        replay_verify_source=source_run,
    ).run()
    metrics = json.loads((Path(replay_run) / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["run_mode"] == "replay-rebuild"
    assert metrics["state_snapshot"]["replay_mode"] == "rebuild"
    assert metrics["replay_verify"]["n_answer_diff"] == 0


def test_run_mode_is_critical_for_comparability(tmp_path: Path) -> None:
    """回放 run 与全量 run 不可比；旧 run 缺 run_mode 按 full 归一（不误伤历史）。"""
    snap_root = tmp_path / "snap"
    full_run = _full_run(tmp_path, snapshot_out=str(snap_root))
    replay_run = Runner(
        _FileMemoryAdapter(tmp_path / "mem2"),
        str(tmp_path / "ds.jsonl"),
        judge="heuristic",
        out_dir=str(tmp_path / "runs2"),
        replay=str(snap_root),
    ).run()
    full = json.loads((Path(full_run) / "metrics.json").read_text(encoding="utf-8"))
    replay = json.loads((Path(replay_run) / "metrics.json").read_text(encoding="utf-8"))
    comp = check_comparability(full, replay)
    dims = {c["dimension"]: c for c in comp["checks"]}
    assert dims["run_mode"]["same"] is False
    assert dims["run_mode"]["severity"] == "critical"
    assert comp["comparable"] is False

    old = dict(full)
    del old["run_mode"]  # 本特性之前的历史 run
    assert check_comparability(old, full)["comparable"] is True


def test_http_adapter_snapshot_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """HttpAdapter 把快照操作转发到桥端点；老桥（异常）按不支持处理。"""
    import sys
    import types

    from membench.adapters.http import HttpAdapter

    calls: list[dict] = []

    class _Resp:
        def __init__(self, payload: dict) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return self._payload

    def _post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json})
        if url.endswith("/snapshot-capabilities"):
            return _Resp({"supported": True, "restore": True, "rebuild": False, "format_version": 2})
        return _Resp({"ok": True})

    monkeypatch.setitem(
        sys.modules,
        "requests",
        types.SimpleNamespace(post=_post),
    )
    adapter = HttpAdapter("http://example.invalid:9")
    caps = adapter.snapshot_capabilities()
    assert caps["supported"] is True and caps["format_version"] == 2
    adapter.snapshot("/tmp/snap/d1", "d1")
    adapter.restore("/tmp/snap/d1", "d1")
    adapter.rebuild("/tmp/snap/d1", "d1")
    assert [c["url"].rsplit("/", 1)[-1] for c in calls] == [
        "snapshot-capabilities",
        "snapshot",
        "restore",
        "rebuild",
    ]
    assert calls[1]["json"] == {"target_dir": "/tmp/snap/d1", "tag": "d1"}
    assert calls[2]["json"] == {"source_dir": "/tmp/snap/d1", "tag": "d1"}


def test_http_adapter_capabilities_absent_is_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    """老桥没有 /snapshot-capabilities（404 抛错）→ 按不支持处理，不中断评测。"""
    import sys
    import types

    from membench.adapters.http import HttpAdapter

    def _post(url, json=None, headers=None, timeout=None):
        raise RuntimeError("HTTP 404")

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=_post))
    adapter = HttpAdapter("http://example.invalid:9")
    caps = adapter.snapshot_capabilities()
    assert caps["supported"] is False and "404" in caps["reason"]
