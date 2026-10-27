"""状态快照与回放测试：导出 → 回放 → 保真校验 → 拒绝与可比性。

产物结构见 :mod:`membench.layout`：``Runner`` 的 ``out_dir`` 是**实验目录**，
``run()`` 返回数据集目录 ``<实验目录>/runs/<dataset>``，指标与逐题明细都在它的
``report/`` 下；快照清单**每个对话一份**，落 ``<对话目录>/memory_snapshot/manifest.json``。

快照**默认导出**（``snapshot=True``）；适配器未声明快照能力时只打印一次警告并跳过
（metrics 记 ``state_snapshot.role="unsupported"``）。回放同样导出，但落任务级
内容寻址池（见 ``tests/test_replay_snapshots.py``），不再写对话目录下的
``memory_snapshot/``。
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
from pathlib import Path

import pytest

from membench import layout
from membench import snapshot as snapshot_mod
from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
from membench.compare import check_comparability
from membench.data.schema import dataset_fingerprint
from membench.runner import Runner

#: 测试数据集文件名 stem —— 即实验目录下数据集目录的名字。
DATASET = "ds"


@pytest.fixture(autouse=True)
def _isolated_output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """默认产物根指向 tmp_path：任何走 output_root() 的路径都不落进仓库。"""
    root = tmp_path / "output-root"
    monkeypatch.setenv("MEMBENCH_OUTPUT_DIR", str(root))
    return root


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    return path


def _dataset_dir_of(experiment: Path) -> Path:
    """按 layout 契约定位实验目录下的数据集产物目录。"""
    return Path(layout.resolve_dataset_dir(str(experiment)))


def _report_file(dataset_dir: Path, filename: str) -> Path:
    return Path(dataset_dir) / layout.DIR_REPORT / filename


def _metrics(dataset_dir: Path) -> dict:
    return json.loads(
        _report_file(dataset_dir, layout.FILE_METRICS).read_text(encoding="utf-8")
    )


def _snapshot_dir(experiment: Path, dialogue_id: str) -> Path:
    return Path(layout.dialogue_snapshot_dir(str(experiment), DATASET, dialogue_id))


def _manifest(experiment: Path, dialogue_id: str) -> dict:
    path = _snapshot_dir(experiment, dialogue_id) / snapshot_mod.MANIFEST_NAME
    return json.loads(path.read_text(encoding="utf-8"))


def _full_run(tmp_path: Path, *, name: str = "src", **kwargs) -> Path:
    """全量 run（2 个对话 × 1 题），返回数据集目录。"""
    ds = _dataset(tmp_path / "ds.jsonl")
    out = Runner(
        _FileMemoryAdapter(tmp_path / f"mem-{name}"),
        str(ds),
        judge="heuristic",
        out_dir=str(tmp_path / name),
        **kwargs,
    ).run()
    assert out is not None
    return Path(out)


def _replay_run(
    tmp_path: Path,
    *,
    name: str,
    replay: Path,
    adapter: BaseMemoryAdapter | None = None,
    verify_source: Path | None = None,
) -> Path:
    """回放 run（新进程空目录：状态只能来自快照），返回数据集目录。"""
    out = Runner(
        adapter or _FileMemoryAdapter(tmp_path / f"mem-{name}"),
        str(tmp_path / "ds.jsonl"),
        judge="heuristic",
        out_dir=str(tmp_path / name),
        replay=str(replay),
        replay_verify_source=str(verify_source) if verify_source else None,
    ).run()
    assert out is not None
    return Path(out)


def test_snapshot_writes_per_dialogue_manifest_and_state(tmp_path: Path) -> None:
    """全量 run + snapshot=True：逐对话清单落账（complete / 指纹 / 条目哈希）与指标元数据。"""
    exp = tmp_path / "src"
    dataset_dir = _full_run(tmp_path, name="src", snapshot=True)
    fingerprint = dataset_fingerprint(str(tmp_path / "ds.jsonl"))["sha256"]
    # 实验目录入参经 layout 归一后就是 run() 返回的数据集目录（读取方唯一入口）
    assert _dataset_dir_of(exp) == dataset_dir

    for dialogue_id in ("d1", "d2"):
        manifest = _manifest(exp, dialogue_id)
        assert manifest["complete"] is True
        assert manifest["dialogue_id"] == dialogue_id
        assert manifest["dataset"] == DATASET
        assert manifest["dataset_fingerprint"]["sha256"] == fingerprint
        assert manifest["capabilities"]["format_version"] == 1
        # 清单自包含：一份清单只登记自己这一段，不携带其它对话的条目
        assert "dialogues" not in manifest
        entry = manifest["entry"]
        assert entry["dir"] == entry["tag"] == dialogue_id
        assert len(entry["sha256"]) == 64 and entry["bytes"] > 0
        assert entry["ingest_payload_sha256"]
        assert (entry["n_sessions"], entry["n_messages"], entry["n_questions"]) == (1, 1, 1)
        # 清单落在它登记的那一层目录里（位置与内容互为凭证）
        assert _snapshot_dir(exp, dialogue_id).parent.name == entry["dir"]

    metrics = _metrics(dataset_dir)
    assert metrics["run_mode"] == "full"
    assert metrics["state_snapshot"]["role"] == "exported"
    assert metrics["state_snapshot"]["n_dialogues"] == 2
    assert metrics["state_snapshot"]["root"] == str(dataset_dir)
    assert metrics["state_snapshot"]["manifest_sha256"] == snapshot_mod.manifests_digest(
        {d: str(_snapshot_dir(exp, d)) for d in ("d1", "d2")}
    )
    assert "token_usage" in metrics  # 全量 run 照常记账

    # 对话级 metrics 只登记该对话那一份清单（合并指纹随范围收窄）
    d1_metrics = json.loads(
        (
            Path(layout.dialogue_report_dir(str(exp), DATASET, "d1"))
            / layout.FILE_METRICS
        ).read_text(encoding="utf-8")
    )
    assert d1_metrics["state_snapshot"]["role"] == "exported"
    assert d1_metrics["state_snapshot"]["n_dialogues"] == 1
    assert d1_metrics["state_snapshot"]["manifest_sha256"] == snapshot_mod.manifests_digest(
        {"d1": str(_snapshot_dir(exp, "d1"))}
    )


def test_replay_same_code_is_faithful(tmp_path: Path) -> None:
    """未改动系统时回放应与源 run 逐题一致；[必跑] ingest 被跳过、口径已标注。"""
    exp = tmp_path / "src"
    source_dataset = _full_run(tmp_path, name="src", snapshot=True)
    # --replay 传实验目录：自动定位到其中的数据集目录
    replay_dataset = _replay_run(
        tmp_path, name="replay", replay=exp, verify_source=source_dataset
    )
    metrics = _metrics(replay_dataset)
    assert metrics["run_mode"] == "replay-restore"
    assert metrics["state_snapshot"]["role"] == "replayed"
    assert metrics["state_snapshot"]["root"] == str(source_dataset)
    assert metrics["state_snapshot"]["source_run"] == _metrics(source_dataset)["run_id"]
    assert metrics["state_snapshot"]["n_dialogues"] == 2
    assert metrics["token_usage"]["ingest_input_tokens"] == 0
    assert metrics["token_usage"]["ingest_llm_calls"] == 0
    assert metrics["token_usage"]["ingest_skipped"] == 1
    assert metrics["replay_verify"]["n_answer_diff"] == 0
    assert metrics["replay_verify"]["n_retrieved_diff"] == 0
    assert metrics["replay_verify"]["answer_match_rate"] == 1.0

    md = _report_file(replay_dataset, layout.FILE_REPORT_MD).read_text(encoding="utf-8")
    assert "快照回放（写入侧固定）" in md
    assert "回放保真校验" in md
    assert "ingest 被跳过" in md

    # 逐对话产物同样标注回放口径，不留只按全量读的半成品
    d1_metrics = json.loads(
        (
            Path(layout.dialogue_report_dir(str(tmp_path / "replay"), DATASET, "d1"))
            / layout.FILE_METRICS
        ).read_text(encoding="utf-8")
    )
    assert d1_metrics["state_snapshot"]["role"] == "replayed"
    assert d1_metrics["state_snapshot"]["n_dialogues"] == 1
    assert d1_metrics["token_usage"]["ingest_skipped"] == 1


def test_replay_detects_read_side_change(tmp_path: Path) -> None:
    """回放能测出读取侧改动：改答案后保真校验报不一致（A/B 用法的地基）。"""
    source_dataset = _full_run(tmp_path, name="src", snapshot=True)
    # --replay 传数据集目录：直接可用
    replay_dataset = _replay_run(
        tmp_path,
        name="replay",
        replay=source_dataset,
        adapter=_FileMemoryAdapter(tmp_path / "mem-replay", answer_suffix="（改版）"),
        verify_source=source_dataset,
    )
    verify = _metrics(replay_dataset)["replay_verify"]
    assert verify["n_answer_diff"] == verify["n_common"] == 2
    assert verify["answer_match_rate"] == 0.0


class _NoSnapshotAdapter(_FileMemoryAdapter):
    def snapshot_capabilities(self) -> dict:
        return {"supported": False}


def test_replay_refuses_without_snapshot_support(tmp_path: Path) -> None:
    source_dataset = _full_run(tmp_path, name="src", snapshot=True)
    with pytest.raises(SystemExit, match="未声明快照能力"):
        Runner(
            _NoSnapshotAdapter(tmp_path / "mem-replay"),
            str(tmp_path / "ds.jsonl"),
            judge="heuristic",
            out_dir=str(tmp_path / "replay"),
            replay=str(source_dataset),
        ).run()
    # 跑前拒绝：不为被拒的回放建实验目录
    assert not (tmp_path / "replay").exists()


def test_snapshot_defaults_on(tmp_path: Path) -> None:
    """快照默认导出：不传 snapshot 也逐对话落清单，metrics 记 exported。"""
    exp = tmp_path / "src"
    dataset_dir = _full_run(tmp_path, name="src")
    for dialogue_id in ("d1", "d2"):
        assert (_snapshot_dir(exp, dialogue_id) / snapshot_mod.MANIFEST_NAME).is_file()
    state = _metrics(dataset_dir)["state_snapshot"]
    assert state["role"] == "exported" and state["n_dialogues"] == 2


def test_snapshot_oversize_warns_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """单份快照超过阈值时只告警一次，并提示可用 --no-snapshot 关闭。"""
    monkeypatch.setattr("membench.runner.SNAPSHOT_WARN_BYTES", 1)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        _full_run(tmp_path, name="src")
    out = buffer.getvalue()
    assert out.count("--no-snapshot 关闭") == 1
    # 告警不改变产物：两份清单照旧写完
    for dialogue_id in ("d1", "d2"):
        assert (_snapshot_dir(tmp_path / "src", dialogue_id) / snapshot_mod.MANIFEST_NAME).is_file()


def test_snapshot_unsupported_adapter_warns_and_skips(tmp_path: Path) -> None:
    """适配器不支持快照：只警告一次、跳过导出、metrics 记 unsupported，评测照常完成。"""
    _dataset(tmp_path / "ds.jsonl")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        dataset_dir = Path(
            Runner(
                _NoSnapshotAdapter(tmp_path / "mem"),
                str(tmp_path / "ds.jsonl"),
                judge="heuristic",
                out_dir=str(tmp_path / "exp"),
            ).run()
        )
    assert buffer.getvalue().count("未声明快照能力") == 1
    metrics = _metrics(dataset_dir)
    state = metrics["state_snapshot"]
    assert state["role"] == "unsupported"
    assert state["root"] == str(dataset_dir)
    assert state["n_dialogues"] == 2
    assert state["manifest_sha256"] is None and state["format_version"] is None
    # 跳过的只是快照：指标与逐对话产物照常，也不留空的 memory_snapshot/ 目录
    assert metrics["overall"]["n_questions"] == 2
    assert not list(dataset_dir.glob(f"*/{layout.DIR_SNAPSHOT}"))


def test_replay_exports_to_task_pool_not_dialogue_dirs(tmp_path: Path) -> None:
    """回放同样导出快照，但落任务级内容寻址池：不写对话目录下的 memory_snapshot/。"""
    exp = tmp_path / "src"
    source_dataset = _full_run(tmp_path, name="src", snapshot=True)

    replay_dataset = _replay_run(tmp_path, name="replay", replay=exp)
    assert _metrics(replay_dataset)["state_snapshot"]["role"] == "replayed"
    assert not list(Path(replay_dataset).glob(f"*/{layout.DIR_SNAPSHOT}"))
    # 池在实验目录（单臂任务根）下：两个对话各一个桶 + 各一个逐对话条目
    task_dir = tmp_path / "replay"
    pool = task_dir / layout.DIR_SNAPSHOTS
    assert layout.has_snapshot_pool(str(task_dir))
    index = layout.materialize_snapshot_index(str(pool))
    assert len(index["buckets"]) == 2
    entries_dir = pool / layout.DIR_SNAPSHOT_ENTRIES / layout.snapshot_arm_segment("")
    entry_files = sorted(entries_dir.glob(f"*/{layout.tag_dir_name('d1')}.json"))
    assert entry_files and entry_files[0].is_file()
    assert sorted(path.stem for path in (pool / layout.DIR_SNAPSHOT_ENTRIES).glob("*/*/*.json")) == [
        "d1",
        "d2",
    ]
    assert _metrics(replay_dataset)["state_snapshot"]["snapshots_dir"] == str(pool)

    # 显式 snapshot=True + replay：与默认一致（导出到池，构造期不报错）
    explicit = Runner(
        _FileMemoryAdapter(tmp_path / "mem-explicit"),
        str(tmp_path / "ds.jsonl"),
        judge="heuristic",
        out_dir=str(tmp_path / "replay-explicit"),
        replay=str(source_dataset),
        snapshot=True,
    )
    explicit_dir = Path(explicit.run())
    assert _metrics(explicit_dir)["state_snapshot"]["role"] == "replayed"
    assert not list(explicit_dir.glob(f"*/{layout.DIR_SNAPSHOT}"))
    assert (
        tmp_path
        / "replay-explicit"
        / layout.DIR_SNAPSHOTS
        / layout.DIR_SNAPSHOT_ENTRIES
        / layout.snapshot_arm_segment("")
    ).is_dir()


def test_replay_refuses_on_other_dataset_or_changed_content(tmp_path: Path) -> None:
    """回放源的两种"不是同一份数据集"：数据集名不同（另一份数据集）与同名但内容漂移（指纹不符）。"""
    source_dataset = _full_run(tmp_path, name="src", snapshot=True)

    # 同名（都叫 ds）但内容改过 -> 指纹不一致
    drifted = _dataset(tmp_path / "drifted" / "ds.jsonl", marker="（另一份数据集）")
    with pytest.raises(SystemExit, match="数据集指纹与快照不一致"):
        Runner(
            _FileMemoryAdapter(tmp_path / "mem-drifted"),
            str(drifted),
            judge="heuristic",
            out_dir=str(tmp_path / "replay-drifted"),
            replay=str(source_dataset),
        ).run()

    # 另一份数据集（文件名不同）-> 清单身份校验先拦下
    other = _dataset(tmp_path / "other.jsonl", marker="（另一份数据集）")
    with pytest.raises(SystemExit, match="快照来自另一个数据集"):
        Runner(
            _FileMemoryAdapter(tmp_path / "mem-other"),
            str(other),
            judge="heuristic",
            out_dir=str(tmp_path / "replay-other"),
            replay=str(source_dataset),
        ).run()
    assert not (tmp_path / "replay-drifted").exists()
    assert not (tmp_path / "replay-other").exists()


def test_replay_refuses_when_dialogue_missing(tmp_path: Path) -> None:
    """快照只覆盖一个对话（limit 冒烟导出）→ 全量回放被拒绝。"""
    exp = tmp_path / "src"
    source_dataset = _full_run(tmp_path, name="src", snapshot=True, limit=1)
    # limit=1 只跑了 d1：d2 连快照目录都没有
    assert (_snapshot_dir(exp, "d1") / snapshot_mod.MANIFEST_NAME).is_file()
    assert not _snapshot_dir(exp, "d2").exists()
    with pytest.raises(SystemExit, match="快照缺少本 run 需要的对话") as excinfo:
        Runner(
            _FileMemoryAdapter(tmp_path / "mem-replay"),
            str(tmp_path / "ds.jsonl"),
            judge="heuristic",
            out_dir=str(tmp_path / "replay"),
            replay=str(source_dataset),
        ).run()
    assert "d2" in str(excinfo.value)  # 报错点名缺哪个对话
    assert not (tmp_path / "replay").exists()


def test_replay_refuses_on_incomplete_or_moved_manifest(tmp_path: Path) -> None:
    """逐对话清单本身是回放门槛：导出被中断（complete=false）或目录被搬动（entry.dir 不符）都拒绝。"""
    exp = tmp_path / "src"
    source_dataset = _full_run(tmp_path, name="src", snapshot=True)

    manifest_path = _snapshot_dir(exp, "d1") / snapshot_mod.MANIFEST_NAME
    original = manifest_path.read_text(encoding="utf-8")
    moved = json.loads(original)
    moved["entry"]["dir"] = "moved-elsewhere"
    manifest_path.write_text(json.dumps(moved, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SystemExit, match="对话目录与所在位置不一致"):
        Runner(
            _FileMemoryAdapter(tmp_path / "mem-moved"),
            str(tmp_path / "ds.jsonl"),
            judge="heuristic",
            out_dir=str(tmp_path / "replay-moved"),
            replay=str(source_dataset),
        ).run()
    manifest_path.write_text(original, encoding="utf-8")

    incomplete_path = _snapshot_dir(exp, "d2") / snapshot_mod.MANIFEST_NAME
    incomplete = json.loads(incomplete_path.read_text(encoding="utf-8"))
    incomplete["complete"] = False
    incomplete_path.write_text(json.dumps(incomplete, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SystemExit, match="未完成"):
        Runner(
            _FileMemoryAdapter(tmp_path / "mem-incomplete"),
            str(tmp_path / "ds.jsonl"),
            judge="heuristic",
            out_dir=str(tmp_path / "replay-incomplete"),
            replay=str(source_dataset),
        ).run()
    assert not (tmp_path / "replay-moved").exists()
    assert not (tmp_path / "replay-incomplete").exists()


def test_replay_incompatible_format_needs_rebuild(tmp_path: Path) -> None:
    """格式不兼容且未声明 rebuild → 拒绝（提示必须重 ingest）。"""
    source_dataset = _full_run(tmp_path, name="src", snapshot=True)
    with pytest.raises(SystemExit, match="该改动必须重新 ingest"):
        Runner(
            _FileMemoryAdapter(tmp_path / "mem-replay", format_version=2),
            str(tmp_path / "ds.jsonl"),
            judge="heuristic",
            out_dir=str(tmp_path / "replay"),
            replay=str(source_dataset),
        ).run()


def test_replay_incompatible_format_with_rebuild(tmp_path: Path) -> None:
    """声明了 rebuild → 自动走 replay-rebuild（派生/结构变更的通道）。"""
    source_dataset = _full_run(tmp_path, name="src", snapshot=True)
    replay_dataset = _replay_run(
        tmp_path,
        name="replay",
        replay=source_dataset,
        adapter=_FileMemoryAdapter(
            tmp_path / "mem-replay", format_version=2, support_rebuild=True
        ),
        verify_source=source_dataset,
    )
    metrics = _metrics(replay_dataset)
    assert metrics["run_mode"] == "replay-rebuild"
    assert metrics["state_snapshot"]["replay_mode"] == "rebuild"
    assert metrics["replay_verify"]["n_answer_diff"] == 0


def test_run_mode_is_critical_for_comparability(tmp_path: Path) -> None:
    """回放 run 与全量 run 不可比；旧 run 缺 run_mode 按 full 归一（不误伤历史）。"""
    exp = tmp_path / "src"
    full_dataset = _full_run(tmp_path, name="src", snapshot=True)
    replay_dataset = _replay_run(tmp_path, name="replay", replay=exp)
    full = _metrics(full_dataset)
    replay = _metrics(replay_dataset)
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
