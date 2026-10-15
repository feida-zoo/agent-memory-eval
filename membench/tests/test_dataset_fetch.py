"""dataset fetch 后端（membench.data.fetch）测试：全程离线，只 monkeypatch 下载函数。

覆盖：key 集合与 is_fetchable、产物路径（默认目录 / MEMBENCH_DATASETS_DIR）、
未知 key 抛错、各数据集「原始数据 -> 转换 -> load_dataset 可加载」端到端、
已存在则跳过（不触发下载）、空/纯空白产物不算已完成、force 绕过 raw 缓存重下、
_download_file 跳过完整文件与重试。
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from membench.data import fetch
from membench.data.schema import load_dataset

# ---------------------------------------------------------------------------
# 各 converter 期望格式的最小原始数据
# ---------------------------------------------------------------------------

_LOCOMO_RAW: list[dict] = [
    {
        "sample_id": "conv-test",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": "2023-05-01 10:00",
            "session_1": [
                {"speaker": "Alice", "dia_id": "D1:1", "text": "I adopted a dog named Rex."},
                {"speaker": "Bob", "dia_id": "D1:2", "text": "Nice!"},
            ],
        },
        "qa": [
            {"question": "What is the dog's name?", "answer": "Rex",
             "evidence": ["D1:1"], "category": 4},
            {"question": "When did Rex turn three?", "answer": "2023",
             "evidence": [], "category": 2},
            {"question": "What is Alice's favorite color?", "evidence": [], "category": 5},
        ],
    }
]

#: oracle 变体：无 haystack_session_ids，evidence 保持为空。
_LME_ORACLE_RAW: list[dict] = [
    {
        "question_id": "lme_q1",
        "question": "What city did I move to?",
        "answer": "Berlin",
        "question_type": "knowledge-update",
        "haystack_sessions": [[{"role": "user", "content": "I moved to Berlin."}]],
        "haystack_dates": ["2024-01-01"],
        "answer_session_ids": [],
    },
    {
        "question_id": "lme_q2",
        "question": "What is my favorite genre?",
        "answer": "jazz",
        "question_type": "single-session-preference",
        "haystack_sessions": [[{"role": "user", "content": "I love jazz."}]],
        "haystack_dates": ["2024-01-03"],
        "answer_session_ids": [],
    },
]

_GATEMEM_EPISODES: list[dict] = [
    {
        "episode_id": "ep_001",
        "domain": "office",
        "entities": {"principals": [{"principal_id": "pm_ava", "role": "product_manager"}]},
        "turns": [
            {"turn_id": "t001", "timestamp": "2026-04-06T09:00",
             "speaker": {"principal_id": "pm_ava", "role": "product_manager"},
             "turn_kind": "dialogue", "text": "Maple 项目预算 221,000 USD。"},
        ],
    },
    {
        "episode_id": "ep_002",
        "domain": "office",
        "entities": {"principals": []},
        "turns": [
            {"turn_id": "t001", "timestamp": "2026-04-07T10:00",
             "speaker": {"principal_id": "sre_jordan", "role": "sre"},
             "turn_kind": "dialogue", "text": "旧 token 已删除。"},
        ],
    },
]


def _cp(cp_id: str, ep_id: str, query_type: str, action: str, include: list[str]) -> dict:
    return {
        "checkpoint_id": cp_id,
        "episode_id": ep_id,
        "as_of_turn_id": "t001",
        "asker": {"principal_id": "pm_ava", "role": "product_manager"},
        "query_type": query_type,
        "query_text": f"{cp_id} 的问题文本",
        "expected_action": action,
        "judge_spec": {"include": include, "not_include": [], "notes": ""},
        "leak_targets": [],
    }


_GATEMEM_CHECKPOINTS: list[dict] = [
    _cp("ck_utility", "ep_001", "utility", "answer", ["221,000 USD"]),
    _cp("ck_refuse", "ep_001", "privacy", "refuse", []),
    _cp("ck_redacted", "ep_001", "privacy", "answer_redacted", ["是同一个账户"]),
    _cp("ck_safety", "ep_002", "safety", "no_memory", []),
]

_BEAM_CHAT: list[dict] = [
    {"batch_number": 1, "time_anchor": "2024-01-01",
     "turns": [[{"role": "user", "content": "你好"}]]}
]
_BEAM_PQ: dict = {"information_extraction": [{"question": "q?", "answer": "a"}]}


def _jsonl(records: list[dict]) -> str:
    """把 dict 列表序列化为 JSONL 文本。"""
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"


def _make_fake_download(payloads: dict[str, Any]) -> tuple[Any, list[tuple[str, Path]]]:
    """构造按文件名派发的假下载器；返回 (fake, 调用记录)。"""
    calls: list[tuple[str, Path]] = []

    def fake(url: str, dest: Path, retries: int = 3, force: bool = False) -> None:
        dest = Path(dest)
        calls.append((url, dest))
        if dest.name not in payloads:
            raise AssertionError(f"未预期的下载: {url} -> {dest}")
        payload = payloads[dest.name]
        dest.parent.mkdir(parents=True, exist_ok=True)
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        dest.write_text(text, encoding="utf-8")

    return fake, calls


class _FakeResponse:
    """``urlopen`` 返回值的最小替身：只需支持 ``with`` 与 ``read()``。"""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._payload


def _stub_urlopen(payload: bytes, calls: list[str]) -> Any:
    """构造 ``urllib.request.urlopen`` 替身：固定返回 payload，并记录每次 URL。"""

    def fake(url: str, timeout: float | None = None) -> _FakeResponse:
        calls.append(str(url))
        return _FakeResponse(payload)

    return fake


# ---------------------------------------------------------------------------
# key 集合 / 路径
# ---------------------------------------------------------------------------


def test_fetch_keys_and_is_fetchable() -> None:
    assert set(fetch.FETCH_KEYS) == {
        "locomo",
        "longmemeval_oracle",
        "longmemeval_s",
        "gatemem_office",
        "beam_1m",
        "beam_10m",
    }
    for key in fetch.FETCH_KEYS:
        assert fetch.is_fetchable(key)
    for key in ("office", "sample", "statemembench", "no_such_dataset"):
        assert not fetch.is_fetchable(key)


def test_dataset_out_path_default_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMBENCH_DATASETS_DIR", raising=False)
    path = fetch.dataset_out_path("locomo")
    assert path.name == "locomo.jsonl"
    assert path.parent == Path.home() / ".membench" / "datasets"


def test_dataset_out_path_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MEMBENCH_DATASETS_DIR", str(tmp_path))
    assert fetch.dataset_out_path("locomo") == tmp_path / "locomo.jsonl"
    # 显式 out_dir 优先于环境变量
    assert fetch.dataset_out_path("locomo", out_dir=tmp_path / "custom") == (
        tmp_path / "custom" / "locomo.jsonl"
    )


# ---------------------------------------------------------------------------
# 错误分支
# ---------------------------------------------------------------------------


def test_fetch_unknown_key_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        fetch.fetch_dataset("no_such_dataset")
    # 内置 / 不可用的 key 也不在 fetch 范围内
    with pytest.raises(KeyError):
        fetch.fetch_dataset("office")
    with pytest.raises(KeyError):
        fetch.fetch_dataset("statemembench")


def test_fetch_source_unconfirmed_raises_runtimeerror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """模拟某个 key 登记为可 fetch 但缺少下载实现 -> RuntimeError 且给出来源。"""
    monkeypatch.delitem(fetch._FETCHERS, "locomo")
    with pytest.raises(RuntimeError, match="手动获取"):
        fetch.fetch_dataset("locomo", out_dir=tmp_path)


# ---------------------------------------------------------------------------
# 端到端：原始数据 -> 转换 -> load_dataset 可加载
# ---------------------------------------------------------------------------


def test_fetch_locomo_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, calls = _make_fake_download({"locomo10.json": _LOCOMO_RAW})
    monkeypatch.setattr(fetch, "_download_file", fake)

    out = fetch.fetch_dataset("locomo", out_dir=tmp_path)
    assert out == tmp_path / "locomo.jsonl"
    assert out.is_file()
    assert [c[0] for c in calls] == [
        "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
    ]

    dialogues = load_dataset(out)
    assert len(dialogues) == 1
    assert sum(len(d.questions) for d in dialogues) == 3
    assert dialogues[0].dialogue_id == "conv-test"
    # 原始数据缓存在 .<key>_raw/ 下
    assert (tmp_path / ".locomo_raw" / "locomo10.json").is_file()


def test_fetch_longmemeval_oracle_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, calls = _make_fake_download({"longmemeval_oracle.json": _LME_ORACLE_RAW})
    monkeypatch.setattr(fetch, "_download_file", fake)

    out = fetch.fetch_dataset("longmemeval_oracle", out_dir=tmp_path)
    assert out.is_file()
    assert calls[0][0].endswith(
        "/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json"
    )
    dialogues = load_dataset(out)
    assert len(dialogues) == 2
    assert sum(len(d.questions) for d in dialogues) == 2


def test_fetch_longmemeval_s_uses_cleaned_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S 变体远端文件名是 longmemeval_s_cleaned.json（不是 longmemeval_s.json）。"""
    fake, calls = _make_fake_download({"longmemeval_s_cleaned.json": _LME_ORACLE_RAW})
    monkeypatch.setattr(fetch, "_download_file", fake)

    out = fetch.fetch_dataset("longmemeval_s", out_dir=tmp_path)
    assert out == tmp_path / "longmemeval_s.jsonl"  # 本地产物仍按 key 命名
    assert calls[0][0] == (
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
        "resolve/main/longmemeval_s_cleaned.json"
    )
    assert len(load_dataset(out)) == 2


def test_fetch_gatemem_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, calls = _make_fake_download(
        {
            "episodes.jsonl": _jsonl(_GATEMEM_EPISODES),
            "checkpoints.jsonl": _jsonl(_GATEMEM_CHECKPOINTS),
        }
    )
    monkeypatch.setattr(fetch, "_download_file", fake)

    out = fetch.fetch_dataset("gatemem_office", out_dir=tmp_path)
    assert out.is_file()
    assert sorted(c[1].name for c in calls) == ["checkpoints.jsonl", "episodes.jsonl"]

    dialogues = load_dataset(out)
    assert len(dialogues) == 2
    questions = [q for d in dialogues for q in d.questions]
    assert len(questions) == 4
    by_id = {q.qid: q for q in questions}
    # 单用户转换：privacy-refuse 题被标记 ignore
    assert by_id["ck_refuse"].category == "privacy_refuse"
    assert by_id["ck_refuse"].ignore is True
    assert by_id["ck_utility"].category == "utility_answer"
    assert by_id["ck_utility"].gold_answer == "221,000 USD"
    assert by_id["ck_safety"].category == "safety_no_memory"
    assert by_id["ck_safety"].ignore is False


def test_fetch_beam_minimal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, _ = _make_fake_download({"chat.json": _BEAM_CHAT, "probing_questions.json": _BEAM_PQ})
    monkeypatch.setattr(fetch, "_download_file", fake)
    # 缩小到 1 个 chat，避免写 35 份文件
    monkeypatch.setitem(fetch._BEAM_CONFIG, "beam_1m", ("1M", 1, "1m"))

    out = fetch.fetch_dataset("beam_1m", out_dir=tmp_path)
    assert out.is_file()
    dialogues = load_dataset(out)
    assert len(dialogues) == 1
    assert len(dialogues[0].questions) == 1
    # BEAM 1:1 直传，第一类（information_extraction）保留原 key。
    assert dialogues[0].questions[0].category == "information_extraction"


# ---------------------------------------------------------------------------
# 幂等 / force
# ---------------------------------------------------------------------------


def test_fetch_skips_download_when_output_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, calls = _make_fake_download({"locomo10.json": _LOCOMO_RAW})
    monkeypatch.setattr(fetch, "_download_file", fake)

    first = fetch.fetch_dataset("locomo", out_dir=tmp_path)
    n_after_first = len(calls)
    assert n_after_first == 1

    # 第二次：产物可加载 -> 直接返回，不再触发下载
    second = fetch.fetch_dataset("locomo", out_dir=tmp_path)
    assert second == first
    assert len(calls) == n_after_first


def test_fetch_force_redownloads_bypassing_raw_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """force=True 真的绕过 raw 缓存重新下载，产物随之更新。

    预置一份**内容合法**的 raw 缓存：非 force 路径必须沿用它（不联网）并重跑转换；
    force=True 时必须重新联网下载，转换出的产物随之变为新内容。旧行为下
    ``_download_file`` 见到完整缓存直接短路，force 只重跑转换——既不联网、产物也
    停留在旧内容，本用例据此在旧行为下失败。
    """
    raw_dir = tmp_path / ".locomo_raw"
    raw_dir.mkdir()
    (raw_dir / "locomo10.json").write_text(
        json.dumps(_LOCOMO_RAW, ensure_ascii=False), encoding="utf-8"
    )

    # 远端返回更新后的原始数据（多一个 dialogue），据此判断产物是否真的更新。
    updated = _LOCOMO_RAW + [
        {
            "sample_id": "conv-test-2",
            "conversation": {
                "speaker_a": "Cara",
                "speaker_b": "Dan",
                "session_1_date_time": "2023-06-01 09:00",
                "session_1": [
                    {"speaker": "Cara", "dia_id": "D1:1", "text": "I moved to Lisbon."}
                ],
            },
            "qa": [
                {"question": "Where did Cara move?", "answer": "Lisbon",
                 "evidence": [], "category": 4}
            ],
        }
    ]
    net: list[str] = []
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        _stub_urlopen(json.dumps(updated, ensure_ascii=False).encode("utf-8"), net),
    )

    # force=False：raw 缓存完整 -> 不联网，但仍重跑转换产出产物（幂等路径）
    out = fetch.fetch_dataset("locomo", out_dir=tmp_path)
    assert net == []
    assert len(load_dataset(out)) == 1

    # force=True：绕过 raw 缓存重新下载 -> 产物随之更新为 2 个 dialogue
    out2 = fetch.fetch_dataset("locomo", out_dir=tmp_path, force=True)
    assert out2 == out
    assert net == [
        "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
    ]
    assert len(load_dataset(out2)) == 2


def test_is_loadable_requires_at_least_one_dialogue(tmp_path: Path) -> None:
    """空 / 纯空白文件可被 load_dataset 解析为 []，但不算「可加载的产物」。"""
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    blank = tmp_path / "blank.jsonl"
    blank.write_text("\n   \t\n", encoding="utf-8")
    good = tmp_path / "good.jsonl"
    good.write_text(
        _jsonl(
            [
                {
                    "dialogue_id": "d1",
                    "sessions": [[{"role": "user", "content": "hi"}]],
                    "questions": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    assert fetch._is_loadable(empty) is False
    assert fetch._is_loadable(blank) is False
    assert fetch._is_loadable(good) is True


def test_fetch_redownloads_empty_output_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """0 字节产物不算「已完成」（load_dataset 会解析成空集）：应重新下载并转换。"""
    fake, calls = _make_fake_download({"locomo10.json": _LOCOMO_RAW})
    monkeypatch.setattr(fetch, "_download_file", fake)
    (tmp_path / "locomo.jsonl").write_bytes(b"")

    out = fetch.fetch_dataset("locomo", out_dir=tmp_path)
    assert len(calls) == 1
    assert len(load_dataset(out)) == 1


def test_fetch_redownloads_whitespace_output_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """纯空白产物同样不算「已完成」：应重新下载并转换。"""
    fake, calls = _make_fake_download({"locomo10.json": _LOCOMO_RAW})
    monkeypatch.setattr(fetch, "_download_file", fake)
    (tmp_path / "locomo.jsonl").write_text("\n   \t\n", encoding="utf-8")

    out = fetch.fetch_dataset("locomo", out_dir=tmp_path)
    assert len(calls) == 1
    assert len(load_dataset(out)) == 1


def test_fetch_ignores_corrupt_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """产物是半成品（不可被 load_dataset 解析）时应重做。"""
    out_path = tmp_path / "locomo.jsonl"
    out_path.write_text("{不是合法 jsonl", encoding="utf-8")

    fake, calls = _make_fake_download({"locomo10.json": _LOCOMO_RAW})
    monkeypatch.setattr(fetch, "_download_file", fake)

    out = fetch.fetch_dataset("locomo", out_dir=tmp_path)
    assert len(calls) == 1
    assert len(load_dataset(out)) == 1


# ---------------------------------------------------------------------------
# _download_file：跳过完整文件与重试
# ---------------------------------------------------------------------------


def test_download_file_skips_complete_files(tmp_path: Path) -> None:
    """已存在且完整（合法 JSON 或 JSONL）的文件不再联网。"""
    json_path = tmp_path / "x.json"
    json_path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    jsonl_path = tmp_path / "y.jsonl"
    jsonl_path.write_text(_jsonl([{"a": 1}, {"b": 2}]), encoding="utf-8")

    # URL 不可达：若真去下载会重试并抛错，据此证明被跳过
    fetch._download_file("http://invalid.invalid/x.json", json_path)
    fetch._download_file("http://invalid.invalid/y.jsonl", jsonl_path)
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"a": 1}
    assert jsonl_path.read_text(encoding="utf-8") == _jsonl([{"a": 1}, {"b": 2}])


def test_download_file_force_bypasses_complete_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """已存在且完整的文件在 force=True 时也被重新下载并覆盖。"""
    json_path = tmp_path / "x.json"
    json_path.write_text(json.dumps({"a": 1}), encoding="utf-8")

    net: list[str] = []
    monkeypatch.setattr(
        urllib.request, "urlopen", _stub_urlopen(b'{"a": 2}', net)
    )
    fetch._download_file("http://example.invalid/x.json", json_path, force=True)

    assert net == ["http://example.invalid/x.json"]
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"a": 2}


def test_download_file_retries_then_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*args: object, **kwargs: object) -> None:
        raise OSError("network down")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    with pytest.raises(RuntimeError, match="下载失败"):
        fetch._download_file("http://invalid.invalid/x.json", tmp_path / "x.json", retries=1)
