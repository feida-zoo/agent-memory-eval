"""数据集介绍功能（membench.data.info + CLI dataset 命令组）测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membench import cli
from membench.data import fetch as dataset_fetch
from membench.data import info as dataset_info
from membench.data.info import (
    DATASET_REGISTRY,
    DatasetInfo,
    describe,
    describe_all,
    get_dataset_info,
    list_datasets,
)

#: 注册表必须包含的 9 个 key。
EXPECTED_KEYS = {
    "office",
    "sample",
    "locomo",
    "longmemeval_oracle",
    "longmemeval_s",
    "beam_1m",
    "beam_10m",
    "gatemem_office",
    "statemembench",
}


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------


def test_registry_has_all_nine_entries() -> None:
    assert set(DATASET_REGISTRY) == EXPECTED_KEYS
    assert len(list_datasets()) == 9


def test_registry_required_fields_non_empty() -> None:
    for key, entry in DATASET_REGISTRY.items():
        assert isinstance(entry, DatasetInfo)
        assert entry.key == key
        for field_name in ("name", "status", "source", "scale", "features", "baseline", "usage"):
            value = getattr(entry, field_name)
            assert isinstance(value, str) and value.strip(), f"{key}.{field_name} 为空"
        assert entry.status in ("builtin", "fetch", "unavailable"), f"{key}.status 非法"
        if entry.status == "builtin":
            assert entry.builtin_file, f"{key} 为 builtin 但缺少 builtin_file"
        else:
            assert entry.builtin_file is None, f"{key} 非 builtin 不应有 builtin_file"


# ---------------------------------------------------------------------------
# describe / describe_all / get_dataset_info
# ---------------------------------------------------------------------------


def test_describe_locomo_contains_source_arxiv_baseline() -> None:
    text = describe("locomo")
    assert "LoCoMo" in text
    assert "arXiv:2402.17753" in text
    # 基准线关键词
    assert "87.9" in text
    assert "32.1" in text
    assert "基准线" in text
    # 必备栏目
    for section in ("来源", "规模", "特点", "接入状态", "用法"):
        assert section in text


def test_describe_builtin_existing_file_includes_actual_stats() -> None:
    text = describe("office")
    assert "实际统计" in text
    assert "对话" in text and "题" in text


def test_describe_all_covers_all_keys() -> None:
    text = describe_all()
    for key in EXPECTED_KEYS:
        assert key in text
    # 每个条目都带 status 标签与基准线要点
    assert text.count("基准线要点") == len(EXPECTED_KEYS)


def test_get_dataset_info_unknown_key_raises_with_available() -> None:
    with pytest.raises(KeyError) as exc_info:
        get_dataset_info("no_such_dataset")
    message = str(exc_info.value)
    for key in EXPECTED_KEYS:
        assert key in message


# ---------------------------------------------------------------------------
# CLI: dataset info
# ---------------------------------------------------------------------------


def test_cli_dataset_info_all(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["dataset", "info"]) == 0
    out = capsys.readouterr().out
    for key in EXPECTED_KEYS:
        assert key in out


def test_cli_dataset_info_single(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["dataset", "info", "locomo"]) == 0
    out = capsys.readouterr().out
    assert "arXiv:2402.17753" in out


def test_cli_dataset_info_unknown_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["dataset", "info", "no_such_dataset"]) == 2
    err = capsys.readouterr().err
    assert "office" in err  # 提示可用 key


# ---------------------------------------------------------------------------
# CLI: dataset fetch（不做真实网络请求）
# ---------------------------------------------------------------------------


def test_cli_fetch_builtin_no_network(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("builtin 数据集不应触发下载")

    monkeypatch.setattr(dataset_fetch, "_download_file", _boom)
    assert cli.main(["dataset", "fetch", "office"]) == 0
    out = capsys.readouterr().out
    assert "内置" in out
    assert "office_dataset.jsonl" in out


def test_cli_fetch_unavailable_returns_2(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("unavailable 数据集不应触发下载")

    monkeypatch.setattr(dataset_fetch, "_download_file", _boom)
    assert cli.main(["dataset", "fetch", "statemembench"]) == 2
    err = capsys.readouterr().err
    assert "暂不可用" in err


def test_cli_fetch_unknown_name_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["dataset", "fetch", "no_such_dataset"]) == 2


def test_cli_fetch_beam_with_mocked_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """mock 下载：写入最小合法 BEAM 原始文件，走真实 convert_beam_split 转换。"""

    def _fake_download(
        url: str, dest: Path, retries: int = 3, force: bool = False
    ) -> None:
        if url.endswith("chat.json"):
            payload = [
                {
                    "batch_number": 1,
                    "time_anchor": "2024-01-01",
                    "turns": [[{"role": "user", "content": "你好"}]],
                }
            ]
        else:
            payload = {"information_extraction": [{"question": "q?", "answer": "a"}]}
        dest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(dataset_fetch, "_download_file", _fake_download)
    # 缩小下载规模，避免写 35 份文件
    monkeypatch.setitem(dataset_fetch._BEAM_CONFIG, "beam_1m", ("1M", 2, "1m"))

    assert cli.main(["dataset", "fetch", "beam_1m", "--out", str(tmp_path)]) == 0
    out_path = tmp_path / "beam_1m.jsonl"
    assert out_path.is_file()
    out = capsys.readouterr().out
    assert str(out_path) in out
    assert "--dataset" in out
    # 转换结果可被 schema 加载
    from membench.data.schema import load_dataset

    dialogues = load_dataset(out_path)
    assert len(dialogues) == 2
