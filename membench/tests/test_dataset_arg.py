"""``--dataset`` 统一解析（run / evaluate 共用）测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from membench import cli, userdata
from membench.data.info import builtin_path
from membench.userdata import resolve_dataset, resolve_dataset_arg


@pytest.fixture()
def isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """隔离用户测试集目录、按需下载目录与默认产物根，避免读到开发机真实文件。"""
    user_dir = tmp_path / "user_dataset"
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    monkeypatch.setenv("MEMBENCH_USER_DATA_DIR", str(user_dir))
    monkeypatch.setenv("MEMBENCH_DATASETS_DIR", str(datasets_dir))
    monkeypatch.setenv("MEMBENCH_OUTPUT_DIR", str(tmp_path / "output"))
    return user_dir, datasets_dir


def test_office_key_falls_back_to_builtin(isolated_dirs: tuple[Path, Path]) -> None:
    resolved = resolve_dataset_arg("office")
    assert resolved.key == "office"
    assert resolved.source == "builtin-office"
    assert Path(resolved.path) == builtin_path("office")


def test_office_key_prefers_installed_user_dataset(
    isolated_dirs: tuple[Path, Path],
) -> None:
    user_dir, _ = isolated_dirs
    user_dir.mkdir(parents=True)
    user_file = user_dir / userdata.USER_DATASET_FILENAME
    user_file.write_text("", encoding="utf-8")
    resolved = resolve_dataset_arg("office")
    assert resolved.source == "user"
    assert resolved.path == str(user_file)


def test_sample_key_uses_builtin_file(isolated_dirs: tuple[Path, Path]) -> None:
    resolved = resolve_dataset_arg("sample")
    assert resolved.key == "sample"
    assert resolved.source == "fetch"
    assert Path(resolved.path) == builtin_path("sample")


def test_existing_jsonl_path_used_as_is(
    isolated_dirs: tuple[Path, Path], tmp_path: Path
) -> None:
    custom = tmp_path / "my_set.jsonl"
    custom.write_text("", encoding="utf-8")
    resolved = resolve_dataset_arg(str(custom))
    assert resolved.key == "my_set"
    assert resolved.path == str(custom.resolve())
    assert resolved.source == "user-specified"


@pytest.mark.parametrize("value", ["missing.jsonl", "/nope/x.jsonl", "~/nope.jsonl"])
def test_missing_path_fails_fast(
    isolated_dirs: tuple[Path, Path], value: str
) -> None:
    """不存在的路径在解析处直接报错，不留给下游抛 FileNotFoundError / schema 错误。"""
    with pytest.raises(SystemExit) as exc_info:
        resolve_dataset_arg(value)
    message = str(exc_info.value)
    assert "数据集文件不存在" in message
    assert "list-datasets" in message


def test_existing_directory_is_not_a_dataset(
    isolated_dirs: tuple[Path, Path], tmp_path: Path
) -> None:
    folder = tmp_path / "folder.jsonl"
    folder.mkdir()
    with pytest.raises(SystemExit, match="不是文件"):
        resolve_dataset_arg(str(folder))


def test_registry_key_not_downloaded_hints_fetch(
    isolated_dirs: tuple[Path, Path],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        resolve_dataset_arg("locomo")
    message = str(exc_info.value)
    assert "dataset fetch locomo" in message
    assert "可直接用" in message and "office" in message


def test_downloaded_registry_key_resolves(isolated_dirs: tuple[Path, Path]) -> None:
    _, datasets_dir = isolated_dirs
    fetched = datasets_dir / "locomo.jsonl"
    fetched.write_text("", encoding="utf-8")
    resolved = resolve_dataset_arg("locomo")
    assert resolved.key == "locomo"
    assert resolved.source == "fetch"
    assert resolved.path == str(fetched)


def test_unknown_value_lists_available_keys(
    isolated_dirs: tuple[Path, Path],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        resolve_dataset_arg("no_such_key")
    message = str(exc_info.value)
    assert "未知数据集" in message
    for key in ("office", "sample", "locomo", "statemembench"):
        assert key in message
    assert "JSONL 路径" in message


def test_resolve_dataset_none_keeps_fallback(
    isolated_dirs: tuple[Path, Path],
) -> None:
    path, source = resolve_dataset(None)
    assert source == "builtin-office"
    assert Path(path) == builtin_path("office")


def test_cli_run_with_office_key(
    isolated_dirs: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """真跑一次：run --dataset office 走内置 office 并打印正确来源。"""
    rc = cli.main(
        [
            "run",
            "--adapter",
            "mock-strong",
            "--dataset",
            "office",
            "--judge",
            "heuristic",
            "--limit",
            "3",
            "--report-out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "(来源: builtin-office)" in out
    assert "office_dataset.jsonl" in out
