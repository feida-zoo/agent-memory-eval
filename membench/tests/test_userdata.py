"""用户测试集管理（membench.userdata）测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membench.data import SAMPLE_DATASET_PATH
from membench import userdata
from membench.userdata import (
    USER_DATASET_FILENAME,
    delete_user_dataset,
    install_user_dataset,
    resolve_dataset,
    user_dataset_status,
)


@pytest.fixture()
def user_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把用户测试集目录指向 tmp_path，避免污染真实工作目录。"""
    d = tmp_path / "user_dataset"
    monkeypatch.setenv("MEMBENCH_USER_DATA_DIR", str(d))
    return d


def test_install_valid_dataset(user_dir: Path) -> None:
    stats = install_user_dataset(str(SAMPLE_DATASET_PATH))
    dest = user_dir / USER_DATASET_FILENAME
    assert dest.is_file()
    assert stats["path"] == str(dest)
    assert stats["n_dialogues"] > 0
    assert stats["n_questions"] > 0
    assert stats["source"] == str(SAMPLE_DATASET_PATH)

    meta = json.loads((user_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["installed_at"]
    assert meta["source"] == str(SAMPLE_DATASET_PATH)
    assert meta["n_dialogues"] == stats["n_dialogues"]
    assert meta["n_questions"] == stats["n_questions"]


def test_install_invalid_dataset_writes_nothing(user_dir: Path, tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"dialogue_id": "d1", "sessions": []}\n', encoding="utf-8")  # 缺 questions
    with pytest.raises(ValueError):
        install_user_dataset(str(bad))
    assert not (user_dir / USER_DATASET_FILENAME).exists()
    assert not (user_dir / "metadata.json").exists()


def test_install_missing_source(user_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        install_user_dataset(str(tmp_path / "nonexistent.jsonl"))
    assert not (user_dir / USER_DATASET_FILENAME).exists()


def test_status_empty(user_dir: Path) -> None:
    status = user_dataset_status()
    assert status == {"exists": False, "path": None, "metadata": None}


def test_status_after_install(user_dir: Path) -> None:
    install_user_dataset(str(SAMPLE_DATASET_PATH))
    status = user_dataset_status()
    assert status["exists"] is True
    assert status["path"] == str(user_dir / USER_DATASET_FILENAME)
    assert status["metadata"]["n_questions"] > 0


def test_delete(user_dir: Path) -> None:
    assert delete_user_dataset() is False
    install_user_dataset(str(SAMPLE_DATASET_PATH))
    assert delete_user_dataset() is True
    assert not (user_dir / USER_DATASET_FILENAME).exists()
    assert not (user_dir / "metadata.json").exists()
    assert user_dataset_status()["exists"] is False


def test_resolve_dataset_priority(user_dir: Path, tmp_path: Path) -> None:
    explicit = tmp_path / "custom.jsonl"
    explicit.write_text("", encoding="utf-8")
    # 1. 显式指定的已有文件优先（路径规范化为绝对路径）
    path, source = resolve_dataset(str(explicit))
    assert path == str(explicit.resolve())
    assert source == "user-specified"
    # 2. 未安装用户测试集时回退内置 office 数据集
    path, source = resolve_dataset(None)
    assert source == "builtin-office"
    assert path.endswith("office_dataset.jsonl")
    assert Path(path).is_file()
    # 3. 安装用户测试集后优先用户测试集
    install_user_dataset(str(SAMPLE_DATASET_PATH))
    path, source = resolve_dataset(None)
    assert source == "user"
    assert path == str(user_dir / USER_DATASET_FILENAME)
    # 显式指定仍最高优先
    path, source = resolve_dataset(str(explicit))
    assert path == str(explicit.resolve())
    assert source == "user-specified"


def test_resolve_dataset_accepts_registry_key(user_dir: Path) -> None:
    """显式值可以是注册表 key：office 走用户集 / 内置 office，其它 key 走注册表。"""
    path, source = resolve_dataset("office")
    assert source == "builtin-office"
    assert path.endswith("office_dataset.jsonl")

    path, source = resolve_dataset("sample")
    assert source == "fetch"
    assert path.endswith("sample_dataset.jsonl")

    install_user_dataset(str(SAMPLE_DATASET_PATH))
    path, source = resolve_dataset("office")
    assert source == "user"
    assert path == str(user_dir / USER_DATASET_FILENAME)
