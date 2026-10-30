"""membench 自身版本标识：构建时哈希、运行时读取、报告头部呈现。

安装后的 skill 包没有 ``.git``，也没有构建过程可追溯，所以哈希必须在**打包时**
定稿并写进 ``membench/_build_hash.py``（该文件自身不参与哈希计算）。开发态读到
空的或过期的常量时按当前代码现算，并在报告里如实标注来源——绝不把过期哈希当事实。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from membench import _build_hash, layout
from membench.adapters.base import BaseMemoryAdapter, IngestStats, QueryResult
from membench.judges.base import JudgeResult
from membench.runner import Runner
from membench.version import (
    PACKAGE_HASH_CHARS,
    __version__,
    compute_hash,
    hash_state,
    package_hash,
    runner_version,
    version_display,
)


class _AnswerAdapter(BaseMemoryAdapter):
    name = "version-test"

    def reset(self) -> None:
        return None

    def ingest(self, messages: list[dict]) -> IngestStats:
        return IngestStats()

    def query(self, question: str) -> QueryResult:
        return QueryResult(answer="不知道。", retrieved=[])


class _Judge:
    name = "version-test"

    def grade(self, question: object, gold: str, prediction: str) -> JudgeResult:
        return JudgeResult(
            correct=True, abstained_pred=True, stale=False, score=1.0, judge=self.name
        )


def _dataset(path: Path) -> Path:
    dialogue = {
        "dialogue_id": "d1",
        "sessions": [[{"role": "user", "content": "随便记一条。", "timestamp": None}]],
        "questions": [
            {
                "qid": "q1",
                "question": "记住了什么？",
                "gold_answer": "不知道。",
                "category": "single_hop",
                "evidence": ["随便记一条。"],
            }
        ],
    }
    path.write_text(json.dumps(dialogue, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _write_package(root: Path, files: dict[str, str]) -> Path:
    """造一个最小"包目录"（compute_hash 只认 ``*.py``）。"""
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def test_compute_hash_is_deterministic_and_short(tmp_path: Path) -> None:
    """内容指纹：同一份代码稳定，长度固定，十六进制。"""
    package = _write_package(
        tmp_path / "pkg", {"a.py": "x = 1\n", "sub/b.py": "y = 2\n"}
    )
    first = compute_hash(str(package))
    assert first == compute_hash(str(package))
    assert len(first) == PACKAGE_HASH_CHARS
    assert re.fullmatch(r"[0-9a-f]+", first)
    # 改一行代码就变
    (package / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert compute_hash(str(package)) != first


def test_compute_hash_ignores_build_hash_and_pycache(tmp_path: Path) -> None:
    """构建哈希文件与 ``__pycache__`` 不参与计算（否则自引用 / 受环境噪声影响）。"""
    package = _write_package(tmp_path / "pkg", {"a.py": "x = 1\n"})
    baseline = compute_hash(str(package))
    (package / "_build_hash.py").write_text('BUILD_HASH = "deadbeef"\n', encoding="utf-8")
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "a.cpython-313.pyc").write_bytes(b"\x00\x01")
    assert compute_hash(str(package)) == baseline


def test_package_hash_matches_build_constant_or_is_computed() -> None:
    """仓库内：打包哈希与现算一致时标 build，否则标 computed 并标出过期。"""
    state = hash_state()
    assert state["hash"] == package_hash() == compute_hash()
    built = str(getattr(_build_hash, "BUILD_HASH", "") or "")
    if built == state["hash"]:
        assert state["source"] == "build"
        assert state["stale_build"] is False
    else:
        assert state["source"] == "computed"
        assert state["stale_build"] is bool(built)


def test_runner_version_has_no_git_commit() -> None:
    """版本标识只有语义版本与代码哈希：没提交时无法知道 commit，不记它。"""
    info = runner_version()
    assert set(info) == {"version", "hash", "hash_source", "stale_build"}
    assert info["version"] == __version__
    assert info["hash"] == package_hash()
    assert info["hash_source"] in {"build", "computed"}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            {"version": "0.1.0", "hash": "abc123def456", "hash_source": "build"},
            "0.1.0+abc123def456（构建哈希）",
        ),
        (
            {"version": "0.1.0", "hash": "abc123def456", "hash_source": "computed"},
            "0.1.0+abc123def456（当前代码指纹；未打包）",
        ),
        (
            {
                "version": "0.1.0",
                "hash": "abc123def456",
                "hash_source": "computed",
                "stale_build": True,
            },
            "0.1.0+abc123def456（当前代码指纹；打包哈希已过期，请重新打包 skill）",
        ),
    ],
)
def test_version_display_variants(value: dict, expected: str) -> None:
    """渲染：打包/未打包/打包后改动过三种来源都要如实标注。"""
    assert version_display(value) == expected


def test_version_display_rejects_non_dict() -> None:
    """旧产物没有该字段（或类型不对）时写 n/a，不猜。"""
    assert version_display(None) == "n/a"
    assert version_display("0.1.0") == "n/a"
    assert version_display({}) == "n/a"


def test_run_records_and_renders_membench_version(tmp_path: Path) -> None:
    """run 把版本标识写进 metrics.json，并在报告「评测配置」段显示。"""
    dataset_dir = Path(
        Runner(
            _AnswerAdapter(),
            str(_dataset(tmp_path / "v.jsonl")),
            judge=_Judge(),
            out_dir=str(tmp_path / "exp"),
            trace_state=False,
        ).run()
    )
    report_dir = dataset_dir / layout.DIR_REPORT
    metrics = json.loads((report_dir / layout.FILE_METRICS).read_text(encoding="utf-8"))
    assert metrics["membench_version"] == runner_version()
    report = (report_dir / layout.FILE_REPORT_MD).read_text(encoding="utf-8")
    assert "| membench 版本 |" in report
    assert package_hash() in report
