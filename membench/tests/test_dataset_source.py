"""数据集来源标签按路径反查（``infer_dataset_source``）。

背景：``dataset fetch`` 的公开基准集一旦以**路径**形式传入（``--dataset
~/.membench/datasets/locomo.jsonl``，或编排器 ``--common`` 先把清单解析成路径），
解析层只看到一个"存在的文件"，默认会记成 ``user-specified``——报告里于是把
公开基准集标成"用户自定义"。这里锁定三种来源的判定与兜底。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from membench import layout
from membench.adapters import MockStrongAdapter
from membench.data import SAMPLE_DATASET_PATH
from membench.data.info import builtin_path, datasets_dir, infer_dataset_source
from membench.runner import Runner
from membench.userdata import resolve_dataset_arg


def test_builtin_dataset_inferred(tmp_path: Path) -> None:
    """内置集：按内置文件名反查，office 记 builtin-office，其余 builtin。"""
    assert infer_dataset_source(SAMPLE_DATASET_PATH) == "builtin"
    assert infer_dataset_source(builtin_path("office")) == "builtin-office"


def test_fetched_dataset_inferred(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``datasets_dir()`` 下与注册表 key 同名的文件记 fetch。"""
    cache = tmp_path / "datasets"
    cache.mkdir()
    target = cache / "locomo.jsonl"
    target.write_text('{"dialogue_id": "d1", "sessions": [], "questions": []}\n', encoding="utf-8")
    monkeypatch.setenv("MEMBENCH_DATASETS_DIR", str(cache))
    assert datasets_dir() == cache
    assert infer_dataset_source(target) == "fetch"


def test_unknown_path_returns_none(tmp_path: Path) -> None:
    """认不出来的路径返回 None，由调用方按 user-specified 兜底。"""
    random = tmp_path / "mydata.jsonl"
    random.write_text("{}\n", encoding="utf-8")
    assert infer_dataset_source(random) is None
    # fetch 目录里但 key 不在注册表：同样不猜
    cache = tmp_path / "cache"
    cache.mkdir()
    other = cache / "whatever.jsonl"
    other.write_text("{}\n", encoding="utf-8")
    assert infer_dataset_source(other) is None


def test_resolve_dataset_arg_keeps_fetch_source_for_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """显式传路径时不再一律 user-specified：基准集按路径反查为 fetch。"""
    cache = tmp_path / "datasets"
    cache.mkdir()
    fetched = cache / "gatemem_office.jsonl"
    fetched.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("MEMBENCH_DATASETS_DIR", str(cache))
    resolved = resolve_dataset_arg(str(fetched))
    assert resolved.source == "fetch"
    assert resolved.key == "gatemem_office"

    plain = tmp_path / "mine.jsonl"
    plain.write_text("{}\n", encoding="utf-8")
    assert resolve_dataset_arg(str(plain)).source == "user-specified"


def test_runner_records_inferred_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Runner 未显式给 dataset_source 时按路径反查（编排器下发路径的常见路径）。"""
    cache = tmp_path / "datasets"
    cache.mkdir()
    copied = cache / "sample.jsonl"
    copied.write_text(SAMPLE_DATASET_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("MEMBENCH_DATASETS_DIR", str(cache))

    dataset_dir = Runner(
        MockStrongAdapter(),
        str(copied),
        judge="heuristic",
        out_dir=str(tmp_path / "exp"),
    ).run()
    assert dataset_dir is not None
    metrics = json.loads(
        (Path(dataset_dir) / layout.DIR_REPORT / layout.FILE_METRICS).read_text(
            encoding="utf-8"
        )
    )
    # sample 是内置集但不是 fetch 条目：放在 fetch 目录里也不算 fetch
    assert metrics["dataset_source"] == "user-specified"
