"""数据模块：Schema、加载器与内置示例数据集路径。"""

from __future__ import annotations

from pathlib import Path

from .schema import (
    ABSTAIN_SENTINEL,
    CATEGORIES,
    Dataset,
    Dialogue,
    Question,
    load_dataset,
)

#: 内置示例数据集（sample_dataset.jsonl）的绝对路径。
SAMPLE_DATASET_PATH = Path(__file__).resolve().parent / "sample_dataset.jsonl"

__all__ = [
    "ABSTAIN_SENTINEL",
    "CATEGORIES",
    "Dataset",
    "Dialogue",
    "Question",
    "load_dataset",
    "SAMPLE_DATASET_PATH",
]
