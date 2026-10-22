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

# 按需下载（dataset fetch）后端：把下载+转换集中到 membench.data.fetch。
from .fetch import (
    FETCH_KEYS,
    dataset_out_path,
    fetch_dataset,
    is_fetchable,
)

# 时间戳归一化：转换器把各原始数据集的时间戳统一成 ISO 8601（精度保持原样）。
from .timestamps import TIMESTAMP_FORMATS_NOTE, normalize_timestamp

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
    "FETCH_KEYS",
    "dataset_out_path",
    "fetch_dataset",
    "is_fetchable",
    "TIMESTAMP_FORMATS_NOTE",
    "normalize_timestamp",
]
