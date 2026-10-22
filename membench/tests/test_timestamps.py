"""时间戳归一化测试：四类原始形态 → ISO 8601，**精度保持原样**。"""

from __future__ import annotations

import pytest

from membench.data.timestamps import normalize_timestamp


class TestKnownFormats:
    """四类原始数据集的实测形态。"""

    @pytest.mark.parametrize(
        ("raw", "want"),
        [
            # LoCoMo：12 小时制散文（含 1 位/2 位小时、12 点边界）
            ("1:56 pm on 8 May, 2023", "2023-05-08T13:56"),
            ("10:37 am on 27 June, 2023", "2023-06-27T10:37"),
            ("12:09 am on 13 September, 2023", "2023-09-13T00:09"),
            ("12:28 pm on 8 November, 2023", "2023-11-08T12:28"),
            # LongMemEval：斜杠 + 星期（可省）
            ("2023/04/10 (Mon) 17:50", "2023-04-10T17:50"),
            ("2023/04/10 17:50", "2023-04-10T17:50"),
            # BEAM：Month-DD-YYYY，只到日
            ("March-01-2024", "2024-03-01"),
            ("April-9-2024", "2024-04-09"),
            # ISO（内置集 / GateMem）：到分 / 到秒 / 只到日；空格分隔规范成 T
            ("2026-04-06T09:00", "2026-04-06T09:00"),
            ("2025-01-05T10:00:00", "2025-01-05T10:00:00"),
            ("2025-01-05 10:00", "2025-01-05T10:00"),
            ("2024-02-01", "2024-02-01"),
        ],
    )
    def test_normalize(self, raw: str, want: str) -> None:
        assert normalize_timestamp(raw) == want

    def test_missing_stays_missing(self) -> None:
        assert normalize_timestamp(None) is None
        assert normalize_timestamp("") is None
        assert normalize_timestamp("   ") is None


class TestPrecisionPreserved:
    """粒度不改变：不补时分秒、不造时区。"""

    def test_date_only_gets_no_time(self) -> None:
        out = normalize_timestamp("March-01-2024")
        assert out == "2024-03-01"
        assert "T" not in out and ":" not in out

    def test_minute_precision_gets_no_seconds(self) -> None:
        out = normalize_timestamp("1:56 pm on 8 May, 2023")
        assert out == "2023-05-08T13:56"
        assert out.count(":") == 1  # 只有一处 HH:MM，没有秒

    def test_second_precision_kept(self) -> None:
        assert normalize_timestamp("2025-01-05T10:00:00") == "2025-01-05T10:00:00"

    def test_no_timezone_fabricated(self) -> None:
        for raw in ("1:56 pm on 8 May, 2023", "2023/04/10 (Mon) 17:50", "March-01-2024"):
            out = normalize_timestamp(raw)
            assert out is not None
            assert not any(ch in out for ch in ("+", "Z", "z"))


class TestRejectsBadInput:
    """不认识的形态或自洽校验失败 → 报错，不静默留混格式。"""

    @pytest.mark.parametrize(
        "raw",
        [
            "yesterday",
            "2024-13-01",              # 月份越界
            "2023/04/10 (Tue) 17:50",  # 星期与日期不符
            "1:56 pm on 8 Foo, 2023",  # 月份名不认识
            "0:00 pm on 8 May, 2023",  # 12 小时制不允许 0 点（应为 12:00 pm）
            "2026-04-06T25:00",        # 时刻越界
            "Feb-30-2024",             # 缩写月份不在支持范围（原始数据用全名）
        ],
    )
    def test_raises(self, raw: str) -> None:
        with pytest.raises(ValueError):
            normalize_timestamp(raw)

    def test_error_mentions_value_and_where(self) -> None:
        with pytest.raises(ValueError) as exc:
            normalize_timestamp("yesterday", where="conv-1 session_3")
        message = str(exc.value)
        assert "yesterday" in message and "conv-1 session_3" in message
