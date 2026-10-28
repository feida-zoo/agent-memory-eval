"""``gold_stale`` 解析（``membench.data.stale``）单元测试。

重点：旧缓存的 ``gold_stale`` 仍是整句描述时，判定侧要能抽出裸旧值；
抽不出时标记为不可机检（不进 ``stale_rate`` 分母），而不是恒判不陈旧。
"""

from __future__ import annotations

import pytest

from membench.data.stale import extract_stale_value, looks_like_prose, resolve_stale

#: 用户机器上旧缓存 BEAM 数据的典型形态：整句 potential_confusion 写进 gold_stale。
_USER_PROSE = (
    "LLM might incorrectly recall the original 4 study sessions "
    "instead of the updated 5 study sessions."
)

#: 只提旧值、没有新旧对照的描述句：抽不出裸旧值，应判不可机检。
_USER_PROSE_NO_CONTRAST = (
    "LLM might incorrectly recall the original March 15, 2024 deadline"
)


class TestResolveStale:
    """``resolve_stale`` 的四种分支。"""

    def test_bare_value_returned_as_is(self) -> None:
        # 裸旧值原样返回（不归一、不改写），并标记可机检。
        assert resolve_stale("4 study sessions", "5 study sessions") == (
            "4 study sessions",
            True,
        )
        assert resolve_stale("9月20日", "9月28日") == ("9月20日", True)

    def test_prose_value_extracted(self) -> None:
        # 整句描述 → 抽出裸旧值，仍可机检。
        assert resolve_stale(_USER_PROSE, "5 study sessions") == (
            "4 study sessions",
            True,
        )

    def test_prose_without_extractable_value(self) -> None:
        # 抽不出 → (None, False)：不可机检，不进入分母。
        assert resolve_stale(_USER_PROSE_NO_CONTRAST, "March 18, 2024") == (
            None,
            False,
        )

    def test_prose_extraction_equal_to_gold_is_uncheckable(self) -> None:
        # 抽出的"旧值"与 gold 相同 → 不是旧值，按解析失败处理。
        assert resolve_stale(
            "original 5 study sessions instead of 5 study sessions",
            "5 study sessions",
        ) == (None, False)

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_empty(self, value: str | None) -> None:
        assert resolve_stale(value, "5 study sessions") == (None, False)

    def test_does_not_mutate_extraction_behavior(self) -> None:
        # 解析口径与转换器共用：同一句抽出同一个值。
        assert extract_stale_value(_USER_PROSE, "5 study sessions") == (
            "4 study sessions"
        )


class TestLooksLikeProse:
    """``looks_like_prose`` 的判定边界。"""

    @pytest.mark.parametrize(
        "value",
        [
            # 含对比连词
            _USER_PROSE,
            "The user previously lived in Beijing, but now lives in Hangzhou.",
            "4 sessions rather than 5 sessions",
            # 以标注者口吻 / 时间副词开头（无对比连词也能识别）
            _USER_PROSE_NO_CONTRAST,
            "previously 4 study sessions now 5 study sessions",
            "originally 4 study sessions; now 5 study sessions",
            # 超长：多半是整句而非值
            "x" * 81,
        ],
    )
    def test_prose(self, value: str) -> None:
        assert looks_like_prose(value) is True

    @pytest.mark.parametrize(
        "value",
        ["4 study sessions", "9月20日", "Redis集群主从切换抖动", "15%", "", None],
    )
    def test_bare(self, value: str | None) -> None:
        assert looks_like_prose(value) is False
