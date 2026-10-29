"""检索命中口径（metrics.evidence）单元测试：substring / overlap / llm。"""

from __future__ import annotations

import pytest

from membench.metrics.evidence import (
    LLMEvidenceMatcher,
    OverlapMatcher,
    SubstringMatcher,
    build_evidence_matcher,
    containment_hit,
    content_tokens,
)


class TestContentTokens:
    def test_chinese_bigram_and_ascii_words(self) -> None:
        tokens = content_tokens("项目截止日是周五 Sprint 12!")
        assert "项目" in tokens and "周五" in tokens
        assert "sprint" in tokens and "12" in tokens

    def test_single_char_run_kept(self) -> None:
        assert "嗯" in content_tokens("嗯")


class TestSubstringMatcher:
    def test_exact_and_wrapped(self) -> None:
        hit, recall = SubstringMatcher().evidence_hits(
            ["项目截止日是周五"], ["记忆：项目截止日是周五（已更新）"]
        )
        assert hit is True and recall == 1.0

    def test_heavy_rewrite_misses(self) -> None:
        hit, recall = SubstringMatcher().evidence_hits(
            ["I went to a LGBTQ support group yesterday"],
            ["Creative outlets: paints and sketches"],
        )
        assert hit is False and recall == 0.0

    def test_empty_evidence(self) -> None:
        assert SubstringMatcher().evidence_hits([], ["x"]) == (False, 0.0)

    def test_top_k_boundary(self) -> None:
        docs = ["无关一", "无关二", "无关三", "无关四", "无关五", "这里有证据句子"]
        hit, recall = SubstringMatcher().evidence_hits(["证据句子"], docs, k=5)
        assert hit is True  # 全量检索里命中
        assert recall == 0.0  # 前 5 条未命中


class TestOverlapMatcher:
    def test_space_separated_near_original_hits(self) -> None:
        """子串 miss、token 包含度命中的近原文改写。"""
        ev = ["项目截止日是周五"]
        docs = ["项目截止日 是 周五"]
        assert SubstringMatcher().evidence_hits(ev, docs) == (False, 0.0)
        hit, recall = OverlapMatcher(threshold=0.6).evidence_hits(ev, docs)
        assert hit is True and recall == 1.0

    def test_heavy_rewrite_misses(self) -> None:
        hit, _ = OverlapMatcher(threshold=0.6).evidence_hits(
            ["I went to a LGBTQ support group yesterday"],
            ["Creative outlets: paints and sketches"],
        )
        assert hit is False

    def test_threshold_respected(self) -> None:
        ev = ["苹果 香蕉 橙子 葡萄"]
        docs = ["苹果 香蕉 橙子 西瓜"]  # 4 个词命中 3 个 → 0.75
        assert OverlapMatcher(threshold=0.8).evidence_hits(ev, docs)[0] is False
        assert OverlapMatcher(threshold=0.7).evidence_hits(ev, docs)[0] is True


class TestLLMEvidenceMatcher:
    def test_semantic_support(self) -> None:
        matcher = LLMEvidenceMatcher(base_url="http://x")
        matcher._chat = lambda evidence, retrieved: [[2], []]  # type: ignore[method-assign]
        hit, recall = matcher.evidence_hits(["e1", "e2"], ["d1", "d2"])
        assert hit is True and recall == pytest.approx(0.5)

    def test_top_k_boundary(self) -> None:
        matcher = LLMEvidenceMatcher(base_url="http://x")
        matcher._chat = lambda evidence, retrieved: [[1], [7]]  # type: ignore[method-assign]
        docs = [f"d{i}" for i in range(1, 8)]
        hit, recall = matcher.evidence_hits(["e1", "e2"], docs, k=5)
        assert hit is True
        assert recall == pytest.approx(0.5)  # 第 7 条超出前 5

    def test_failure_falls_back_to_substring(self) -> None:
        matcher = LLMEvidenceMatcher(base_url="http://x")

        def _boom(evidence: list[str], retrieved: list[str]) -> list[list[int]]:
            raise ConnectionError("model service down")

        matcher._chat = _boom  # type: ignore[method-assign]
        hit, recall = matcher.evidence_hits(["原始句子"], ["前缀 原始句子 后缀"])
        assert hit is True and recall == 1.0
        assert matcher.fallbacks == 1


class TestBuildEvidenceMatcher:
    def test_substring_and_overlap(self) -> None:
        assert build_evidence_matcher("substring").name == "substring"
        matcher = build_evidence_matcher("overlap", threshold=0.5)
        assert matcher.name == "overlap"
        assert matcher.threshold == 0.5

    def test_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="未知 evidence-match"):
            build_evidence_matcher("magic")

    def test_llm_requires_base_url(self, monkeypatch) -> None:
        monkeypatch.delenv("MEMBENCH_JUDGE_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        with pytest.raises(ValueError, match="--llm-base-url"):
            build_evidence_matcher("llm")


class TestLLMFallbacksConcurrency:
    def test_fallbacks_thread_safe(self) -> None:
        """--jobs 并发下回退计数不丢：32 并发同时失败应精确计 32。"""
        import threading

        matcher = LLMEvidenceMatcher(base_url="http://x")
        barrier = threading.Barrier(32)

        def _boom(evidence: list[str], retrieved: list[str]) -> list[list[int]]:
            barrier.wait(timeout=5)
            raise ConnectionError("model service down")

        matcher._chat = _boom  # type: ignore[method-assign]
        threads = [
            threading.Thread(target=matcher.evidence_hits, args=(["e"], ["d"]))
            for _ in range(32)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert matcher.fallbacks == 32


class TestSessionEvidenceContainment:
    """session 级长证据的 containment 兜底（短证据行为不变）。"""

    #: 长证据（> 500 字符），以换行分隔——检索块用空格拼接，普通子串判定必 miss。
    LONG_EV = "\n".join(f"line{i} some session text" for i in range(60))
    #: 取自证据内部连续三行的检索块（去空白后是证据的子串）。
    CHUNK = "line10 some session text line11 some session text line12 some session text"

    def test_long_evidence_chunk_hits(self) -> None:
        assert len(self.LONG_EV) > 500
        assert self.CHUNK not in self.LONG_EV  # 换行差异让严格子串判定 miss
        hit, recall = SubstringMatcher().evidence_hits([self.LONG_EV], [self.CHUNK])
        assert hit is True
        assert recall == 1.0

    def test_short_evidence_unchanged(self) -> None:
        """短证据不走 containment：去空白后才相等的检索条目仍判 miss。"""
        short = "line10 some session text"  # < 500 字符
        retrieved = "line10\nsome session text"
        hit, recall = SubstringMatcher().evidence_hits([short], [retrieved])
        assert hit is False and recall == 0.0

    def test_short_retrieved_item_not_a_container(self) -> None:
        """检索条目去空白后不足 20 字符：不作容器，避免短串伪命中。"""
        long_ev = "\n".join(f"line{i} some session text" for i in range(60))
        hit, _ = SubstringMatcher().evidence_hits([long_ev], ["line10\nsome"])  # 9 字符
        assert hit is False

    def test_containment_applies_to_recall_at_5(self) -> None:
        """同一判定用于 recall@5：容器落在第 6 条时 hit=True / recall=0。"""
        docs = [f"unrelated memory {i} " + "z" * 30 for i in range(5)]
        docs.append(self.CHUNK)
        hit, recall = SubstringMatcher().evidence_hits([self.LONG_EV], docs, k=5)
        assert hit is True
        assert recall == 0.0

    def test_overlap_matcher_containment(self) -> None:
        long_ev = " ".join(f"w{i}" for i in range(200))  # > 500 字符，token 包含度不足 0.6
        chunk = "\n".join(f"w{i}" for i in range(10, 18))  # 去空白后 24 字符
        hit, recall = OverlapMatcher(threshold=0.6).evidence_hits([long_ev], [chunk])
        assert hit is True and recall == 1.0

    def test_containment_hit_direction(self) -> None:
        long_ev = "a" * 600
        # 证据是条目的子串（条目更长）同样算命中
        assert containment_hit(long_ev, ["prefix " + long_ev]) is True
        assert containment_hit(long_ev, ["short"]) is False
