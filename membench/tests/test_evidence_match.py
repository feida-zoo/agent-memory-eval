"""检索命中口径（metrics.evidence）单元测试：substring / overlap / llm。"""

from __future__ import annotations

import pytest

from membench.metrics.evidence import (
    LLMEvidenceMatcher,
    OverlapMatcher,
    SubstringMatcher,
    build_evidence_matcher,
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
