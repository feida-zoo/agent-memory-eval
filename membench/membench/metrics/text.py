"""文本质量与检索质量指标（纯标准库实现）。

- ``token_f1(gold, prediction)``：token 多集 F1（SQuAD 风格）。
- ``bleu1(reference, hypothesis)``：unigram 精确率 + brevity penalty。
- ``recall_at_k(evidence, retrieved, k)``：证据子串在前 k 条检索结果中的命中率。

分词复用 adapters.mock 的轻量策略（中文 bigram + 英文单词），保证全链路一致。
"""

from __future__ import annotations

import math
from collections import Counter

from membench.adapters.mock import tokenize

__all__ = ["tokenize", "token_f1", "bleu1", "recall_at_k"]


def token_f1(gold: str, prediction: str) -> float:
    """token 多集 F1。

    Args:
        gold: 标准答案。
        prediction: 系统回答。

    Returns:
        [0, 1] 的 F1 分数；两者皆空返回 1.0，仅一方为空返回 0.0。
    """
    gold_tokens = tokenize(gold or "")
    pred_tokens = tokenize(prediction or "")
    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0
    common = Counter(gold_tokens) & Counter(pred_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def bleu1(reference: str, hypothesis: str) -> float:
    """简化 BLEU-1：unigram 截断精确率 × brevity penalty。

    Args:
        reference: 参考答案。
        hypothesis: 系统回答。

    Returns:
        [0, 1] 的 BLEU-1 分数；hypothesis 为空返回 0.0。
    """
    ref_tokens = tokenize(reference or "")
    hyp_tokens = tokenize(hypothesis or "")
    if not hyp_tokens:
        return 0.0
    if not ref_tokens:
        return 0.0
    ref_counts = Counter(ref_tokens)
    clipped = sum((Counter(hyp_tokens) & ref_counts).values())
    precision = clipped / len(hyp_tokens)
    bp = 1.0 if len(hyp_tokens) > len(ref_tokens) else math.exp(
        1.0 - len(ref_tokens) / len(hyp_tokens)
    )
    return bp * precision


def recall_at_k(evidence: list[str], retrieved: list[str], k: int = 5) -> float:
    """证据召回率：evidence 中有多少比例的子串命中前 k 条检索结果。

    命中判定：证据字符串（原样）是某条检索结果的子串。

    Args:
        evidence: 支持证据列表（对话原文子串）。
        retrieved: 检索命中的记忆条目原文列表。
        k: 只看前 k 条检索结果。

    Returns:
        [0, 1] 的召回率；evidence 为空时返回 0.0（调用方通常对
        abstention 题不调用本函数）。
    """
    if not evidence:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for ev in evidence if any(ev in doc for doc in top))
    return hits / len(evidence)
