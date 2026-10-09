"""Retrieval metrics."""

from __future__ import annotations

import math


def precision_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for c in top if c in relevant) / len(top)


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return sum(1 for c in top if c in relevant) / len(relevant)


def mrr(relevant: set[str], retrieved: list[str]) -> float:
    for i, c in enumerate(retrieved, start=1):
        if c in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 1) for i, c in enumerate(retrieved[:k], start=1) if c in relevant)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0
