"""Reciprocal Rank Fusion (RRF) for merging retrieval channel results."""
from raglib.models.retrieval import RetrievalResult


def reciprocal_rank_fusion(
    *ranked_lists: list[RetrievalResult],
    weights: list[float] | None = None,
    k: int = 60,
) -> list[RetrievalResult]:
    """
    Merge multiple ranked lists using RRF.

    score(d) = Σ weight_i / (k + rank_i)  for all lists where d appears

    Items appearing in multiple channels get boosted naturally.
    k=60 is the standard constant that prevents top-ranked items from dominating.
    """
    n = len(ranked_lists)
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n:
        raise ValueError("weights length must match number of ranked lists")

    scores: dict[str, float] = {}
    result_map: dict[str, RetrievalResult] = {}

    for ranked, weight in zip(ranked_lists, weights):
        for rank, result in enumerate(ranked):
            # Use first 120 chars of content as dedup key (same as original)
            key = f"{result.source[:1]}:{result.content[:120]}"
            rrf_score = weight / (k + rank + 1)
            scores[key] = scores.get(key, 0.0) + rrf_score
            if key not in result_map:
                result_map[key] = result

    # Apply fused scores
    for key, result in result_map.items():
        result_map[key] = result.model_copy(update={"score": scores[key]})

    return sorted(result_map.values(), key=lambda x: x.score, reverse=True)
