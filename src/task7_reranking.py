"""
Task 7 — Reciprocal Rank Fusion.

RRF gộp nhiều bảng xếp hạng mà không cộng trực tiếp cosine score với BM25
score. Công thức: RRF(d) = sum(1 / (k + rank)), rank bắt đầu từ 1.

Lưu ý: RRF score chỉ phản ánh thứ hạng, không dùng để quyết định fallback.

-> Dùng Jina hoặc self host hoặc bất cứ công cụ nào bạn quen
"""

try:
    from .contracts import validate_search_results
except ImportError:
    from src.contracts import validate_search_results


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse nhiều ranked lists và trả hybrid SearchResult."""
    if not ranked_lists or top_k <= 0:
        return []

    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0.0) + (1.0 / (k + rank))
            if item_id not in items:
                items[item_id] = item

    if not scores:
        return []

    ranked_ids = sorted(scores, key=lambda item_id: scores[item_id], reverse=True)
    results = []
    for item_id in ranked_ids[:top_k]:
        item = items[item_id]
        meta = dict(item.get("metadata", {}))
        meta.setdefault("url", None)
        results.append({
            "id": item["id"],
            "content": item["content"],
            "score": float(scores[item_id]),
            "metadata": meta,
            "retrieval_method": "hybrid",
        })

    validate_search_results(results, top_k=top_k, expected_method="hybrid")
    return results


if __name__ == "__main__":
    from src.task5_semantic_search import semantic_search
    from src.task6_lexical_search import lexical_search

    query = "IELTS writing task 2 criteria"
    dense = semantic_search(query, top_k=5)
    sparse = lexical_search(query, top_k=5)
    hybrid = rerank_rrf([dense, sparse], top_k=3)
    for r in hybrid:
        print(r)
