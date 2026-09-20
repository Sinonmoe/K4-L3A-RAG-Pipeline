"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.
"""

try:
    from .contracts import validate_search_results
    from .task4_chunking_indexing import embed_texts, get_collection
except ImportError:
    from src.contracts import validate_search_results
    from src.task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    if not query.strip() or top_k <= 0:
        return []

    collection = get_collection()
    query_vector = embed_texts([query])[0]

    try:
        count = collection.count() if callable(getattr(collection, "count", None)) else top_k
        actual_top_k = min(top_k, count) if count > 0 else top_k
    except Exception:
        actual_top_k = top_k

    if actual_top_k <= 0:
        return []

    response = collection.query(
        query_embeddings=[query_vector],
        n_results=actual_top_k,
        include=["documents", "metadatas", "distances"],
    )

    if not response or not response.get("ids") or not response["ids"][0]:
        return []

    results = []
    seen_ids = set()
    for item_id, content, metadata, distance in zip(
        response["ids"][0],
        response["documents"][0],
        response["metadatas"][0],
        response["distances"][0],
    ):
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)

        meta = dict(metadata) if metadata else {}
        meta.setdefault("url", None)
        score = max(0.0, 1.0 - float(distance))
        results.append({
            "id": item_id,
            "content": content,
            "score": score,
            "metadata": meta,
            "retrieval_method": "dense",
        })

    sorted_results = sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]
    validate_search_results(sorted_results, top_k=top_k, expected_method="dense")
    return sorted_results


if __name__ == "__main__":
    for result in semantic_search("IELTS writing task 2 criteria", top_k=3):
        print(result)
