"""
Task 9 — Retrieval pipeline hoàn chỉnh.

Luồng xử lý:
    1. Chạy semantic_search và lexical_search.
    2. Fuse hai danh sách bằng RRF đúng một lần.
    3. Lấy best cosine score gốc từ dense results.
    4. Nếu score dưới threshold, thử PageIndex fallback.
    5. Nếu fallback lỗi, trả hybrid results thay vì crash.

Không so sánh threshold với RRF score vì hai thang đo khác nhau.
"""

import os
from dotenv import load_dotenv

try:
    from .contracts import validate_search_results
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank_rrf
    from .task8_pageindex_vectorless import pageindex_search
except ImportError:
    from src.contracts import validate_search_results
    from src.task5_semantic_search import semantic_search
    from src.task6_lexical_search import lexical_search
    from src.task7_reranking import rerank_rrf
    from src.task8_pageindex_vectorless import pageindex_search

load_dotenv()

SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.5")) if os.getenv("SCORE_THRESHOLD") else 0.5
DEFAULT_TOP_K = 5


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Trả về hybrid hoặc pageindex SearchResult."""
    if not query.strip() or top_k <= 0:
        return []

    dense = semantic_search(query, top_k=top_k * 2)
    sparse = lexical_search(query, top_k=top_k * 2)

    hybrid = (
        rerank_rrf([dense, sparse], top_k=top_k)
        if use_reranking
        else dense[:top_k]
    )

    best_dense_score = dense[0]["score"] if dense else 0.0
    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                validate_search_results(fallback, top_k=top_k, expected_method="pageindex")
                return fallback[:top_k]
        except Exception:
            # Fallback error must not crash the pipeline; return hybrid result
            pass

    return hybrid[:top_k]


if __name__ == "__main__":
    for result in retrieve("IELTS writing task 2 band descriptors criteria", top_k=3):
        print(result)
