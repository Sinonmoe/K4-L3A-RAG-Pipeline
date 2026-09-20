"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""

import numpy as np
from rank_bm25 import BM25Okapi

try:
    from .contracts import validate_search_results
    from .task4_chunking_indexing import chunk_documents, load_documents
except ImportError:
    from src.contracts import validate_search_results
    from src.task4_chunking_indexing import chunk_documents, load_documents


CORPUS: list[dict] = []


def _get_corpus() -> list[dict]:
    global CORPUS
    if not CORPUS:
        CORPUS = chunk_documents(load_documents())
    return CORPUS


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    if not corpus:
        return None
    tokenized = [item["content"].lower().split() for item in corpus]
    bm25 = BM25Okapi(tokenized)
    # Đảm bảo các từ đều có IDF dương ngay cả với tập ngữ liệu nhỏ (corpus_size <= 2)
    for word in bm25.idf:
        if bm25.idf[word] <= 0:
            bm25.idf[word] = 0.25
    return bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    global CORPUS
    corpus = CORPUS if CORPUS else _get_corpus()
    if not corpus or not query.strip() or top_k <= 0:
        return []

    bm25 = build_bm25_index(corpus)
    if bm25 is None:
        return []

    tokens = query.lower().split()
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)
    indices = np.argsort(scores)[::-1]

    results = []
    seen_ids = set()
    for index in indices:
        if len(results) >= top_k:
            break
        score = float(scores[index])
        if score <= 0:
            continue
        item = corpus[index]
        if item["id"] in seen_ids:
            continue
        seen_ids.add(item["id"])

        meta = dict(item.get("metadata", {}))
        meta.setdefault("url", None)
        results.append({
            "id": item["id"],
            "content": item["content"],
            "score": score,
            "metadata": meta,
            "retrieval_method": "bm25",
        })

    validate_search_results(results, top_k=top_k, expected_method="bm25")
    return results


if __name__ == "__main__":
    for result in lexical_search("writing criteria", top_k=3):
        print(result)
