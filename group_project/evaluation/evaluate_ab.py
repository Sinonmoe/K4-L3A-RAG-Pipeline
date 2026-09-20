"""Reproducible offline A/B evaluation for the IELTS Writing corpus."""

import json
import re
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.task4_chunking_indexing import chunk_documents, load_documents
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank_rrf


ROOT = Path(__file__).parents[2]
GOLDEN_PATH = Path(__file__).with_name("golden_dataset.json")
OUTPUT_PATH = Path(__file__).with_name("ab_results.json")
TOP_K = 5


def build_local_dense(corpus: list[dict]):
    """Build a deterministic dense-like TF-IDF character index."""
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    matrix = vectorizer.fit_transform(item["content"] for item in corpus)

    def search(query: str, top_k: int) -> list[dict]:
        scores = cosine_similarity(vectorizer.transform([query]), matrix).ravel()
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "id": corpus[index]["id"],
                "content": corpus[index]["content"],
                "score": float(scores[index]),
                "metadata": corpus[index]["metadata"],
                "retrieval_method": "dense",
            }
            for index in indices
            if scores[index] > 0
        ]

    return search


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def extractive_answer(question: str, contexts: list[dict]) -> str:
    sentences = []
    for item in contexts:
        sentences.extend(
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+|\n+", item["content"])
            if len(part.split()) >= 4
        )
    if not sentences:
        return ""
    matrix = TfidfVectorizer(stop_words="english").fit_transform([question, *sentences])
    scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    return sentences[int(np.argmax(scores))]


def semantic_similarity(left: str, right: str) -> float:
    if not left.strip() or not right.strip():
        return 0.0
    matrix = TfidfVectorizer(stop_words="english").fit_transform([left, right])
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0, 0])


def evaluate_case(case: dict, contexts: list[dict]) -> dict:
    answer = extractive_answer(case["question"], contexts)
    joined_context = " ".join(item["content"] for item in contexts)
    answer_tokens = tokens(answer)
    context_tokens = tokens(joined_context)
    expected_tokens = tokens(case["expected_answer"])
    expected_source = case["expected_context"].split(":", 1)[0].strip()

    faithfulness = (
        len(answer_tokens & context_tokens) / len(answer_tokens) if answer_tokens else 0.0
    )
    recall = (
        len(expected_tokens & context_tokens) / len(expected_tokens)
        if expected_tokens
        else 0.0
    )
    precision = (
        sum(item["metadata"]["source"] == expected_source for item in contexts)
        / len(contexts)
        if contexts
        else 0.0
    )
    return {
        "question": case["question"],
        "answer": answer,
        "sources": [item["metadata"]["source"] for item in contexts],
        "faithfulness": faithfulness,
        "answer_relevance": semantic_similarity(answer, case["expected_answer"]),
        "context_recall": recall,
        "context_precision": precision,
    }


def main() -> None:
    dataset = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    corpus = chunk_documents(load_documents())
    local_dense_search = build_local_dense(corpus)
    output = {
        "method": "offline-proxy-v1-tfidf-char-3-5",
        "top_k": TOP_K,
        "corpus_chunks": len(corpus),
        "configs": {},
    }

    for config in ("dense-only", "hybrid-rrf"):
        started = time.perf_counter()
        cases = []
        for case in dataset:
            dense = local_dense_search(case["question"], top_k=TOP_K * 2)
            if config == "dense-only":
                contexts = dense[:TOP_K]
            else:
                sparse = lexical_search(case["question"], top_k=TOP_K * 2)
                contexts = rerank_rrf([dense, sparse], top_k=TOP_K)
            cases.append(evaluate_case(case, contexts))

        metric_names = (
            "faithfulness",
            "answer_relevance",
            "context_recall",
            "context_precision",
        )
        output["configs"][config] = {
            "metrics": {
                name: float(np.mean([case[name] for case in cases]))
                for name in metric_names
            },
            "elapsed_seconds": time.perf_counter() - started,
            "cases": cases,
        }

    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        name: {"metrics": value["metrics"], "elapsed_seconds": value["elapsed_seconds"]}
        for name, value in output["configs"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
