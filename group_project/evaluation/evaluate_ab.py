"""A/B evaluation with OpenAI models and RAGAS 0.4.3.

Config A: dense-only (OpenAI embeddings + Chroma cosine).
Config B: hybrid = dense top 2k + BM25 top 2k fused by RRF.
Golden set, generator, prompt, evaluator and top_k are identical for both configs.

Run from the repository root:
    python -m group_project.evaluation.evaluate_ab

Requires OPENAI_API_KEY, LLM_MODEL, EMBEDDING_PROVIDER=openai and EMBEDDING_MODEL in .env.
The vector index is built in a temporary directory, so ./chroma_db is not touched.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import ragas
import tiktoken
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

import src.task4_chunking_indexing as task4
from src.task4_chunking_indexing import chunk_documents, embed_chunks, index_to_vectorstore, load_documents
from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank_rrf
from src.task10_generation import (
    SAFE_REFUSAL,
    SYSTEM_PROMPT,
    _citations_match_sources,
    _normalize_citations,
    call_llm,
    format_context,
    reorder_for_llm,
)

GOLDEN_PATH = Path(__file__).with_name("golden_dataset.json")
OUTPUT_PATH = Path(__file__).with_name("ab_results.json")
TOP_K = 5
CONFIGS = ("dense-only", "hybrid-rrf")
CALIBRATION_QUERIES = {
    "in-domain": "IELTS writing task 2 band descriptors criteria",
    "out-of-domain": "công thức nấu phở bò Hà Nội gia truyền",
}
# List prices, USD per 1M tokens (gpt-4o-mini input/output).
PRICE_IN, PRICE_OUT = 0.15, 0.60
INPUT_COLUMNS = {"user_input", "response", "retrieved_contexts", "reference"}

LLM_MODEL = os.environ["LLM_MODEL"]
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
ENCODER = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))


def build_index() -> int:
    """Embed the standardized corpus with OpenAI into a throwaway Chroma dir."""
    task4.CHROMA_DIR = Path(tempfile.mkdtemp(prefix="eval_chroma_"))
    task4._chroma_client = None
    chunks = chunk_documents(load_documents())
    index_to_vectorstore(embed_chunks(chunks))
    return len(chunks)


def retrieve(config: str, query: str) -> list[dict]:
    dense = semantic_search(query, top_k=TOP_K * 2)
    if config == "dense-only":
        return dense[:TOP_K]
    return rerank_rrf([dense, lexical_search(query, top_k=TOP_K * 2)], top_k=TOP_K)


def generate(query: str, chunks: list[dict]) -> tuple[str, int, int]:
    """Same steps as generate_with_citation, applied to already-retrieved chunks."""
    context = format_context(reorder_for_llm(chunks))
    user_message = (
        f"Context:\n{context}\n\n"
        f"Câu hỏi: {query}\n\n"
        "Hãy trả lời ngắn gọn và trích dẫn bằng đúng ID đặt trong dấu ngoặc vuông."
    )
    answer = call_llm(SYSTEM_PROMPT, user_message)
    prompt_tokens = count_tokens(SYSTEM_PROMPT + user_message)
    completion_tokens = count_tokens(answer)
    if answer != SAFE_REFUSAL:
        answer = _normalize_citations(answer, chunks)
        if not _citations_match_sources(answer, chunks):
            answer = SAFE_REFUSAL
    return answer, prompt_tokens, completion_tokens


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main() -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    n_chunks = build_index()
    output = {
        "date": date.today().isoformat(),
        "framework": f"ragas {ragas.__version__}",
        "generator_model": LLM_MODEL,
        "evaluator_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "top_k": TOP_K,
        "golden_size": len(golden),
        "corpus_chunks": n_chunks,
        "corpus_commit": git("log", "-1", "--format=%h", "--", "data/standardized"),
        "code_commit": git("rev-parse", "--short", "HEAD"),
        "calibration_scores": {
            name: semantic_search(query, top_k=1)[0]["score"]
            for name, query in CALIBRATION_QUERIES.items()
        },
        "configs": {},
    }

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=LLM_MODEL, temperature=0))
    evaluator_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBEDDING_MODEL))
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextRecall(),
        LLMContextPrecisionWithReference(),
    ]

    for config in CONFIGS:
        rows, cases = [], []
        retrieval_s = generation_s = 0.0
        tok_in = tok_out = 0
        for case in golden:
            started = time.perf_counter()
            chunks = retrieve(config, case["question"])
            retrieval_s += time.perf_counter() - started

            started = time.perf_counter()
            answer, prompt_tokens, completion_tokens = generate(case["question"], chunks)
            generation_s += time.perf_counter() - started
            tok_in += prompt_tokens
            tok_out += completion_tokens

            rows.append({
                "user_input": case["question"],
                "response": answer,
                "retrieved_contexts": [c["content"] for c in chunks],
                "reference": case["expected_answer"],
            })
            cases.append({
                "question": case["question"],
                "expected_context": case["expected_context"],
                "answer": answer,
                "retrieved_ids": [c["id"] for c in chunks],
                "retrieved_sources": [c["metadata"]["source"] for c in chunks],
            })

        frame = evaluate(
            EvaluationDataset.from_list(rows),
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_emb,
            show_progress=False,
        ).to_pandas()
        score_columns = [c for c in frame.columns if c not in INPUT_COLUMNS]
        for case, (_, row) in zip(cases, frame.iterrows()):
            case["scores"] = {
                column: None if row[column] != row[column] else float(row[column])
                for column in score_columns
            }

        output["configs"][config] = {
            "metrics": {column: float(frame[column].mean()) for column in score_columns},
            "nan_counts": {column: int(frame[column].isna().sum()) for column in score_columns},
            "retrieval_seconds": retrieval_s,
            "generation_seconds": generation_s,
            "generator_tokens": {"input": tok_in, "output": tok_out},
            "generator_cost_usd": tok_in / 1e6 * PRICE_IN + tok_out / 1e6 * PRICE_OUT,
            "cases": cases,
        }
        print(config, json.dumps(output["configs"][config]["metrics"], indent=2))

    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
