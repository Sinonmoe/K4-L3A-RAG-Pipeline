"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

try:
    from .contracts import validate_search_results
except ImportError:
    from src.contracts import validate_search_results

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CACHE_FILE = Path(__file__).parent.parent / "data" / "pageindex_doc_ids.json"


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    if not PAGEINDEX_API_KEY:
        print("PAGEINDEX_API_KEY is not set in .env; skipping upload.")
        return

    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    doc_mapping = {}
    if CACHE_FILE.exists():
        try:
            doc_mapping = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            doc_mapping = {}

    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if path.name.startswith("."):
            continue
        rel_path = path.relative_to(STANDARDIZED_DIR).as_posix()
        if rel_path in doc_mapping:
            continue
        try:
            # Upload document using PageIndex SDK
            response = client.submit_document(file_path=str(path))
            doc_id = response.get("document_id") or response.get("id") or str(response)
            doc_mapping[rel_path] = doc_id
            print(f"Uploaded {rel_path} -> {doc_id}")
        except Exception as err:
            print(f"Failed to upload {rel_path}: {err}")

    CACHE_FILE.write_text(json.dumps(doc_mapping, indent=2), encoding="utf-8")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    if not query.strip() or top_k <= 0 or not PAGEINDEX_API_KEY:
        return []

    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    try:
        response = client.submit_query(query=query)
        # Parse retrieved items from PageIndex response
        raw_results = response.get("results") or response.get("nodes") or []
        results = []
        for index, item in enumerate(raw_results[:top_k]):
            content = item.get("text") or item.get("content") or str(item)
            item_id = item.get("id") or f"pageindex::{index}"
            score = float(item.get("score", 1.0 / (1.0 + index)))
            metadata = {
                "source": item.get("source", "pageindex"),
                "title": item.get("title", "PageIndex Fallback"),
                "doc_type": item.get("doc_type", "legal"),
                "url": item.get("url"),
                "chunk_index": index,
            }
            results.append({
                "id": str(item_id),
                "content": content,
                "score": score,
                "metadata": metadata,
                "retrieval_method": "pageindex",
            })
        if results:
            validate_search_results(results, top_k=top_k, expected_method="pageindex")
        return results
    except Exception as err:
        print(f"PageIndex search error: {err}")
        raise


if __name__ == "__main__":
    results = pageindex_search("IELTS test rules", top_k=3)
    print(results)
