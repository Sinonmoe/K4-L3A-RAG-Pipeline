"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

import os
from pathlib import Path

from dotenv import load_dotenv

try:
    from .contracts import validate_document
except ImportError:
    from src.contracts import validate_document

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024

COLLECTION_NAME = "rag_documents"

_chroma_client = None
_st_models: dict[str, object] = {}


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed danh sách chuỗi văn bản theo cấu hình EMBEDDING_PROVIDER."""
    if not texts:
        return []

    provider = os.getenv("EMBEDDING_PROVIDER", EMBEDDING_PROVIDER).lower().strip()
    model_name = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL).strip()

    if provider == "sentence_transformers":
        global _st_models
        if model_name not in _st_models:
            from sentence_transformers import SentenceTransformer

            _st_models[model_name] = SentenceTransformer(model_name)
        model = _st_models[model_name]
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [vec.tolist() for vec in embeddings]

    elif provider == "openai":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        client = OpenAI(api_key=api_key)
        openai_model = model_name if "/" not in model_name and model_name else "text-embedding-3-small"
        response = client.embeddings.create(input=texts, model=openai_model)
        return [item.embedding for item in response.data]

    elif provider == "gemini":
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when EMBEDDING_PROVIDER=gemini")
        client = genai.Client(api_key=api_key)
        gemini_model = model_name if "/" not in model_name and model_name else "gemini-embedding-001"
        if "text-embedding-004" in gemini_model:
            gemini_model = "gemini-embedding-001"
        result = client.models.embed_content(model=gemini_model, contents=texts)
        return [e.values for e in result.embeddings]

    else:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    global _chroma_client
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if path.name.startswith("."):
            continue
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in path.parts else "news"

        title = path.stem.replace("_", " ").title()
        url = None
        for line in content.splitlines()[:15]:
            stripped = line.strip()
            if stripped.startswith("# ") and title == path.stem.replace("_", " ").title():
                extracted_title = stripped[2:].strip()
                if extracted_title:
                    title = extracted_title
            elif stripped.startswith("**Source:** "):
                extracted_url = stripped[len("**Source:** "):].strip()
                if extracted_url:
                    url = extracted_url

        doc = {
            "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
            "content": content,
            "metadata": {
                "source": path.name,
                "title": title,
                "doc_type": doc_type,
                "url": url,
            },
        }
        validate_document(doc, require_chunk=False)
        documents.append(doc)

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for document in documents:
        raw_splits = splitter.split_text(document["content"])
        splits = [s.strip() for s in raw_splits if s.strip()]
        if not splits and document["content"].strip():
            splits = [document["content"].strip()]

        for index, text in enumerate(splits):
            chunk_metadata = dict(document.get("metadata", {}))
            chunk_metadata["chunk_index"] = index
            chunk = {
                "id": f"{document['id']}::chunk-{index}",
                "content": text,
                "metadata": chunk_metadata,
            }
            validate_document(chunk, require_chunk=True)
            chunks.append(chunk)

    return chunks


def embed_chunks(chunks: list[dict], batch_size: int = 32) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    if not chunks:
        return []

    texts = [chunk["content"] for chunk in chunks]
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_vectors = embed_texts(batch_texts)
        vectors.extend(batch_vectors)

    embedded_chunks = []
    for chunk, vector in zip(chunks, vectors):
        chunk_copy = dict(chunk)
        chunk_copy["embedding"] = vector
        embedded_chunks.append(chunk_copy)

    return embedded_chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    if not chunks:
        return

    if "embedding" not in chunks[0]:
        chunks = embed_chunks(chunks)

    collection = get_collection()
    batch_size = 200
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=[chunk["metadata"] for chunk in batch],
        )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")
    embedded_chunks = embed_chunks(chunks)
    print(f"Embedded {len(embedded_chunks)} chunks")
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks into collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    run_pipeline()
