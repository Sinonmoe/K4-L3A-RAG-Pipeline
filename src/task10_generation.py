"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env.
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.
"""

import os

from dotenv import load_dotenv

from .contracts import validate_generation_result
from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")

SYSTEM_PROMPT = """Trả lời chỉ từ context được cung cấp.
Mỗi khẳng định phải có citation. Nếu thiếu evidence, hãy từ chối xác minh."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context."""
    if len(chunks) <= 2:
        return list(chunks)
    return list(chunks[::2]) + list(reversed(chunks[1::2]))


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label."""
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk["metadata"]
        parts.append(
            f"[Document {index} | Title: {metadata['title']} | "
            f"Source: {metadata['source']}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình."""
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).lower().strip()
    model = os.getenv("LLM_MODEL", LLM_MODEL).strip()

    if provider == "openai":
        from openai import OpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        response = OpenAI().responses.create(
            model=model or "gpt-4.1-mini",
            instructions=system_prompt,
            input=user_message,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.output_text.strip()

    if provider == "gemini":
        from google import genai
        from google.genai import types

        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model=model or "gemini-2.5-flash",
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            ),
        )
        return (response.text or "").strip()

    if provider == "anthropic":
        from anthropic import Anthropic

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        response = Anthropic().messages.create(
            model=model or "claude-3-5-haiku-latest",
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=800,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return "".join(block.text for block in response.content if hasattr(block, "text")).strip()

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult."""
    refusal = {
        "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
        "sources": [],
        "retrieval_source": "none",
    }
    if not query.strip() or top_k <= 0:
        return refusal

    try:
        chunks = retrieve(query, top_k=top_k)
        if not chunks:
            return refusal
        context = format_context(reorder_for_llm(chunks))
        answer = call_llm(
            SYSTEM_PROMPT,
            f"Context:\n{context}\n\nQuestion: {query}\n"
            "Cite sources using [Document N].",
        )
        if not answer:
            return refusal
        result = {
            "answer": answer,
            "sources": chunks,
            "retrieval_source": chunks[0]["retrieval_method"],
        }
        validate_generation_result(result)
        return result
    except Exception:
        return refusal


if __name__ == "__main__":
    print(generate_with_citation("test query"))
