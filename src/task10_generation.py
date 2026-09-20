"""Task 10 — Sinh câu trả lời có citation từ kết quả retrieval."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = 1024

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()

SAFE_REFUSAL = "Tôi không thể xác minh thông tin này từ các nguồn hiện có."

SYSTEM_PROMPT = """Bạn là trợ lý hỏi đáp dựa trên tài liệu.
Chỉ trả lời bằng thông tin có trong context được cung cấp.
Mỗi khẳng định thực tế phải kèm citation chứa đúng ID nguồn, ví dụ [chunk-01].
Không tự tạo URL, tên tài liệu hoặc citation. Nếu context không đủ bằng chứng,
hãy dùng đúng câu từ chối an toàn đã được yêu cầu."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đặt các chunk có thứ hạng cao ở đầu và cuối context.

    Retrieval đã sắp xếp chunk theo độ liên quan giảm dần. Cách xen kẽ này
    giữ chunk mạnh nhất ở đầu và chunk mạnh thứ hai ở cuối để giảm hiệu ứng
    "lost in the middle". Hàm chỉ tạo list mới, không sửa list đầu vào.
    """
    if len(chunks) <= 2:
        return list(chunks)

    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có nhãn ID, tiêu đề và nguồn để kiểm chứng citation."""
    parts: list[str] = []
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        chunk_id = str(chunk.get("id", "unknown-source"))
        title = str(metadata.get("title") or "Không có tiêu đề")
        source = str(metadata.get("source") or "Không rõ nguồn")
        url = metadata.get("url")

        labels = [
            f"ID: {chunk_id}",
            f"Title: {title}",
            f"Source: {source}",
        ]
        if url:
            labels.append(f"URL: {url}")

        content = str(chunk.get("content") or "").strip()
        parts.append(f"[{' | '.join(labels)}]\n{content}")

    return "\n\n---\n\n".join(parts)


def _require_setting(name: str, value: str | None) -> str:
    """Trả cấu hình đã trim hoặc báo lỗi rõ ràng khi chưa thiết lập."""
    normalized = (value or "").strip()
    if not normalized:
        raise RuntimeError(f"Thiếu cấu hình {name} trong file .env")
    return normalized


def _join_anthropic_text(blocks: Iterable[object]) -> str:
    """Ghép các text block trong phản hồi Anthropic thành một chuỗi."""
    return "".join(
        str(getattr(block, "text", ""))
        for block in blocks
        if getattr(block, "type", None) == "text"
    ).strip()


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình trong ``.env``."""
    provider = LLM_PROVIDER.strip().lower()
    model = _require_setting("LLM_MODEL", LLM_MODEL)

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(
            api_key=_require_setting("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        text = response.choices[0].message.content or ""

    elif provider == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=_require_setting("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
        )
        response = client.models.generate_content(
            model=model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )
        text = response.text or ""

    elif provider == "anthropic":
        from anthropic import Anthropic

        client = Anthropic(
            api_key=_require_setting(
                "ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY")
            )
        )
        response = client.messages.create(
            model=model,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        text = _join_anthropic_text(response.content)

    else:
        raise ValueError(
            f"LLM_PROVIDER không hợp lệ: {provider!r}. "
            "Chọn openai, gemini hoặc anthropic."
        )

    text = text.strip()
    if not text:
        raise RuntimeError(f"Provider {provider} trả về nội dung rỗng")
    return text


def _safe_refusal() -> dict:
    return {
        "answer": SAFE_REFUSAL,
        "sources": [],
        "retrieval_source": "none",
    }


def _citations_match_sources(answer: str, chunks: list[dict]) -> bool:
    """Bảo đảm answer có citation và mọi citation đều trỏ tới source thật."""
    cited_ids = {match.strip() for match in re.findall(r"\[([^\[\]]+)\]", answer)}
    source_ids = {str(chunk.get("id", "")).strip() for chunk in chunks}
    return bool(cited_ids) and cited_ids <= source_ids


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Chạy retrieval và sinh ``GenerationResult`` có nguồn kiểm chứng được."""
    normalized_query = query.strip() if isinstance(query, str) else ""
    if not normalized_query or not isinstance(top_k, int) or top_k <= 0:
        return _safe_refusal()

    try:
        chunks = retrieve(normalized_query, top_k=top_k)
    except Exception:
        return _safe_refusal()

    if not chunks:
        return _safe_refusal()

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    if not context.strip():
        return _safe_refusal()

    user_message = (
        f"Context:\n{context}\n\n"
        f"Câu hỏi: {normalized_query}\n\n"
        "Hãy trả lời ngắn gọn và trích dẫn bằng đúng ID đặt trong dấu ngoặc vuông."
    )

    try:
        answer = call_llm(SYSTEM_PROMPT, user_message)
    except Exception:
        return _safe_refusal()

    if answer == SAFE_REFUSAL or not _citations_match_sources(answer, chunks):
        return _safe_refusal()

    retrieval_source = (
        "pageindex"
        if all(chunk.get("retrieval_method") == "pageindex" for chunk in chunks)
        else "hybrid"
    )
    return {
        "answer": answer,
        "sources": list(chunks),
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    print(generate_with_citation("test query"))
