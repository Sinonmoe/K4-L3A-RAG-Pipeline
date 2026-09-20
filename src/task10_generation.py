"""Task 10 — Sinh câu trả lời có citation từ kết quả retrieval."""

from __future__ import annotations

import os
import re
import logging
from collections.abc import Iterable

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = 1024

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()

SAFE_REFUSAL = "Tôi không thể xác minh thông tin này từ các nguồn hiện có."

SYSTEM_PROMPT = """Bạn là trợ lý hỏi đáp dựa trên tài liệu.
Chỉ trả lời bằng thông tin có trong context được cung cấp.
Mỗi khẳng định thực tế phải kèm citation chứa toàn bộ ID nguồn đúng như context,
bao gồm cả đường dẫn và hậu tố chunk; không được rút gọn ID.
Không tự tạo URL, tên tài liệu hoặc citation. Nếu context không đủ bằng chứng,
hãy dùng đúng câu từ chối an toàn đã được yêu cầu."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đặt các chunk có thứ hạng cao ở đầu và cuối context.

    Retrieval đã sắp xếp chunk theo độ liên quan giảm dần. Cách xen kẽ này
    giữ chunk mạnh nhất ở đầu và chunk mạnh thứ hai ở cuối để giảm hiệu ứng
    "lost in the middle". Hàm chỉ tạo list mới, không sửa list đầu vào.
    """
    logger.debug(
        "Reordering %d chunks for LLM; input_ids=%s",
        len(chunks),
        [chunk.get("id") for chunk in chunks],
    )
    if len(chunks) <= 2:
        reordered = list(chunks)
        logger.debug("Reorder unchanged; output_ids=%s", [c.get("id") for c in reordered])
        return reordered

    front = chunks[::2]
    back = chunks[1::2]
    reordered = front + back[::-1]
    logger.debug("Reorder complete; output_ids=%s", [c.get("id") for c in reordered])
    return reordered


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

    context = "\n\n---\n\n".join(parts)
    logger.info("Formatted context: chunks=%d, characters=%d", len(chunks), len(context))
    return context


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
    logger.info(
        "Calling LLM: provider=%s, model=%s, prompt_characters=%d",
        provider,
        model,
        len(system_prompt) + len(user_message),
    )

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
        chat = client.chats.create(
            model=model,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )
        response = chat.send_message(user_message)
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
    logger.info("LLM response received: characters=%d", len(text))
    logger.info("LLM response preview: %r", text[:500])
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
    valid = bool(cited_ids) and cited_ids <= source_ids
    logger.info(
        "Citation validation: valid=%s, cited_ids=%s, available_source_ids=%s",
        valid,
        sorted(cited_ids),
        sorted(source_ids),
    )
    return valid


def _normalize_citations(answer: str, chunks: list[dict]) -> str:
    """Mở rộng citation rút gọn khi nó khớp duy nhất một source ID.

    Model đôi khi biến ``[path/file.md::chunk-0]`` thành ``[chunk-0]``. Việc
    chuẩn hóa chỉ được thực hiện khi hậu tố đó xác định duy nhất một source;
    citation mơ hồ hoặc không tồn tại được giữ nguyên để validator từ chối.
    """
    source_ids = {str(chunk.get("id", "")).strip() for chunk in chunks}

    def resolve(cited_id: str) -> str | None:
        cited_id = cited_id.strip()
        if cited_id.startswith("ID:"):
            # Model đôi khi chép cả nhãn context: "ID: <id> | Title: ... | Source: ...".
            cited_id = cited_id[len("ID:"):].split(" | ", 1)[0].strip()
        if cited_id in source_ids:
            return cited_id

        suffix = f"::{cited_id}"
        matches = sorted(source_id for source_id in source_ids if source_id.endswith(suffix))
        if len(matches) == 1:
            logger.info("Expanded citation %r to %r", cited_id, matches[0])
            return matches[0]
        return None

    def replace_brackets(match: re.Match[str]) -> str:
        resolved = resolve(match.group(1))
        if resolved is not None:
            return f"[{resolved}]"
        return match.group(0)

    normalized = re.sub(r"\[([^\[\]]+)\]", replace_brackets, answer)

    def replace_parentheses(match: re.Match[str]) -> str:
        resolved = resolve(match.group(1))
        if resolved is not None:
            logger.info("Normalized parenthesized citation %r", match.group(1).strip())
            return f"[{resolved}]"
        return match.group(0)

    return re.sub(r"\(([^()]+)\)", replace_parentheses, normalized)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Chạy retrieval và sinh ``GenerationResult`` có nguồn kiểm chứng được."""
    normalized_query = query.strip() if isinstance(query, str) else ""
    logger.info(
        "Generation request: query=%r, top_k=%r",
        normalized_query[:200],
        top_k,
    )
    if not normalized_query or not isinstance(top_k, int) or top_k <= 0:
        logger.warning("Safe refusal: query rỗng hoặc top_k không hợp lệ")
        return _safe_refusal()

    try:
        logger.info("Starting retrieval")
        chunks = retrieve(normalized_query, top_k=top_k)
    except Exception:
        logger.exception("Safe refusal: retrieval raised an exception")
        return _safe_refusal()

    if not chunks:
        logger.warning("Safe refusal: retrieval returned no chunks")
        return _safe_refusal()

    logger.info(
        "Retrieval completed: count=%d, methods=%s, results=%s",
        len(chunks),
        sorted({str(chunk.get("retrieval_method")) for chunk in chunks}),
        [
            {
                "id": chunk.get("id"),
                "score": round(float(chunk.get("score", 0.0)), 6),
            }
            for chunk in chunks
        ],
    )

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    if not context.strip():
        logger.warning("Safe refusal: formatted context is empty")
        return _safe_refusal()

    user_message = (
        f"Context:\n{context}\n\n"
        f"Câu hỏi: {normalized_query}\n\n"
        "Hãy trả lời ngắn gọn và trích dẫn bằng đúng ID đặt trong dấu ngoặc vuông."
    )

    try:
        answer = call_llm(SYSTEM_PROMPT, user_message)
    except Exception:
        logger.exception("Safe refusal: LLM provider raised an exception")
        return _safe_refusal()

    if answer == SAFE_REFUSAL:
        logger.warning("Safe refusal: LLM returned the refusal message")
        return _safe_refusal()
    answer = _normalize_citations(answer, chunks)
    if not _citations_match_sources(answer, chunks):
        logger.warning("Safe refusal: answer has no valid source citation")
        return _safe_refusal()

    retrieval_source = (
        "pageindex"
        if all(chunk.get("retrieval_method") == "pageindex" for chunk in chunks)
        else "hybrid"
    )
    logger.info(
        "Generation completed: retrieval_source=%s, sources=%d, answer_characters=%d",
        retrieval_source,
        len(chunks),
        len(answer),
    )
    return {
        "answer": answer,
        "sources": list(chunks),
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    print(generate_with_citation("test query"))
