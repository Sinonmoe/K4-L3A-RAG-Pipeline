"""Giao diện Streamlit cho chatbot RAG."""

from __future__ import annotations

from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation


load_dotenv()

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🔎",
    layout="wide",
)


def _is_public_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def render_sources(sources: list[dict], retrieval_source: str, *, key_prefix: str) -> None:
    """Hiển thị đúng các SearchResult đã được đưa vào generation."""
    if not sources:
        st.caption("Không có nguồn được sử dụng cho câu trả lời này.")
        return

    st.caption(f"Đường truy xuất: {retrieval_source}")
    with st.expander(f"Nguồn tham khảo ({len(sources)})"):
        for index, source in enumerate(sources, start=1):
            metadata = source.get("metadata") or {}
            title = metadata.get("title") or metadata.get("source") or source.get("id")
            score = float(source.get("score", 0.0))
            method = source.get("retrieval_method", "unknown")

            st.markdown(f"**{index}. {title}**")
            st.code(str(source.get("id", "unknown-source")), language=None)
            st.caption(
                f"Tệp nguồn: {metadata.get('source', 'Không rõ')} · "
                f"Phương thức: {method} · Điểm: {score:.4f}"
            )

            url = metadata.get("url")
            if _is_public_url(url):
                st.link_button(
                    "Mở nguồn",
                    url,
                    key=f"{key_prefix}-source-{index}",
                )
            if index < len(sources):
                st.divider()


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("RAG Chatbot")
    st.caption("Hỏi đáp từ corpus của nhóm, có citation kiểm chứng được.")
    top_k = st.slider("Số chunks truy xuất", 3, 10, 5)
    if st.button("Xóa lịch sử", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("RAG Chatbot")
st.caption("Câu trả lời chỉ sử dụng bằng chứng từ tài liệu đã thu thập.")

for message_index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(
                message.get("sources", []),
                message.get("retrieval_source", "none"),
                key_prefix=f"history-{message_index}",
            )

query = st.chat_input("Nhập câu hỏi...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm bằng chứng và tạo câu trả lời..."):
            result = generate_with_citation(query, top_k)

        st.markdown(result["answer"])
        render_sources(
            result["sources"],
            result["retrieval_source"],
            key_prefix=f"current-{len(st.session_state.messages)}",
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "retrieval_source": result["retrieval_source"],
        }
    )
