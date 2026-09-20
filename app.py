import streamlit as st
from dotenv import load_dotenv
from src.task10_generation import generate_with_citation


load_dotenv()

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("RAG Chatbot")
    st.caption("Trợ lý tra cứu tài liệu IELTS Writing")
    top_k = st.slider("Số chunks", 3, 10, 5)

st.title("RAG Chatbot")
st.caption("Đặt câu hỏi về tiêu chí chấm điểm, cấu trúc và cách viết IELTS Writing.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Nguồn tham khảo"):
                for source in message["sources"]:
                    metadata = source["metadata"]
                    st.markdown(
                        f"- **{metadata['title']}** — `{metadata['source']}` "
                        f"(score: {source['score']:.3f}, {source['retrieval_method']})"
                    )

query = st.chat_input("Nhập câu hỏi...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        result = generate_with_citation(query, top_k)
        answer = result["answer"]
        sources = result["sources"]
        st.markdown(answer)

        if sources:
            with st.expander("Nguồn tham khảo"):
                for source in sources:
                    metadata = source["metadata"]
                    st.markdown(
                        f"- **{metadata['title']}** — `{metadata['source']}` "
                        f"(score: {source['score']:.3f}, {source['retrieval_method']})"
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
