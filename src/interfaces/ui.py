from __future__ import annotations

import json
import httpx
import streamlit as st
from ..config import settings
from .styles import GLOBAL_CSS

_API = settings.api_url.rstrip("/")


def _api(method: str, path: str, **kwargs):
    response = httpx.request(method, f"{_API}{path}", timeout=180, **kwargs)
    if response.is_error:
        detail = response.text
        try: detail = response.json().get("detail", detail)
        except Exception: pass
        raise RuntimeError(detail)
    return response


def _sidebar():
    st.sidebar.header("📚 Kho tài liệu")
    upload = st.sidebar.file_uploader("Tải PDF", type=["pdf"])
    if upload is not None and st.sidebar.button("Nạp & index tài liệu", use_container_width=True):
        with st.spinner("Đang đọc, chia chunk và index..."):
            try:
                r = _api("POST", "/upload", files={"file": (upload.name, upload.getvalue(), "application/pdf")})
                st.sidebar.success(f"Đã index {r.json()['chunks_indexed']} chunks")
                st.rerun()
            except Exception as exc: st.sidebar.error(str(exc))
    try: docs = _api("GET", "/documents").json()
    except Exception:
        docs = []
        st.sidebar.warning("Chưa kết nối được FastAPI backend.")
    filenames = [d["filename"] for d in docs]
    selected = st.sidebar.multiselect("Giới hạn theo tài liệu", filenames)
    page = st.sidebar.number_input("Trang (0 = tất cả)", min_value=0, value=0, step=1)
    if docs:
        st.sidebar.caption("Tài liệu đã index")
        for d in docs: st.sidebar.write(f"• {d['filename']} — {d['pages']} trang / {d['chunks']} chunks")
    return selected, int(page)


def _filters(filenames, page):
    data = {}
    if filenames: data["filenames"] = filenames
    if page > 0 and len(filenames) <= 1: data["page"] = page
    return data or None


def _download_buttons(prefix: str, payload: dict):
    left, right = st.columns(2)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    left.download_button("⬇️ JSON", data=encoded, file_name=f"{prefix}.json", mime="application/json", use_container_width=True)
    right.download_button("⬇️ Markdown", data="```json\n" + encoded + "\n```", file_name=f"{prefix}.md", mime="text/markdown", use_container_width=True)


def _show_sources(citations):
    if not citations: return
    with st.expander("Nguồn tham chiếu"):
        for c in citations:
            st.markdown(f"<div class='source-card'><b>[{c['source_marker']}]</b> {c['filename']} — trang {c['page']}</div>", unsafe_allow_html=True)


def _tab_chat(filenames, page):
    st.subheader("💬 Hỏi đáp trên tài liệu")
    q = st.text_area("Câu hỏi", placeholder="Ví dụ: RAG là gì và hoạt động như thế nào?")
    k = st.slider("Số chunk truy xuất", 1, 15, 5, key="ask_k")
    if st.button("Hỏi", type="primary", use_container_width=True) and q.strip():
        with st.spinner("Đang truy xuất và tạo câu trả lời..."):
            try:
                data = _api("POST", "/ask", json={"question": q, "k": k, "filters": _filters(filenames, page)}).json()
                st.markdown(data["answer"])
                _show_sources(data.get("citations", []))
                with st.expander("Debug: chunks đã truy xuất"): st.json(data.get("chunks", []))
            except Exception as exc: st.error(str(exc))


def _scope_payload(filenames, page, query):
    return {"document": filenames[0] if len(filenames) == 1 and not page else None, "query": query or None, "filters": _filters(filenames, page)}


def _tab_summary(filenames, page):
    st.subheader("📝 Tóm tắt & hệ thống hóa")
    query = st.text_input("Truy vấn trọng tâm (để trống = tóm tắt phạm vi đã chọn)", key="summary_query")
    if st.button("Tạo tóm tắt", type="primary", use_container_width=True):
        with st.spinner("Đang tóm tắt..."):
            try:
                data = _api("POST", "/summarize", json=_scope_payload(filenames, page, query)).json()
                st.markdown(data["summary"])
                st.markdown("#### Ý chính")
                for p in data.get("key_points", []): st.markdown(f"- {p}")
                _show_sources(data.get("citations", [])); _download_buttons("summary", data)
            except Exception as exc: st.error(str(exc))


def _tab_quiz(filenames, page):
    st.subheader("🧠 Tạo Quiz")
    query = st.text_input("Chủ đề/truy vấn", key="quiz_query")
    count = st.slider("Số câu", 1, 20, 5, key="quiz_count")
    if st.button("Tạo Quiz", type="primary", use_container_width=True):
        with st.spinner("Đang tạo câu hỏi..."):
            try:
                payload = _scope_payload(filenames, page, query); payload["count"] = count
                data = _api("POST", "/quiz", json=payload).json()
                for i, item in enumerate(data.get("items", []), 1):
                    st.markdown(f"### Câu {i}. {item['question']}")
                    st.radio("Chọn đáp án", options=list(range(4)), format_func=lambda x, it=item: f"{chr(65+x)}. {it['options'][x]}", key=f"quiz_{i}")
                    with st.expander("Xem đáp án & giải thích"):
                        st.success(f"Đáp án: {chr(65+item['correct_index'])}. {item['options'][item['correct_index']]}")
                        st.write(item["explanation"])
                _show_sources(data.get("citations", [])); _download_buttons("quiz", data)
            except Exception as exc: st.error(str(exc))


def _tab_flashcards(filenames, page):
    st.subheader("🗂️ Flashcards")
    query = st.text_input("Chủ đề/truy vấn", key="flash_query")
    count = st.slider("Số thẻ", 1, 30, 8, key="flash_count")
    if st.button("Tạo Flashcards", type="primary", use_container_width=True):
        with st.spinner("Đang tạo flashcards..."):
            try:
                payload = _scope_payload(filenames, page, query); payload["count"] = count
                data = _api("POST", "/flashcards", json=payload).json()
                for i, card in enumerate(data.get("cards", []), 1):
                    with st.expander(f"Thẻ {i}: {card['front']}"):
                        st.write(card["back"])
                        if card.get("hint"): st.caption(f"Gợi ý: {card['hint']}")
                        st.caption("Nguồn: " + ", ".join(card.get("source_markers", [])))
                _show_sources(data.get("citations", [])); _download_buttons("flashcards", data)
            except Exception as exc: st.error(str(exc))


def run():
    st.set_page_config(page_title="Simple NotebookLM", page_icon="📚", layout="wide")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    st.markdown("<div class='hero'><h1>📚 Simple NotebookLM</h1><p>RAG Learning System — hỏi đáp, tóm tắt, quiz và flashcards dựa trên PDF.</p></div>", unsafe_allow_html=True)
    filenames, page = _sidebar()
    tabs = st.tabs(["Hỏi đáp", "Tóm tắt", "Quiz", "Flashcards"])
    for tab, fn in zip(tabs, [_tab_chat, _tab_summary, _tab_quiz, _tab_flashcards]):
        with tab: fn(filenames, page)

run()
