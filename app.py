import os
import re
import json
from dataclasses import dataclass
from typing import List

import fitz
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

st.set_page_config(page_title="Simple NotebookLM", page_icon="📚", layout="wide")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()


@dataclass
class Chunk:
    text: str
    filename: str
    page: int


@st.cache_resource
def load_embedder():
    return SentenceTransformer(MODEL_NAME)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or " ").strip()
    return text


def extract_pdf(uploaded_file) -> List[Chunk]:
    chunks = []
    doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
    for page_no, page in enumerate(doc, start=1):
        text = clean_text(page.get_text("text"))
        if not text:
            continue
        words = text.split()
        chunk_size, overlap = 220, 40
        start = 0
        while start < len(words):
            piece = " ".join(words[start:start + chunk_size])
            if piece:
                chunks.append(Chunk(piece, uploaded_file.name, page_no))
            if start + chunk_size >= len(words):
                break
            start += chunk_size - overlap
    return chunks


def build_index(chunks: List[Chunk]):
    model = load_embedder()
    texts = [c.text for c in chunks]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors)


def retrieve(query: str, chunks: List[Chunk], vectors, k=5):
    if not query.strip() or not chunks:
        return []
    model = load_embedder()
    q = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    scores = cosine_similarity(q, vectors)[0]
    ids = np.argsort(scores)[::-1][: min(k, len(chunks))]
    return [(chunks[i], float(scores[i])) for i in ids]


def gemini_generate(prompt: str):
    if not GOOGLE_API_KEY:
        return None
    try:
        from google import genai
        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return (response.text or "").strip()
    except Exception as exc:
        st.warning(f"Gemini tạm thời không dùng được, chuyển sang Offline Demo Mode: {exc}")
        return None


def context_text(items):
    return "\n\n".join(
        f"[Nguồn: {c.filename} - trang {c.page}]\n{c.text}" for c, _ in items
    )


def offline_answer(question, items):
    if not items:
        return "Chưa có dữ liệu phù hợp để trả lời."
    snippets = [c.text for c, _ in items[:3]]
    return (
        "Dựa trên các đoạn liên quan nhất trong tài liệu, nội dung chính có thể tóm lược như sau:\n\n"
        + "\n\n".join(f"• {s[:650]}" for s in snippets)
        + "\n\nBạn có thể bật Gemini bằng GOOGLE_API_KEY để có câu trả lời diễn đạt tự nhiên và tổng hợp tốt hơn."
    )


def answer_question(question, items):
    prompt = f"""Bạn là trợ lý học tập dùng RAG. Chỉ trả lời dựa trên ngữ cảnh bên dưới. Nếu không đủ dữ kiện, hãy nói rõ. Trả lời bằng tiếng Việt, ngắn gọn nhưng đầy đủ, và ghi nguồn theo dạng [tên file - trang X].\n\nCÂU HỎI:\n{question}\n\nNGỮ CẢNH:\n{context_text(items)}"""
    return gemini_generate(prompt) or offline_answer(question, items)


def summarize_docs(chunks):
    source = "\n\n".join(c.text for c in chunks[:18])
    prompt = f"""Hãy tóm tắt tài liệu sau bằng tiếng Việt. Gồm: (1) tóm tắt 1 đoạn, (2) 5-8 ý chính, (3) 3 khái niệm quan trọng. Chỉ dùng nội dung tài liệu.\n\n{source}"""
    result = gemini_generate(prompt)
    if result:
        return result
    sentences = re.split(r"(?<=[.!?])\s+", source)
    key = [s.strip() for s in sentences if len(s.strip()) > 60][:8]
    return "## Tóm tắt nhanh\n\n" + "\n".join(f"- {x[:350]}" for x in key)


def make_quiz(chunks, count=5):
    source = "\n\n".join(c.text for c in chunks[:14])
    prompt = f"""Tạo {count} câu hỏi trắc nghiệm tiếng Việt từ nội dung sau. Mỗi câu có 4 đáp án A/B/C/D, ghi rõ đáp án đúng và giải thích 1 câu. Không dùng kiến thức ngoài tài liệu.\n\n{source}"""
    result = gemini_generate(prompt)
    if result:
        return result
    words = [w.strip(".,:;()[]") for w in source.split() if len(w) > 8]
    unique = list(dict.fromkeys(words))[: count]
    out = []
    for i, word in enumerate(unique, 1):
        out.append(f"**Câu {i}.** Thuật ngữ nào xuất hiện trong tài liệu?\n\nA. {word}\nB. Không có trong tài liệu\nC. Nội dung ngoài phạm vi\nD. Không xác định\n\n**Đáp án: A**")
    return "\n\n---\n\n".join(out) if out else "Không đủ dữ liệu để tạo quiz."


def make_flashcards(chunks, count=8):
    source = "\n\n".join(c.text for c in chunks[:12])
    prompt = f"""Tạo {count} flashcards tiếng Việt từ nội dung sau. Mỗi thẻ theo định dạng:\nQ: ...\nA: ...\nCâu trả lời ngắn, đúng trọng tâm và chỉ dựa trên tài liệu.\n\n{source}"""
    result = gemini_generate(prompt)
    if result:
        return result
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", source) if len(s.strip()) > 70][:count]
    return "\n\n".join(f"**Q{i}: Ý chính cần nhớ là gì?**\n\nA: {s[:300]}" for i, s in enumerate(sentences, 1))


def init_state():
    st.session_state.setdefault("chunks", [])
    st.session_state.setdefault("vectors", None)
    st.session_state.setdefault("indexed_files", [])


init_state()

st.markdown("# 📚 Simple NotebookLM")
st.caption("Học từ tài liệu PDF bằng Retrieval-Augmented Generation (RAG)")

with st.sidebar:
    st.header("📄 Tài liệu")
    files = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)
    if st.button("⚡ Index documents", type="primary", use_container_width=True):
        if not files:
            st.error("Hãy upload ít nhất một file PDF.")
        else:
            with st.spinner("Đang đọc PDF, chia chunk và tạo embedding..."):
                all_chunks = []
                for file in files:
                    all_chunks.extend(extract_pdf(file))
                if all_chunks:
                    st.session_state.chunks = all_chunks
                    st.session_state.vectors = build_index(all_chunks)
                    st.session_state.indexed_files = [f.name for f in files]
                    st.success(f"Đã index {len(all_chunks)} chunks từ {len(files)} file.")
                else:
                    st.error("Không trích xuất được văn bản từ PDF.")

    st.divider()
    mode = "Gemini 2.5 Flash" if GOOGLE_API_KEY else "Offline Demo Mode"
    st.write(f"**AI mode:** {mode}")
    if st.session_state.indexed_files:
        st.write("**Đã index:**")
        for name in st.session_state.indexed_files:
            st.write(f"- {name}")

if not st.session_state.chunks:
    st.info("👈 Upload PDF ở thanh bên trái và bấm **Index documents** để bắt đầu.")
    st.markdown("### 4 chức năng AI")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💬", "Hỏi đáp")
    c2.metric("📝", "Tóm tắt")
    c3.metric("🧠", "Quiz")
    c4.metric("🗂️", "Flashcards")
    st.stop()

qa_tab, summary_tab, quiz_tab, flash_tab = st.tabs(["💬 Hỏi đáp", "📝 Tóm tắt", "🧠 Quiz", "🗂️ Flashcards"])

with qa_tab:
    st.subheader("Hỏi đáp theo tài liệu")
    question = st.text_input("Nhập câu hỏi", placeholder="Ví dụ: RAG là gì và tại sao giúp giảm hallucination?")
    top_k = st.slider("Số đoạn truy xuất", 2, 8, 5)
    if st.button("Trả lời", key="ask"):
        if not question.strip():
            st.warning("Hãy nhập câu hỏi.")
        else:
            items = retrieve(question, st.session_state.chunks, st.session_state.vectors, top_k)
            with st.spinner("Đang truy xuất và tạo câu trả lời..."):
                st.markdown(answer_question(question, items))
            st.markdown("### Nguồn tham khảo")
            for chunk, score in items:
                with st.expander(f"{chunk.filename} — trang {chunk.page} — similarity {score:.3f}"):
                    st.write(chunk.text)

with summary_tab:
    st.subheader("Tóm tắt tài liệu")
    if st.button("Tạo bản tóm tắt", key="summary"):
        with st.spinner("Đang tóm tắt..."):
            st.markdown(summarize_docs(st.session_state.chunks))

with quiz_tab:
    st.subheader("Tạo Quiz")
    q_count = st.slider("Số câu hỏi", 3, 10, 5)
    if st.button("Tạo Quiz", key="quiz"):
        with st.spinner("Đang tạo câu hỏi..."):
            st.markdown(make_quiz(st.session_state.chunks, q_count))

with flash_tab:
    st.subheader("Tạo Flashcards")
    f_count = st.slider("Số flashcards", 4, 15, 8)
    if st.button("Tạo Flashcards", key="flash"):
        with st.spinner("Đang tạo flashcards..."):
            st.markdown(make_flashcards(st.session_state.chunks, f_count))

st.divider()
st.caption("Simple NotebookLM • Streamlit • RAG • Sentence Transformers • Gemini")
