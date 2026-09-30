# BẢN NỘP — SIMPLE NOTEBOOKLM

## Link project

Repository: `dvt1903/doan-van-tam`

Nhánh nộp chính: `main`

## Đối chiếu yêu cầu đề bài

| Yêu cầu | Trạng thái | Mã nguồn |
|---|---|---|
| Nạp PDF, tách nội dung theo trang và chunk | ✅ | `src/indexing.py` |
| Embedding + vector database Qdrant | ✅ | `src/store.py` |
| Retrieval-Augmented Generation (RAG) | ✅ | `src/rag.py` |
| Hỏi đáp dựa trên tài liệu + citation | ✅ | `src/rag.py`, `src/schemas.py` |
| Tóm tắt tài liệu / truy vấn / filter | ✅ | `src/learning.py` |
| Tóm tắt tài liệu dài bằng Map-Reduce | ✅ | `src/learning.py`, `src/prompts/summary_*.jinja2` |
| Tạo Quiz | ✅ | `src/learning.py`, `src/prompts/quiz.jinja2` |
| Tạo Flashcards | ✅ | `src/learning.py`, `src/prompts/flashcards.jinja2` |
| Export JSON / Markdown | ✅ | `src/export.py` |
| REST API FastAPI | ✅ | `src/interfaces/api.py` |
| CLI Typer | ✅ | `src/interfaces/cli.py` |
| Web UI Streamlit | ✅ | `src/interfaces/ui.py` |
| Backend LLM HuggingFace / Gemini / vLLM | ✅ | `src/llm.py`, `src/config.py` |
| Metadata filter | ✅ | `src/filters.py` |
| Đánh giá RAG bằng Ragas | ✅ | `src/evaluation/ragas_evaluator.py` |
| Recursive / Semantic Chunking | ✅ | `src/evaluation/chunking_strategies.py` |
| Reranking bằng Cross-Encoder | ✅ | `src/evaluation/run_reranking.py` |
| Unit tests | ✅ | `tests/` |
| GitHub Actions CI | ✅ | `.github/workflows/ci.yml` |
| Docker / Docker Compose | ✅ | `Dockerfile`, `docker-compose.yml` |
| Chạy không cần API key để demo | ✅ | `mock` LLM + `hash` embedding trong `.env.example` |

## Cách demo nhanh

### Windows

```bat
start.bat
```

### macOS/Linux

```bash
chmod +x start.sh
./start.sh
```

Sau đó mở:

- Streamlit UI: `http://localhost:8501`
- FastAPI Swagger: `http://localhost:8000/docs`

Mặc định project chạy ở chế độ demo offline để giảng viên có thể kiểm tra luồng chương trình mà không cần API key. Có thể chuyển sang Gemini/HuggingFace/vLLM bằng cách sửa `.env` theo hướng dẫn trong `README.md`.

## Kiểm tra tự động

GitHub Actions thực hiện:

```bash
python -m pytest -q
python -m compileall -q src tests
```

Bản `main` đã vượt qua workflow kiểm tra cơ bản.

## Lưu ý trước khi gửi link cho giảng viên

Repository hiện có thể đang để **Private**. Nếu giảng viên không phải collaborator, hãy chuyển repository sang **Public** hoặc thêm tài khoản GitHub của giảng viên làm collaborator trước khi nộp link.
