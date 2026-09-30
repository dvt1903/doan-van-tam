# 📚 Simple NotebookLM — RAG Learning System

Project mô phỏng một **NotebookLM đơn giản**: người dùng tải PDF, hệ thống chia tài liệu thành chunk, vector hóa và lưu vào **Qdrant**, sau đó dùng kiến trúc **Retrieval-Augmented Generation (RAG)** để trả lời câu hỏi dựa trên tài liệu. Ngoài Q&A, project có **Tóm tắt**, **Quiz**, **Flashcards**, REST API, CLI, Streamlit UI và phần đánh giá RAG bằng **Ragas**.

## Chức năng hoàn chỉnh

- ✅ Upload PDF + index tự động vào Qdrant
- ✅ Semantic retrieval + metadata filter theo tài liệu/trang
- ✅ Hỏi đáp có trích dẫn `[S1]`, `[S2]` và nguồn trang
- ✅ Tóm tắt toàn tài liệu / theo filter / theo truy vấn; hỗ trợ map-reduce
- ✅ Tạo Quiz 4 lựa chọn, đáp án, giải thích và nguồn
- ✅ Tạo Flashcards có front/back/hint/topic/source
- ✅ Export JSON / Markdown
- ✅ FastAPI: `/health`, `/documents`, `/upload`, `/ask`, `/debug-retrieval`, `/summarize`, `/quiz`, `/flashcards`
- ✅ Typer CLI
- ✅ Streamlit UI 4 tab
- ✅ Ragas: faithfulness, answer relevancy, context precision, context recall
- ✅ Thực nghiệm Recursive/Semantic Chunking
- ✅ Reranking bằng `BAAI/bge-reranker-v2-m3`
- ✅ Docker / Docker Compose
- ✅ Offline demo mode (`mock` LLM + `hash` embeddings) để chạy không cần API key

## Kiến trúc

```text
PDF -> PyPDFLoader -> Chunking -> Embedding -> Qdrant
                                      |
User -> Query -> Retrieval -----------+
                 -> Prompt -> LLM -> Answer + Citations
                           -> Summary / Quiz / Flashcards
```

## Cấu trúc thư mục

```text
.
├── data/
├── src/
│   ├── prompts/
│   ├── interfaces/
│   │   ├── api.py
│   │   ├── cli.py
│   │   ├── ui.py
│   │   └── styles.py
│   ├── evaluation/
│   │   ├── benchmark_rag.csv
│   │   ├── chunking_strategies.py
│   │   ├── ragas_evaluator.py
│   │   ├── run_chunking.py
│   │   └── run_reranking.py
│   ├── config.py
│   ├── schemas.py
│   ├── indexing.py
│   ├── store.py
│   ├── rag.py
│   ├── learning.py
│   ├── llm.py
│   ├── filters.py
│   └── export.py
├── storage/qdrant/
├── tests/
├── .env.example
├── requirements.txt
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```

## Chạy nhanh nhất

Sau khi cài dependencies (`pip install -r requirements.txt`), có thể chạy trực tiếp:

**Windows**
```bat
start.bat
```

**macOS/Linux**
```bash
chmod +x start.sh
./start.sh
```

Script sẽ tạo `.env` từ `.env.example` nếu chưa có, rồi mở FastAPI và Streamlit.

## Chạy nhanh trên Windows / macOS / Linux

### 1) Tạo môi trường

```bash
python -m venv .venv
```

Windows:
```powershell
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

### 2) Cài thư viện

```bash
pip install -r requirements.txt
```

### 3) Cấu hình

```bash
copy .env.example .env
```

Trên macOS/Linux dùng `cp .env.example .env`.

Mặc định `.env.example` dùng **offline demo mode**:

```env
RAG_LLM_PROVIDER=mock
RAG_EMBEDDING_PROVIDER=hash
```

Để dùng AI thật bằng Gemini:

```env
RAG_LLM_PROVIDER=gemini
RAG_EMBEDDING_PROVIDER=hf
GOOGLE_API_KEY=YOUR_KEY
```

### 4) Chạy backend

```bash
uvicorn src.interfaces.api:app --reload --port 8000
```

Mở API docs: `http://localhost:8000/docs`

### 5) Chạy giao diện

Mở terminal thứ hai:

```bash
streamlit run src/interfaces/ui.py
```

Mở: `http://localhost:8501`

### 6) Sử dụng

1. Ở sidebar, upload một file PDF.
2. Bấm **Nạp & index tài liệu**.
3. Chọn một hoặc nhiều tài liệu.
4. Dùng 4 tab: **Hỏi đáp**, **Tóm tắt**, **Quiz**, **Flashcards**.

## CLI

```bash
python -m src.interfaces.cli ingest --recreate
python -m src.interfaces.cli ask "RAG là gì?"
python -m src.interfaces.cli debug-retrieval "RAG là gì?" --k 5
python -m src.interfaces.cli summarize --document ten-file.pdf --fmt md --output outputs/summary.md
python -m src.interfaces.cli quiz --document ten-file.pdf --count 5 --fmt json
python -m src.interfaces.cli flashcards --document ten-file.pdf --count 10 --fmt md
```

## REST API ví dụ

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"RAG là gì?","k":5}'
```

## Chạy bằng Docker

```bash
cp .env.example .env
docker compose up --build
```

- API: `http://localhost:8000/docs`
- UI: `http://localhost:8501`

## Đánh giá RAG

### Chunking

```bash
python -m src.evaluation.run_chunking \
  --benchmark src/evaluation/benchmark_rag.csv \
  --judge-provider gemini
```

Các cấu hình Recursive: `500/50`, `800/100`, `1000/150`, `1500/200`.
Semantic Chunking: percentile, standard deviation, interquartile.

### Reranking

```bash
python -m src.evaluation.run_reranking \
  --judge-provider gemini \
  --initial-k 15 \
  --rerank-k 5
```

## Trạng thái kiểm tra mã nguồn

- `python -m pytest -q` → **3 passed**
- `python -m compileall -q src tests` → **passed**
- Lưu ý: môi trường thực thi cần cài `requirements.txt`; lần đầu dùng embedding/model Hugging Face cần Internet để tải model.

## Kiểm thử

```bash
pytest -q
python -m compileall -q src tests
```

## Lưu ý khi nộp bài

- Không commit file `.env` hoặc API key.
- Không commit dữ liệu vector trong `storage/qdrant/`.
- Có thể thêm PDF mẫu vào `data/` khi demo tại máy local; repo mặc định bỏ qua PDF để tránh đẩy tài liệu nặng.
- Nếu repo đang để **Private**, hãy chuyển sang **Public** hoặc mời giảng viên làm collaborator trước khi nộp link.

## Công nghệ

Python 3.11+, FastAPI, Streamlit, Qdrant, LangChain, Hugging Face, Gemini, vLLM/OpenAI-compatible API, Jinja2, Pydantic, Typer, Sentence Transformers, Ragas.

## AI assistance disclosure

Project này được hoàn thiện với sự hỗ trợ của **ChatGPT — GPT-5.6 Sol** trong việc triển khai, rà soát cấu trúc và tài liệu hóa mã nguồn.
