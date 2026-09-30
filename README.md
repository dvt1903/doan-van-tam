# AI Studio — Web tích hợp 4 chức năng AI

Dự án bài tập xây dựng website có **4 chức năng AI**, phát triển từ notebook `AI_Web_Apps_Streamlit_React.ipynb` của giảng viên.

## ✅ Nội dung bài nộp

- **Mã nguồn đầy đủ** trên repository GitHub này.
- **README có ảnh giao diện** để giảng viên xem nhanh kết quả.
- **01 slide PowerPoint ngắn gọn về cách làm**: [`docs/AI_Studio_Cach_Lam.pptx`](docs/AI_Studio_Cach_Lam.pptx).

## 4 chức năng AI

| Chức năng | Mô hình / công nghệ |
|---|---|
| 🌼 Nhận diện loài hoa | ResNet-18, ImageNet1K V1 + TF Flowers |
| 🚗 Phát hiện đối tượng | YOLO11n, 80 lớp COCO |
| 🔎 Tìm kiếm ảnh | CLIP ViT-B/32 + FAISS |
| 💬 Chatbot RAG ShopLite | Qwen2.5-0.5B-Instruct + MiniLM |

## Ảnh giao diện

![Tổng quan giao diện 4 chức năng AI](docs/screenshots/ui-overview.jpg)

Ảnh tổng hợp thể hiện giao diện các chức năng nhận diện hoa, phát hiện đối tượng, tìm kiếm ảnh, chatbot RAG và bố cục responsive/mobile.

## Cách làm

```text
React (web/) → FastAPI (api/) → AI models (core/)
                              ├─ ResNet-18
                              ├─ YOLO11n
                              ├─ CLIP + FAISS
                              └─ Qwen2.5 + MiniLM RAG
```

Frontend React gửi ảnh/câu hỏi đến FastAPI. Backend giữ các model AI trong bộ nhớ và gọi module suy luận trong `core/`. Kết quả được trả về để giao diện hiển thị trực tiếp.

## Chạy dự án

### Windows

```bat
start.bat
```

### Linux / macOS

```bash
chmod +x start.sh
./start.sh
```

Sau đó mở:

```text
http://localhost:8000
```

Lần chạy đầu cần Internet để tải dữ liệu và pretrained models. Không cần API key.

## Kết quả thử nghiệm

ResNet-18 đạt **90,19% accuracy trên 367 ảnh test riêng** trong bộ dữ liệu Flowers của project.

## Cấu trúc thư mục

```text
api/             FastAPI backend
core/            module AI
web/             React + Vite frontend
scripts/         chuẩn bị dữ liệu/model
data/kb/         tài liệu RAG ShopLite
docs/            ảnh giao diện + slide nộp bài
requirements.txt thư viện Python
start.bat        chạy nhanh trên Windows
start.sh         chạy trên Linux/macOS
```

## Slide ngắn gọn về cách làm

📎 **PowerPoint:** [`AI_Studio_Cach_Lam.pptx`](docs/AI_Studio_Cach_Lam.pptx)

Nội dung slide: kiến trúc React → FastAPI → AI models, 4 tính năng AI, cách chạy và kết quả chính.

## AI và phiên bản

### AI chạy trong sản phẩm

- ResNet-18 / ImageNet1K V1
- YOLO11n / Ultralytics 8.3.203
- OpenAI CLIP ViT-B/32 + FAISS 1.12.0
- Qwen2.5-0.5B-Instruct
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

### AI hỗ trợ hoàn thiện bài

- **ChatGPT — GPT-5.6 Sol**: hỗ trợ kiểm tra cấu trúc bài nộp, README và tổ chức repository GitHub.

## Môi trường

- Python 3.12
- PyTorch 2.8.0
- torchvision 0.23.0
- transformers 4.57.1
- React + Vite
- FastAPI

> Không commit `.venv`, `node_modules`, cache hoặc trọng số model lớn lên GitHub. Các tài nguyên cần thiết được tải khi chuẩn bị/chạy project.

---

### Checklist trước khi nộp

- [x] Có mã nguồn trên GitHub
- [x] README có ảnh giao diện
- [x] Có slide PowerPoint ngắn gọn về cách làm
- [x] Có hướng dẫn chạy project
- [x] Có ghi AI/model và phiên bản sử dụng
