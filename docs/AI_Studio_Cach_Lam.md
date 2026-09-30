# AI Studio — Cách xây dựng web với 4 chức năng AI

> **Slide tóm tắt cách làm**

**Kiến trúc:** React gọi FastAPI → backend nạp mô hình một lần → giao diện nhận JSON; chatbot trả dữ liệu streaming qua SSE.

| # | Chức năng | AI / Model |
|---|---|---|
| 01 | Nhận diện 5 loài hoa | ResNet-18 |
| 02 | Phát hiện 80 lớp đối tượng | YOLO11n |
| 03 | Tìm kiếm ảnh | CLIP ViT-B/32 + FAISS |
| 04 | Chatbot RAG | Qwen2.5-0.5B + MiniLM |

**Cách chạy:** `start.bat` (Windows) hoặc `start.sh` (Linux/macOS).

**Chuẩn bị:** dữ liệu và mô hình được tải tự động ở lần chạy đầu.

**Kết quả nhận diện hoa:** Accuracy 90,19% trên 367 ảnh test riêng.

**Cấu trúc triển khai:** `core/` chứa AI inference, `api/` chứa FastAPI, `web/` chứa React, `scripts/prepare.py` chuẩn bị model và dữ liệu.