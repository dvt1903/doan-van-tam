# AI Studio — Web tích hợp 4 chức năng AI

Dự án bài tập xây dựng một website có **4 chức năng AI**, phát triển từ notebook `AI_Web_Apps_Streamlit_React.ipynb` của giảng viên.

## 4 chức năng

| Chức năng | Mô hình |
|---|---|
| 🌼 Nhận diện loài hoa | ResNet-18, ImageNet1K V1 + TF Flowers |
| 🚗 Phát hiện đối tượng | YOLO11n, 80 lớp COCO |
| 🔎 Tìm kiếm ảnh | CLIP ViT-B/32 + FAISS |
| 💬 Chatbot RAG ShopLite | Qwen2.5-0.5B-Instruct + MiniLM |

## Ảnh giao diện

![Tổng quan giao diện 4 chức năng AI](docs/screenshots/ui-overview.jpg)

Ảnh tổng hợp gồm giao diện nhận diện hoa, phát hiện đối tượng, tìm kiếm ảnh, chatbot RAG và giao diện mobile.

## Cách làm

```text
React (web/) → FastAPI (api/) → AI models (core/)
                              ├─ ResNet-18
                              ├─ YOLO11n
                              ├─ CLIP + FAISS
                              └─ Qwen2.5 + MiniLM RAG
```

Backend nạp mô hình một lần. Giao diện gửi ảnh/câu hỏi đến FastAPI; API trả JSON cho các tác vụ ảnh và chatbot trả kết quả từ pipeline RAG.

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

Sau đó mở `http://localhost:8000`.

Lần chạy đầu cần Internet để tải dữ liệu và pretrained models. Không cần API key.

## Kết quả

ResNet-18 đạt **90,19% accuracy trên 367 ảnh test riêng**.

## Cấu trúc

```text
api/             FastAPI backend
core/            4 module AI
web/             React + Vite frontend
scripts/         chuẩn bị dữ liệu/model
data/kb/         tài liệu RAG ShopLite
docs/            ảnh giao diện và slide
```

## Slide ngắn gọn về cách làm

📎 [Xem slide tóm tắt](docs/AI_Studio_Cach_Lam.md)

## AI và phiên bản

- ResNet-18 / ImageNet1K V1
- YOLO11n / Ultralytics 8.3.203
- OpenAI CLIP ViT-B/32 + FAISS 1.12.0
- Qwen2.5-0.5B-Instruct + paraphrase-multilingual-MiniLM-L12-v2
- Python 3.12, PyTorch 2.8.0, torchvision 0.23.0, transformers 4.57.1

> Không commit `.venv`, `node_modules`, cache, dữ liệu tải về hoặc trọng số model lớn lên GitHub.