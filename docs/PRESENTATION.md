# Slide ngắn — Simple NotebookLM

## Slide 1 — Bài toán và mục tiêu

**Simple NotebookLM — Học từ tài liệu PDF bằng RAG**

- Người dùng upload tài liệu PDF.
- Hệ thống chia tài liệu thành các đoạn nhỏ và tạo embedding.
- Khi đặt câu hỏi, hệ thống truy xuất các đoạn liên quan nhất.
- LLM chỉ trả lời dựa trên ngữ cảnh truy xuất để giảm hallucination.

**4 chức năng chính:**
1. Hỏi đáp theo tài liệu.
2. Tóm tắt tài liệu.
3. Tạo Quiz.
4. Tạo Flashcards.

### Lời nói ngắn
> Nhóm em xây dựng một phiên bản NotebookLM đơn giản. Điểm chính của hệ thống là không để AI trả lời hoàn toàn theo kiến thức có sẵn, mà truy xuất nội dung từ PDF trước rồi mới tạo câu trả lời.

---

## Slide 2 — Cách làm và công nghệ

**Pipeline:** PDF → Extract text → Chunking → Embedding → Semantic Search → Top-k Context → Gemini → Kết quả có nguồn.

**Công nghệ:**
- Streamlit: giao diện web.
- PyMuPDF: đọc PDF.
- Sentence Transformers: embedding.
- Cosine Similarity: semantic retrieval.
- Gemini 2.5 Flash: sinh câu trả lời, tóm tắt, quiz, flashcards.

### Lời nói ngắn
> Sau khi upload PDF, chương trình đọc từng trang, chia thành chunk rồi vector hóa. Khi có câu hỏi, hệ thống tìm những chunk gần nhất về ngữ nghĩa và đưa chúng vào prompt cho Gemini. Nhờ đó câu trả lời bám sát tài liệu hơn.

---

## Slide 3 — Demo và kết quả

**Demo 4 tính năng:**
- Hỏi một câu về nội dung PDF và xem nguồn tham khảo.
- Tạo bản tóm tắt.
- Sinh câu hỏi trắc nghiệm.
- Sinh flashcards ôn tập.

**Điểm nổi bật:**
- Có trích dẫn tên file và số trang trong phần retrieval.
- Có Offline Demo Mode khi chưa cấu hình API key.
- Code gọn, dễ chạy bằng một lệnh: `streamlit run app.py`.

### Lời nói ngắn
> Khi demo, nhóm em chỉ cần upload tài liệu rồi bấm Index. Sau đó có thể dùng đủ bốn chức năng AI trên cùng một giao diện. Hệ thống cũng hiển thị đoạn nguồn được truy xuất để người dùng kiểm tra lại nội dung.
