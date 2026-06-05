# Chatbot Law VN RAG: Hệ thống tư vấn / pháp luật thông minh

Chatbot Law VN là hệ thống hỏi đáp pháp luật Việt Nam theo kiến trúc RAG local-first. Người dùng tải mã nguồn từ GitHub về máy, build Docker, chạy backend/web, sau đó hỏi đáp trên giao diện chat đơn giản. Hệ thống luôn ưu tiên căn cứ trong kho văn bản nội bộ; Gemini chỉ được dùng như fallback runtime khi kho nội bộ thiếu hoặc không đủ căn cứ.

## 📖 Giới Thiệu

Dự án phục vụ việc tra cứu, hỏi đáp và thử nghiệm chatbot pháp luật Việt Nam trên 5 nhóm dữ liệu chính:

- `CCCD`: căn cước, căn cước công dân.
- `DatDai`: đất đai, cấp giấy chứng nhận, quyền sử dụng đất.
- `DoanhNghiep`: thành lập, quản trị và đăng ký doanh nghiệp.
- `HoTich`: khai sinh, cải chính hộ tịch, cập nhật thông tin hộ tịch.
- `Thue`: mã số thuế, đăng ký thuế, cập nhật thông tin thuế.

Nguồn văn bản đã xử lý nằm trong `data/processed/`. Vector store hiện dùng chiến lược chunk theo cấu trúc pháp lý `legal_article_clause_v1`, gồm 6.277 chunk, embedding model `data/models/vietnamese-embedding-legal`, backend index FAISS.

## ✨ Tính Năng Kỹ Thuật Nổi Bật

- Local RAG ưu tiên tuyệt đối: truy xuất `data/vector_store/` trước khi dùng API ngoài.
- Fine-tuned Vietnamese legal embedding: model local tại `data/models/vietnamese-embedding-legal`.
- Chunk pháp lý theo Điều/Khoản: metadata có `article_id`, `clause_id`, `source_file`, giúp câu trả lời nêu căn cứ rõ hơn.
- Gemini runtime fallback: chỉ bổ sung khi local không có căn cứ hoặc thiếu một văn bản cụ thể.
- API supplement tiết kiệm token: văn bản đã có trong kho nội bộ không bị gửi đi tìm lại qua Gemini.
- SQLite persistence: lưu lịch sử chat, cache fallback/API supplement và danh sách nhận báo cáo.
- OCR/convert tài liệu: chuyển `.doc/.docx/.txt` và hỗ trợ OCR để tạo `data/processed/`.
- Scheduler báo cáo email: tạo Markdown, render PDF và gửi email hằng ngày theo danh mục theo dõi.
- Docker-first: có service CPU, backend web và profile GPU để fine-tune.

## 🏗️ Kiến Trúc Hệ Thống

```text
GitHub repo
  |
  |-- Multi-Agent/                 # Văn bản gốc theo lĩnh vực
  |-- data/processed/              # Văn bản .txt đã xử lý
  |-- data/train/, data/test/      # Bộ câu hỏi/câu trả lời mẫu
  |-- data/finetune/               # JSONL train/test + báo cáo đánh giá
  |-- data/models/                 # Model embedding fine-tuned local
  |-- data/vector_store/           # FAISS index + metadata + manifest
  |-- data/chat_history.sqlite3    # Lịch sử chat/cache/subscriber email
  |-- scripts/                     # Convert, train, evaluate, query CLI
  |-- backend/app.py               # FastAPI, RAG API, scheduler email
  |-- frontend/                    # Giao diện chat web
  |-- Dockerfile
  |-- docker-compose.yml
```

Luồng hỏi đáp:

```text
Người dùng -> Frontend -> FastAPI /chat
  -> Embed câu hỏi
  -> Search FAISS local
  -> Lọc score/hết hiệu lực/metadata
  -> Sinh câu trả lời + căn cứ pháp lý
  -> Nếu thiếu căn cứ: gọi Gemini fallback/supplement
  -> Lưu lịch sử và cache vào SQLite
```

Luồng huấn luyện và cập nhật dữ liệu:

```text
Multi-Agent/ hoặc tài liệu mới
  -> scripts/convert_docs_to_txt.py hoặc scripts/OCR.py
  -> data/processed/
  -> scripts/build_vector_store.py
  -> data/vector_store/
  -> /reload backend
```

## 🧪 Đánh Giá Chất Lượng Kết Quả

Kết quả retrieval hiện tại trên `data/finetune/test_pairs.jsonl`:

| Model | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dangvantuan/vietnamese-embedding` | 0.460 | 0.636 | 0.676 | 0.744 | 0.5599 | 0.6047 |
| `data/models/vietnamese-embedding-legal` | 0.724 | 0.896 | 0.924 | 0.944 | 0.8103 | 0.8437 |

Lệnh đánh giá baseline:

```bash
docker compose run --rm app python scripts/evaluate_retrieval.py \
  --model-name dangvantuan/vietnamese-embedding \
  --test-file data/finetune/test_pairs.jsonl \
  --train-file data/finetune/train_pairs.jsonl \
  --valid-file data/finetune/valid_pairs.jsonl \
  --output-json data/finetune/retrieval_eval_baseline.json
```

Lệnh đánh giá model fine-tuned:

```bash
docker compose run --rm app python scripts/evaluate_retrieval.py \
  --model-name data/models/vietnamese-embedding-legal \
  --test-file data/finetune/test_pairs.jsonl \
  --train-file data/finetune/train_pairs.jsonl \
  --valid-file data/finetune/valid_pairs.jsonl \
  --output-json data/finetune/retrieval_eval_finetuned.json
```

Khi đánh giá chất lượng local RAG, nên tắt Gemini để kết quả khách quan:

```bash
docker compose run --rm app python scripts/query_cli.py \
  --query "Công dân cần đổi thẻ căn cước khi nào?" \
  --domain CCCD \
  --top-k 5 \
  --no-gemini-fallback
```

## 🛠️ Tech Stack

- Python 3.10
- FastAPI + Uvicorn
- Sentence Transformers
- Transformers, Datasets, Accelerate
- PyTorch CPU/GPU
- FAISS
- NumPy
- SQLite
- Google Gemini API qua `google-genai`
- PyMuPDF để render PDF
- Tesseract OCR, PyMuPDF, OpenCV, Pillow
- HTML/CSS/JavaScript frontend tĩnh
- Docker, Docker Compose

## 🚀 Hướng Dẫn Cài Đặt & Triển Khai

### 1. Clone mã nguồn

```bash
git clone <URL_GITHUB_CUA_DU_AN>
cd Chatbot_law_vn
```

### 2. Yêu cầu máy chạy

- Linux/macOS/Windows có Docker.
- Docker Compose plugin: `docker compose`.
- Internet lần đầu để tải image, Python packages và model Hugging Face nếu chưa có cache.
- Nếu fine-tune bằng GPU: NVIDIA driver + NVIDIA Container Toolkit.

Kiểm tra Docker:

```bash
docker --version
docker compose version
```

Kiểm tra GPU nếu có:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 3. Tạo file cấu hình môi trường

```bash
cp .env.example .env
```

Chạy local-only, không dùng Gemini:

```env
GEMINI_API_KEY=
GEMINI_FALLBACK_ENABLED=false
EMBEDDING_MODEL_NAME=data/models/vietnamese-embedding-legal
MIN_RETRIEVAL_SCORE=0.45
CHAT_DB_PATH=data/chat_history.sqlite3
```

Bật Gemini fallback runtime:

```env
GEMINI_API_KEY=your_real_gemini_key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_FALLBACK_ENABLED=true
EMBEDDING_MODEL_NAME=data/models/vietnamese-embedding-legal
MIN_RETRIEVAL_SCORE=0.45
CHAT_DB_PATH=data/chat_history.sqlite3
```

### 4. Build Docker image

```bash
docker compose build app backend
```

Build image GPU để fine-tune:

```bash
docker compose --profile gpu build gpu
```

Nếu vừa sửa dependencies hoặc gặp lỗi version:

```bash
docker compose build --no-cache app backend
docker compose --profile gpu build --no-cache gpu
```

### 5. Chạy web nhanh

Điều kiện tốt nhất là repo đã có sẵn:

- `data/models/vietnamese-embedding-legal/`
- `data/vector_store/manifest.json`
- `data/vector_store/faiss.index`
- `data/vector_store/metadata.jsonl`

Chạy backend + frontend:

```bash
docker compose up -d backend
```

Mở trình duyệt:

```text
http://localhost:8000
```

Kiểm tra API:

```bash
curl http://localhost:8000/health
```

Gửi câu hỏi thử:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Công dân cần đổi thẻ căn cước khi nào?",
    "domain": "CCCD",
    "top_k": 5,
    "gemini_fallback": false
  }'
```

Restart hoặc dừng:

```bash
docker compose restart backend
docker compose down
```

### 6. Chuyển văn bản gốc sang `data/processed`

Nếu repo đã có `data/processed/`, có thể bỏ qua bước này.

Chuyển toàn bộ tài liệu:

```bash
docker compose run --rm app python scripts/convert_docs_to_txt.py --clean-output
```

Chuyển riêng một lĩnh vực:

```bash
docker compose run --rm app python scripts/convert_docs_to_txt.py --domain Thue --overwrite
```

OCR tài liệu scan/ảnh/PDF:

```bash
docker compose run --rm app python scripts/OCR.py
```

Kiểm tra kết quả:

```bash
find data/processed -maxdepth 2 -type f -name "*.txt" | sort
```

### 7. Build lại vector store

```bash
docker compose run --rm app python scripts/build_vector_store.py \
  --embedding-model data/models/vietnamese-embedding-legal \
  --batch-size 64
```

Kiểm tra manifest:

```bash
cat data/vector_store/manifest.json
```

Sau khi backend đang chạy, reload index:

```bash
curl -X POST http://localhost:8000/reload
```

### 8. Fine-tune lại embedding model

Chuẩn bị JSONL từ `data/train/` và `data/test/`:

```bash
docker compose run --rm app python scripts/prepare_qa_finetune_data.py \
  --valid-ratio 0 \
  --seed 20260603
```

Validate dữ liệu:

```bash
docker compose run --rm app python scripts/validate_finetune_data.py \
  --model-name dangvantuan/vietnamese-embedding \
  --max-seq-length 256 \
  --output-json data/finetune/validation_report.json
```

Train GPU:

```bash
docker compose --profile gpu run --rm gpu python scripts/train_embedding.py \
  --model-name dangvantuan/vietnamese-embedding \
  --train-file data/finetune/train_pairs.jsonl \
  --valid-file data/finetune/valid_pairs.jsonl \
  --output-dir data/models/vietnamese-embedding-legal \
  --epochs 3 \
  --batch-size 8 \
  --lr 2e-5 \
  --warmup-ratio 0.1 \
  --max-seq-length 256 \
  --use-amp
```

Nếu CUDA out of memory:

```bash
docker compose --profile gpu run --rm gpu python scripts/train_embedding.py \
  --model-name dangvantuan/vietnamese-embedding \
  --train-file data/finetune/train_pairs.jsonl \
  --valid-file data/finetune/valid_pairs.jsonl \
  --output-dir data/models/vietnamese-embedding-legal \
  --epochs 3 \
  --batch-size 4 \
  --lr 2e-5 \
  --warmup-ratio 0.1 \
  --max-seq-length 128 \
  --use-amp
```

Sau khi train xong, build lại vector store:

```bash
docker compose run --rm app python scripts/build_vector_store.py \
  --embedding-model data/models/vietnamese-embedding-legal \
  --batch-size 64
```

### 9. Query bằng CLI

Local-only:

```bash
docker compose run --rm app python scripts/query_cli.py \
  --query "Công dân cần cấp đổi thẻ căn cước khi nào?" \
  --domain CCCD \
  --top-k 5 \
  --no-gemini-fallback
```

Bật fallback Gemini:

```bash
docker compose run --rm app python scripts/query_cli.py \
  --query "Tư vấn một vấn đề pháp luật mới chưa có trong kho nội bộ" \
  --top-k 5 \
  --gemini-fallback
```

### 10. API backend

Endpoint chính:

```text
GET  /health
GET  /history
POST /chat
POST /reload
GET  /reports/subscribers
POST /reports/subscribers
GET  /reports/preview
POST /reports/send
GET  /  # frontend
```

Payload `/chat`:

```json
{
  "message": "Câu hỏi của người dùng",
  "domain": "CCCD",
  "top_k": 5,
  "min_score": 0.45,
  "gemini_fallback": true
}
```

Nguyên tắc Gemini:

- Backend luôn search local vector store trước.
- Nếu local có căn cứ hợp lệ, response là `mode="local_rag"` và `gemini_used=false`.
- Nếu local thiếu một văn bản cụ thể, response có thể là `mode="local_rag_with_api_supplement"`.
- Nếu local không có căn cứ và bật fallback, response là `mode="gemini_fallback"`.
- Lịch sử và cache được lưu ở `data/chat_history.sqlite3`.

### 11. Báo cáo email tự động

Cấu hình `.env`:

```env
REPORT_SCHEDULER_ENABLED=true
REPORT_DAILY_TIME=07:00
REPORT_TIMEZONE=Asia/Ho_Chi_Minh
REPORTS_DIR=data/reports
REPORT_WATCHLIST_TOPICS=cập nhật thông tin căn cước,cải chính hộ tịch,đăng ký doanh nghiệp,cấp giấy chứng nhận quyền sử dụng đất,thay đổi thông tin đăng ký thuế

SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USERNAME=your_smtp_username
SMTP_PASSWORD=your_smtp_password
SMTP_FROM_EMAIL=chatbot-law-vn@example.com
SMTP_USE_TLS=true
```

Thêm người nhận:

```bash
curl -X POST http://localhost:8000/reports/subscribers \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","name":"User","active":true}'
```

Xem danh sách nhận email:

```bash
curl http://localhost:8000/reports/subscribers
```

Xem trước báo cáo Markdown/PDF:

```bash
curl http://localhost:8000/reports/preview
```

Gửi thử thủ công:

```bash
curl -X POST http://localhost:8000/reports/send \
  -H "Content-Type: application/json" \
  -d '{"force":true}'
```

Khi bật scheduler, backend sẽ tự gửi báo cáo lúc `REPORT_DAILY_TIME` mỗi ngày.

## 🔮 Roadmap

- Bổ sung giao diện quản lý người nhận báo cáo email ngay trên frontend.
- Thêm trang dashboard hiển thị chất lượng retrieval theo từng lĩnh vực.
- Tự động phát hiện văn bản hết hiệu lực và gợi ý văn bản thay thế.
- Bổ sung nhiều lĩnh vực pháp luật hơn như ngân hàng, bảo hiểm, lao động, dân sự.
- Cải thiện OCR cho tài liệu scan chất lượng thấp.
- Thêm reranker để tăng độ chính xác top-1/top-3.
- Thêm trích dẫn theo điều/khoản ổn định hơn cho câu hỏi đa văn bản.
- Tách worker scheduler riêng nếu triển khai nhiều replica backend.
- Thêm cơ chế phân quyền người dùng và quản trị viên.
- Đóng gói model/vector store theo release artifact để người dùng GitHub tải về nhanh hơn.
