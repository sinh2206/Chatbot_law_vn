# Chatbot Law VN

Hệ thống chatbot pháp luật đa agent (RAG + fallback web) kèm API, dashboard đánh giá và test runner tự động.

## 1. Cấu trúc dự án (đã gộp module)

```text
chatbot-law/
│
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
│
├── config.py
├── api.py
├── chatbot_core.py
├── vector_store.py
├── expiry.py
├── test_runner.py
├── dashboard.py
├── test.py
│
├── Multi-Agent/              # data văn bản pháp luật theo domain
├── metadata/
├── vector_store/
├── logs/
├── reports/
├── frontend/                 # chat UI tĩnh
├── dashboard/                # dashboard static (served qua API)
└── testsuite/
```

## 2. Chức năng từng file chính

`config.py`
- Cấu hình môi trường, model, chunk, đường dẫn thư mục, SMTP, CORS, test settings.

`vector_store.py`
- Đọc file `.txt/.doc/.docx`.
- Chunk văn bản và gắn metadata pháp lý.
- Tạo embedding (Gemini hoặc sentence-transformers).
- Lưu/truy vấn Chroma vector DB.
- CLI build/update:
  - `python vector_store.py --build`
  - `python vector_store.py --build --domain Thue`
  - `python vector_store.py --update --domain DatDai`

`expiry.py`
- Quản lý metadata hiệu lực văn bản (`metadata/legal_documents_metadata.csv`).
- Quét văn bản hết hiệu lực.
- Monitor định kỳ:
  - `python expiry.py --monitor`
  - `python expiry.py --watch`

`chatbot_core.py`
- Intent Agent (phân loại lĩnh vực, tách câu hỏi con).
- Domain RAG Agent (truy vấn vector store theo domain).
- Fallback Web Search khi dữ liệu local không hợp lệ/hết hiệu lực.
- Merge nhiều câu trả lời agent.
- Orchestrator điều phối toàn luồng.

`api.py`
- FastAPI endpoint chat và metrics.
- Gộp telemetry store (SQLite) và admin notifier (log + SMTP).
- Serve frontend tại `/` và dashboard static tại `/dashboard`.

`test_runner.py`
- Gộp bootstrap test set, chạy test suite, xuất report, scheduler định kỳ.
- Chế độ:
  - `--mode smoke|monitoring|full`
  - `--bootstrap --bootstrap-size 50`
  - `--schedule hourly|daily|weekly`
  - `--send-report-email`

`dashboard.py`
- Streamlit dashboard đánh giá test run.
- KPI, bảng chi tiết, biểu đồ, manual review, tải báo cáo.

`test.py`
- Unit test cơ bản cho parser/chunk/metrics breakdown.

## 3. Chuẩn bị môi trường local

1. Tạo virtual environment và cài thư viện:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Tạo file `.env` từ `.env.example`:

```bash
copy .env.example .env
```

3. Điền tối thiểu:
- `GEMINI_API_KEY`
- `EMBEDDING_BACKEND` (`gemini` hoặc `sentence-transformers`)
- SMTP nếu muốn gửi email admin/report.

## 4. Chuẩn bị dữ liệu pháp lý

1. Đặt văn bản vào `Multi-Agent/<Domain>/` với 5 domain:
- `DoanhNghiep`
- `HoTich`
- `CCCD`
- `DatDai`
- `Thue`

2. Tạo template metadata hiệu lực:

```bash
python expiry.py --bootstrap
```

3. Cập nhật file `metadata/legal_documents_metadata.csv`:
- `document_number`
- `issue_date`
- `effective_date`
- `expiry_date`
- `status_override` (`active|expired|`)
- `replacement_document`
- `source_url`

## 5. Build vector store

Build toàn bộ:

```bash
python vector_store.py --build
```

Build một lĩnh vực:

```bash
python vector_store.py --build --domain Thue
```

Update incremental một lĩnh vực:

```bash
python vector_store.py --update --domain DatDai
```

## 6. Chạy chatbot API + frontend

Chạy API:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Truy cập:
- Chat UI: `http://localhost:8000/`
- Dashboard static: `http://localhost:8000/dashboard`
- Health: `http://localhost:8000/health`

Endpoint chat:
- `POST /chat`
- `POST /ask`

Request mẫu:

```json
{
  "question": "Hồ sơ đăng ký doanh nghiệp gồm những gì?",
  "domain": "DoanhNghiep",
  "run_type": "user"
}
```

## 7. Monitor văn bản hết hiệu lực

Quét một lần:

```bash
python expiry.py --monitor
```

Chạy vòng lặp định kỳ:

```bash
python expiry.py --watch --interval-hours 24
```

Khi fallback hoặc phát hiện văn bản hết hiệu lực:
- ghi log `logs/admin_alerts.log`
- gửi email nếu đã cấu hình SMTP.

## 8. Chạy test runner

Bootstrap test set 50 câu:

```bash
python test_runner.py --bootstrap --bootstrap-size 50
```

Smoke test:

```bash
python test_runner.py --mode smoke --runner-mode api
```

Monitoring test:

```bash
python test_runner.py --mode monitoring --monitoring-size 20 --repeats 2 --runner-mode api
```

Full test + LLM grader:

```bash
python test_runner.py --mode full --runner-mode api --llm-grader
```

Gửi report email:

```bash
python test_runner.py --mode full --send-report-email
```

Scheduler:

```bash
python test_runner.py --schedule hourly --runner-mode api
python test_runner.py --schedule daily --runner-mode api
python test_runner.py --schedule weekly --runner-mode api --send-report-email
```

## 9. Chạy Streamlit dashboard

```bash
streamlit run dashboard.py
```

Mặc định: `http://localhost:8501`

Dashboard có:
- chạy test trực tiếp
- KPI tổng quan
- biểu đồ theo domain/độ khó
- bảng chi tiết từng câu
- manual review và recompute summary
- tải report `.md/.html/.csv`

## 10. Docker chạy trên server

### 10.1 Build và chạy API + Streamlit dashboard

```bash
docker compose up -d api dashboard
```

Port:
- API: `8000`
- Streamlit: `8501`

### 10.2 Chạy scheduler container

```bash
docker compose --profile scheduler up -d scheduler
```

Scheduler container đang dùng:

```bash
python test_runner.py --schedule weekly --runner-mode api --send-report-email
```

### 10.3 Dừng toàn bộ

```bash
docker compose down
```

## 11. Unit test

```bash
pytest -q test.py
```

## 12. Quy trình cập nhật văn bản mới

1. Thêm file văn bản mới vào `Multi-Agent/<Domain>/`.
2. Cập nhật metadata hiệu lực trong `metadata/legal_documents_metadata.csv`.
3. Cập nhật vector store:

```bash
python vector_store.py --update --domain <Domain>
```

4. Nếu thay đổi lớn, build lại toàn bộ:

```bash
python vector_store.py --build
```
