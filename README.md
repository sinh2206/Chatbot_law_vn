# Chatbot Law VN - Multi-Agent RAG + Evaluation Dashboard

Hệ thống gồm 3 khối chính:
- Chatbot pháp luật đa agent (RAG + fallback web khi văn bản hết hiệu lực).
- Bộ test tự động đánh giá chất lượng trả lời.
- Dashboard theo dõi KPI + review thủ công + xuất báo cáo.

## 1) Thành phần đã triển khai

- `api.py`
  - `POST /chat` (hoặc `/ask`) nhận `question`, `domain`, `run_type` (`user`/`test`).
  - `POST /feedback` lưu phản hồi người dùng.
  - `GET /metrics/*` trả KPI và lịch sử test.
  - `POST /metrics/tests/review` cập nhật điểm thủ công.
  - Serve chat frontend tại `/` và dashboard realtime tại `/dashboard`.

- `run_test_suite.py`
  - Chạy toàn bộ test set tự động.
  - Hỗ trợ mode:
    - `smoke` (nhanh)
    - `monitoring` (random N câu)
    - `full` (toàn bộ)
  - Hỗ trợ runner:
    - `api` (gọi API nội bộ `/chat`)
    - `direct` (gọi orchestrator trực tiếp)
  - Chấm điểm theo:
    - heuristic
    - tùy chọn LLM grader (Gemini)
  - Tính và lưu:
    - accuracy
    - mean score
    - avg latency
    - fallback rate
    - citation validity rate
    - citation F1
    - legal entity F1
  - Xuất báo cáo:
    - Markdown
    - HTML
    - CSV
  - Tùy chọn gửi báo cáo qua email admin (SMTP).

- `dashboard.py` (Streamlit)
  - Chạy test trực tiếp từ giao diện.
  - Hiển thị KPI tổng quan.
  - Bảng chi tiết từng câu hỏi: question, answer, ground truth, score, pass/fail.
  - Biểu đồ:
    - accuracy theo lĩnh vực
    - accuracy theo độ khó
    - latency
    - F1 theo từng câu
  - Review bán tự động:
    - chỉnh `manual_score`, `manual_passed`, `manual_note`
    - lưu vào DB để lần sau dùng lại.

- `telemetry_store.py`
  - SQLite (`logs/telemetry.db`) lưu:
    - `interactions`, `agent_traces`, `retrieved_chunks`, `feedback`
    - `test_runs`, `test_results`
  - Có cột dành cho manual review và metric nâng cao.

- `testsuite/default_cases.py`, `testsuite/test_cases.json`
  - Bộ test nhiều cấp độ (dễ/trung bình/khó), phân bố theo 5 lĩnh vực.
  - Mặc định `bootstrap_testset.py` sinh 45 câu cân bằng (9 câu mỗi lĩnh vực).

- `schedule_test_runner.py`
  - Chạy test định kỳ:
    - hourly monitoring
    - daily full
    - weekly full

## 2) Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Tạo `.env` từ `.env.example`.

## 3) Chuẩn bị dữ liệu pháp lý

1. Đặt văn bản tại `Multi-Agent/<Domain>/`.
2. Sinh metadata hiệu lực:

```bash
python build_store.py --bootstrap-metadata
```

3. Điền `issue_date`, `effective_date`, `expiry_date`, `status_override` trong `metadata/legal_documents_metadata.csv`.
4. Build vector store:

```bash
python build_store.py
```

## 4) Chạy backend + frontend chat

```bash
uvicorn api:app --reload
```

- Chat UI: `http://localhost:8000/`
- Dashboard realtime: `http://localhost:8000/dashboard`

## 5) Chạy dashboard đánh giá (Streamlit)

```bash
streamlit run dashboard.py
```

Mặc định mở ở `http://localhost:8501`.

## 6) Chạy test tự động bằng CLI

Smoke:

```bash
python run_test_suite.py --mode smoke --runner-mode api
```

Full:

```bash
python run_test_suite.py --mode full --runner-mode api --llm-grader
```

Monitoring lặp 3 lần để lấy trung bình:

```bash
python run_test_suite.py --mode monitoring --monitoring-size 20 --repeats 3 --runner-mode api
```

Sinh lại test set:

```bash
python bootstrap_testset.py                 # 45 câu cân bằng
python bootstrap_testset.py --all           # full bộ mặc định
python bootstrap_testset.py --target-size 50
```

Xuất báo cáo + gửi email:

```bash
python run_test_suite.py --mode full --runner-mode api --send-report-email
```

## 7) Lịch chạy định kỳ

```bash
python schedule_test_runner.py --plan hourly-monitoring --runner-mode api
python schedule_test_runner.py --plan daily-full --runner-mode api
python schedule_test_runner.py --plan weekly-full --runner-mode api --send-report-email
```

Nếu deploy Linux server, có thể chạy bằng `cron`:

```cron
0 2 * * 1 cd /opt/chatbot_law_vn && /usr/bin/python run_test_suite.py --mode full --runner-mode api --send-report-email
```

## 8) Docker chạy server

### Build và chạy API + Streamlit dashboard

```bash
docker compose up -d api dashboard
```

- API: `http://<server-ip>:8000`
- Streamlit dashboard: `http://<server-ip>:8501`

### Chạy thêm scheduler định kỳ trong container

```bash
docker compose --profile scheduler up -d scheduler
```

### Dừng

```bash
docker compose down
```

## 9) Cấu hình quan trọng trong `.env`

- Test runner:
  - `TEST_RUNNER_MODE`
  - `TEST_API_BASE_URL`
  - `TEST_API_TIMEOUT_SECONDS`
  - `TEST_PASS_SCORE`
  - `TEST_USE_LLM_GRADER`
  - `TEST_REPORT_SEND_EMAIL`

- Báo cáo:
  - `REPORTS_DIR`

- Email SMTP:
  - `ADMIN_EMAIL`
  - `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`
  - `SMTP_USE_TLS` hoặc `SMTP_USE_SSL`

## 10) Cập nhật văn bản pháp lý mới

1. Thêm file mới vào đúng domain.
2. Cập nhật metadata hiệu lực.
3. Cập nhật vector store:

```bash
python update_store.py --domain Thue
```

Hoặc full rebuild:

```bash
python update_store.py
```
