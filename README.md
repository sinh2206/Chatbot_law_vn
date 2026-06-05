# Chatbot Law VN - Legal RAG

Du an xay chatbot hoi dap phap luat Viet Nam bang RAG local:

- Van ban phap luat nam trong `data/processed/`.
- Bo Q/A fine-tune nam trong `data/train/` va `data/test/`.
- Model embedding goc: `dangvantuan/vietnamese-embedding`.
- Model fine-tuned local: `data/models/vietnamese-embedding-legal`.
- Vector store nam trong `data/vector_store/`.
- Gemini chi la fallback, chi goi khi local RAG khong co can cu hop le.
- Frontend tinh nam trong `frontend/`.

## 1. Thu muc chinh

```text
Multi-Agent/          # Tai lieu goc .doc/.docx/.txt theo linh vuc
data/processed/       # Tai lieu .txt da xu ly, dung de build vector store
data/train/           # questions.txt + reference_answers.txt
data/test/            # questons.txt + reference_answers.txt
data/finetune/        # JSONL sinh tu train/test
data/models/          # Model fine-tuned local, khong commit Git
data/vector_store/    # faiss.index/embeddings.npy + metadata + manifest
metadata/             # Metadata hieu luc van ban
scripts/              # Script convert, train, evaluate, query
backend/              # FastAPI backend cho frontend va API /chat
frontend/             # Giao dien chat don gian
```

## 2. Chuan bi moi truong

Yeu cau:

- Linux server.
- Docker Engine.
- Docker Compose plugin: `docker compose`.
- Internet lan dau de tai Python packages va model Hugging Face.
- Neu fine-tune GPU: NVIDIA driver va NVIDIA Container Toolkit.

Kiem tra GPU:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

Build Docker image:

```bash
docker compose build app backend
docker compose --profile gpu build gpu
```

Kiem tra Python packages:

```bash
docker compose run --rm app python --version
docker compose run --rm app python -c "import sentence_transformers, numpy; print('ok')"
```

## 3. Cau hinh `.env`

Tao file `.env`:

```bash
cp .env.example .env
```

Neu muon danh gia/fine-tune khach quan, tat Gemini:

```env
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash-lite
GEMINI_FALLBACK_ENABLED=false
MIN_RETRIEVAL_SCORE=0.45
EMBEDDING_MODEL_NAME=data/models/vietnamese-embedding-legal
```

Neu muon chatbot goi Gemini fallback khi local RAG khong tra loi duoc:

```env
GEMINI_API_KEY=your_real_gemini_key
GEMINI_MODEL=gemini-2.5-flash-lite
GEMINI_FALLBACK_ENABLED=true
MIN_RETRIEVAL_SCORE=0.45
EMBEDDING_MODEL_NAME=data/models/vietnamese-embedding-legal
```

Kiem tra Docker da nhan bien:

```bash
docker compose config | grep -E "GEMINI_API_KEY|GOOGLE_API_KEY|GEMINI_FALLBACK_ENABLED|EMBEDDING_MODEL_NAME|MIN_RETRIEVAL_SCORE"
```

## 4. Chuan bi tai lieu phap luat

Neu `data/processed/` da co file `.txt`, co the bo qua buoc nay.

Convert tai lieu trong `Multi-Agent/`:

```bash
docker compose run --rm app python scripts/convert_docs_to_txt.py --clean-output
```

Convert rieng mot linh vuc:

```bash
docker compose run --rm app python scripts/convert_docs_to_txt.py --domain Thue --overwrite
```

OCR anh scan neu co:

```bash
docker compose run --rm app python scripts/convert_images_to_txt_ocr.py \
  --overwrite \
  --lang vie+eng \
  --preprocess adaptive
```

Kiem tra output:

```bash
find data/processed -maxdepth 2 -type f -name "*.txt" | sort
```

## 5. Tao lai fine-tune JSONL

Bo hien tai dung:

```text
data/train/questions.txt
data/train/reference_answers.txt
data/test/questons.txt
data/test/reference_answers.txt
```

Kiem tra so dong:

```bash
wc -l data/train/questions.txt \
      data/train/reference_answers.txt \
      data/test/questons.txt \
      data/test/reference_answers.txt
```

Ky vong:

```text
1000 data/train/questions.txt
1000 data/train/reference_answers.txt
 250 data/test/questons.txt
 250 data/test/reference_answers.txt
```

Tao JSONL, dung toan bo 1000 cau train va khong tach validation:

```bash
docker compose run --rm app python scripts/prepare_qa_finetune_data.py \
  --valid-ratio 0 \
  --seed 20260603
```

Kiem tra:

```bash
wc -l data/finetune/train_pairs.jsonl \
      data/finetune/valid_pairs.jsonl \
      data/finetune/test_pairs.jsonl
cat data/finetune/summary.json
```

## 6. Validate data

```bash
docker compose run --rm app python scripts/validate_finetune_data.py \
  --model-name dangvantuan/vietnamese-embedding \
  --max-seq-length 256 \
  --output-json data/finetune/validation_report.json
```

Xem bao cao:

```bash
cat data/finetune/validation_report.json
```

Can de y:

- Duplicate nen thap.
- Cau hoi/cau tra loi khong rong.
- Ty le truncate thap.
- Train/test khong bi leak qua nhau.

## 7. Danh gia baseline

Chay model goc truoc khi fine-tune:

```bash
docker compose run --rm app python scripts/evaluate_retrieval.py \
  --model-name dangvantuan/vietnamese-embedding \
  --test-file data/finetune/test_pairs.jsonl \
  --train-file data/finetune/train_pairs.jsonl \
  --valid-file data/finetune/valid_pairs.jsonl \
  --output-json data/finetune/retrieval_eval_baseline.json
```

Xem diem:

```bash
cat data/finetune/retrieval_eval_baseline.json
```

Chi so chinh:

- `recall_at_1`
- `recall_at_5`
- `recall_at_10`
- `mrr_at_10`
- `ndcg_at_10`

## 8. Fine-tune model embedding

Xoa model cu neu muon train lai tu dau:

```bash
rm -rf data/models/vietnamese-embedding-legal
```

Kiem tra GPU co bi chiem VRAM khong:

```bash
nvidia-smi
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

Neu CUDA OOM:

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

Train CPU chi dung de smoke test vi rat cham:

```bash
docker compose run --rm app python scripts/train_embedding.py \
  --model-name dangvantuan/vietnamese-embedding \
  --train-file data/finetune/train_pairs.jsonl \
  --valid-file data/finetune/valid_pairs.jsonl \
  --output-dir data/models/vietnamese-embedding-legal \
  --epochs 1 \
  --batch-size 4 \
  --lr 2e-5 \
  --warmup-ratio 0.1 \
  --max-seq-length 128
```

Kiem tra model da luu:

```bash
ls data/models/vietnamese-embedding-legal/modules.json \
   data/models/vietnamese-embedding-legal/config_sentence_transformers.json
```

## 9. Danh gia model fine-tuned

```bash
docker compose run --rm app python scripts/evaluate_retrieval.py \
  --model-name data/models/vietnamese-embedding-legal \
  --test-file data/finetune/test_pairs.jsonl \
  --train-file data/finetune/train_pairs.jsonl \
  --valid-file data/finetune/valid_pairs.jsonl \
  --output-json data/finetune/retrieval_eval_finetuned.json
```

So sanh baseline va fine-tuned:

```bash
docker compose run --rm -T app python - <<'PY'
import json
from pathlib import Path

for name in ["baseline", "finetuned"]:
    path = Path(f"data/finetune/retrieval_eval_{name}.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    print(name)
    for key in ["recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10"]:
        print(f"  {key}: {data.get(key)}")
PY
```

## 10. Build vector store

Sau khi fine-tune, build lai vector store bang model moi:

```bash
docker compose run --rm app python scripts/build_vector_store.py \
  --embedding-model data/models/vietnamese-embedding-legal \
  --batch-size 64
```

Kiem tra:

```bash
cat data/vector_store/manifest.json
```

Can thay:

```text
"embedding_model": "data/models/vietnamese-embedding-legal"
"total_chunks": > 0
```

## 11. Query CLI local va Gemini fallback

Local RAG khong goi Gemini:

```bash
docker compose run --rm app python scripts/query_cli.py \
  --query "Cong dan can cap doi the can cuoc khi nao?" \
  --domain CCCD \
  --top-k 5 \
  --no-gemini-fallback
```

Bat fallback Gemini:

```bash
docker compose run --rm app python scripts/query_cli.py \
  --query "Tu van chien luoc marketing cho quan ca phe" \
  --domain CCCD \
  --top-k 5 \
  --min-score 0.99 \
  --gemini-fallback
```

Nguyen tac:

- Neu local co chunk hop le voi score >= `MIN_RETRIEVAL_SCORE`: in `[LOCAL RAG]` va khong goi Gemini.
- Neu khong co chunk, sai domain, score thap, hoac chunk bi het hieu luc: in `[FALLBACK REQUIRED]`.
- Gemini chi goi sau do neu `--gemini-fallback` hoac `GEMINI_FALLBACK_ENABLED=true`.

Smoke test 5 linh vuc:

```bash
docker compose run --rm app python scripts/query_cli.py --query "Dang ky khai sinh can giay to gi?" --domain HoTich --top-k 5 --gemini-fallback
docker compose run --rm app python scripts/query_cli.py --query "Ma so doanh nghiep co dong thoi la ma so thue khong?" --domain Thue --top-k 5 --gemini-fallback
docker compose run --rm app python scripts/query_cli.py --query "Tach thua dat can dieu kien gi?" --domain DatDai --top-k 5 --gemini-fallback
docker compose run --rm app python scripts/query_cli.py --query "Co dong sang lap la ai?" --domain DoanhNghiep --top-k 5 --gemini-fallback
docker compose run --rm app python scripts/query_cli.py --query "The can cuoc chua thong tin gi?" --domain CCCD --top-k 5 --gemini-fallback
```

## 12. Cap nhat tai lieu moi

Them hoac sua tai lieu trong `Multi-Agent/`, sau do chay:

```bash
docker compose run --rm app python scripts/convert_docs_to_txt.py --clean-output
docker compose run --rm app python scripts/build_vector_store.py \
  --embedding-model data/models/vietnamese-embedding-legal \
  --batch-size 64
```

Chi cap nhat mot domain:

```bash
docker compose run --rm app python scripts/convert_docs_to_txt.py --domain Thue --overwrite
docker compose run --rm app python scripts/update_vector_store.py \
  --domain Thue \
  --scope domain \
  --overwrite \
  --embedding-model data/models/vietnamese-embedding-legal
```

## 13. Backend API

Backend API nam trong `backend/app.py`, dung FastAPI va tai su dung logic retrieval trong `scripts/query_cli.py`.

Chay backend:

```bash
docker compose up --build backend
```

Hoac chay nen:

```bash
docker compose up -d backend
```

Kiem tra health:

```bash
curl http://localhost:8000/health
```

Goi chat API:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Cong dan can cap doi the can cuoc khi nao?",
    "domain": "CCCD",
    "top_k": 5,
    "gemini_fallback": true
  }'
```

Reload model/index sau khi build lai vector store:

```bash
curl -X POST http://localhost:8000/reload
```

Endpoint:

```text
GET  /health
POST /chat
POST /reload
GET  /              # serve frontend/
```

Payload `/chat`:

```json
{
  "message": "Cau hoi cua nguoi dung",
  "domain": "CCCD",
  "top_k": 5,
  "min_score": 0.45,
  "gemini_fallback": true
}
```

Nguyen tac fallback:

- Backend luon search local vector store truoc.
- Neu local co can cu hop le, response co `mode="local_rag"` va `gemini_used=false`.
- Neu local khong du tin cay va `gemini_fallback=true`, backend moi goi Gemini.
- Neu Gemini tat hoac thieu key, response co `mode="fallback_required"` hoac `mode="gemini_error"`.

## 14. Frontend

Frontend nam trong `frontend/`. Day la giao dien chat tinh, kieu ChatGPT don gian.

Chay cung backend:

```bash
docker compose up backend
```

Mo:

```text
http://localhost:8000
```

Hoac serve rieng frontend bang Python:

```bash
cd frontend
python3 -m http.server 8088
```

Mo:

```text
http://localhost:8088
```

Khi chay chung qua backend, mac dinh frontend goi endpoint:

```text
/chat
```

Neu serve rieng frontend bang `python3 -m http.server`, nhap endpoint `http://localhost:8000/chat` tren giao dien.

Payload frontend gui:

```json
{
  "message": "Cau hoi cua nguoi dung",
  "domain": "CCCD",
  "top_k": 5,
  "gemini_fallback": true
}
```

Neu backend nam o host/port khac, sua o Endpoint tren giao dien roi bam nut luu.

## 15. Bao cao can xem

```text
data/finetune/summary.json
data/finetune/validation_report.json
data/finetune/retrieval_eval_baseline.json
data/finetune/retrieval_eval_finetuned.json
data/models/vietnamese-embedding-legal/train_summary.json
data/vector_store/manifest.json
```

## 16. Ghi chu ve tinh khach quan

- Fine-tune embedding: nen tat API.
- Evaluate retrieval: nen tat API.
- Query/demo san pham: co the bat Gemini fallback.
- Khi bat fallback, local RAG van duoc thu truoc. Gemini chi duoc goi khi local khong co can cu hop le.
