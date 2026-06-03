# Chatbot Law VN - Offline Legal RAG

Du an hien tai chua co frontend va chua mo API web. He thong chay tren server Linux bang script/CLI de:
- Chuyen van ban phap luat `.doc/.docx/.txt/anh scan` thanh `.txt`.
- Tao vector store tu `data/processed`.
- Fine-tune embedding model tu bo Q/A trong `data/train`.
- Danh gia model bang `data/test`.
- Truy van thu bang CLI.
- Goi Gemini API chi trong truong hop fallback: khong tim thay can cu noi bo, score qua thap, sai linh vuc, hoac tai lieu lien quan bi danh dau het hieu luc.

Mac dinh local RAG luon co uu tien cao nhat. Gemini co trong so thap nhat va chi chay khi bat `--gemini-fallback` hoac `GEMINI_FALLBACK_ENABLED=true`. Model embedding chinh la `dangvantuan/vietnamese-embedding` qua `sentence-transformers`.

## 1. Kien truc RAG

Khi nguoi dung nhap cau hoi, luong xu ly dung Retrieval-Augmented Generation:
- Query embedding: cau hoi duoc chuyen thanh vector.
- Vector search: tim cac chunk gan nghia nhat trong FAISS hoac numpy index.
- Retrieval output: tra ve doan van ban, domain, source file, chunk id va score.
- Expiry check: loai bo chunk thuoc van ban bi danh dau het hieu luc trong `metadata/legal_documents_metadata.csv`.
- Fallback decision: neu khong co can cu noi bo hop le thi in ly do truoc, sau do moi goi Gemini neu duoc bat.
- Answer generation: Gemini chi sinh cau tra loi fallback, khong duoc xem la can cu tu `data/processed`.

Trong du an nay:
- "Kien thuc" nam trong `data/processed` va `data/vector_store`.
- Fine-tune data trong `data/train` giup model embedding tim dung doan phap ly hon.
- Sau moi lan fine-tune, bat buoc build lai vector store bang model moi.
- Trang thai hieu luc van ban nam trong `metadata/legal_documents_metadata.csv`.

## 2. Cau truc thu muc

```text
Multi-Agent/
  DoanhNghiep/
  HoTich/
  CCCD/
  DatDai/
  Thue/

data/
  processed/      # File .txt da chuan hoa theo tung linh vuc
  train/          # questions.txt + reference_answers.txt de fine-tune
  test/           # questons.txt/questions.txt + reference_answers.txt de danh gia
  finetune/       # JSONL sinh ra tu data/train va data/test
  vector_store/   # faiss.index hoac embeddings.npy + metadata.jsonl + manifest.json
  models/         # Model fine-tuned local, khong commit Git

scripts/
  convert_docs_to_txt.py
  convert_images_to_txt_ocr.py
  prepare_qa_finetune_data.py
  validate_finetune_data.py
  evaluate_retrieval.py
  train_embedding.py
  build_vector_store.py
  update_vector_store.py
  query_cli.py

metadata/
  legal_documents_metadata.csv

Dockerfile
docker-compose.yml
.dockerignore
.env.example
requirements.txt
config.py
```

## 3. Chay bang Docker tren Linux server

Day la cach khuyen nghi de chay tren server. Khong can frontend.

### 3.1. Yeu cau server

Can co:
- Linux server.
- Docker Engine.
- Docker Compose plugin: `docker compose`.
- Internet lan dau de tai Python packages va model Hugging Face.

Neu muon train bang GPU:
- Server co NVIDIA GPU.
- Driver NVIDIA hoat dong: `nvidia-smi`.
- NVIDIA Container Toolkit da duoc cai.
- Kiem tra Docker GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

Neu lenh tren loi, sua Docker GPU truoc khi chay service `gpu`.

### 3.2. Clone project

```bash
git clone https://github.com/sinh2206/Chatbot_law_vn.git
cd Chatbot_law_vn
```

Tao file `.env` tu mau:

```bash
cp .env.example .env
```

Neu muon dung Gemini fallback, sua `.env`:

```text
GEMINI_API_KEY=your_real_key
GEMINI_MODEL=gemini-2.5-flash-lite
GEMINI_FALLBACK_ENABLED=true
MIN_RETRIEVAL_SCORE=0.45
```

Neu chua muon goi API, giu:

```text
GEMINI_FALLBACK_ENABLED=false
```

### 3.3. Build Docker image CPU

Dung de convert data, validate, evaluate, build vector store CPU hoac query CLI:

```bash
docker compose build app
```

Kiem tra container:

```bash
docker compose run --rm app python --version
docker compose run --rm app python -c "import sentence_transformers, numpy; print('ok')"
```

### 3.4. Build Docker image GPU

Chi can neu fine-tune tren GPU:

```bash
docker compose --profile gpu build gpu
```

Kiem tra PyTorch thay GPU:

```bash
docker compose --profile gpu run --rm -T gpu python - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
print("device=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

Neu `cuda_available=False`, container chua truy cap duoc GPU.

## 4. Chuan bi du lieu van ban

Neu `data/processed` da co `.txt` dung tieng Viet, co the bo qua buoc convert/OCR.

### 4.1. Convert `.doc/.docx/.txt` tu `Multi-Agent`

Chuyen toan bo raw documents:

```bash
docker compose run --rm app python scripts/convert_docs_to_txt.py --clean-output
```

Chuyen rieng mot linh vuc:

```bash
docker compose run --rm app python scripts/convert_docs_to_txt.py --domain Thue --overwrite
```

### 4.2. OCR anh scan van ban phap luat

Neu trong `Multi-Agent` co anh chup/scan:

```bash
docker compose run --rm app python scripts/convert_images_to_txt_ocr.py \
  --overwrite \
  --lang vie+eng \
  --preprocess adaptive
```

OCR rieng mot linh vuc:

```bash
docker compose run --rm app python scripts/convert_images_to_txt_ocr.py \
  --domain DatDai \
  --overwrite \
  --lang vie+eng
```

Sau buoc nay, kiem tra:

```bash
find data/processed -type f -name "*.txt" | wc -l
find data/processed -maxdepth 2 -type f -name "*.txt"
```

## 5. Tai/kiem tra model embedding goc

Model goc:

```text
dangvantuan/vietnamese-embedding
```

Kiem tra model load duoc:

```bash
docker compose run --rm -T app python - <<'PY'
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("dangvantuan/vietnamese-embedding")
print("embedding_dim=", m.get_sentence_embedding_dimension())
print("max_seq_length=", m.max_seq_length)
PY
```

Neu server can chay offline sau khi tai xong, luu model vao `model/`:

```bash
docker compose run --rm -T app python - <<'PY'
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("dangvantuan/vietnamese-embedding")
m.save("model")
PY
```

Sau do co the ep project dung model local:

```bash
export EMBEDDING_MODEL_NAME=/app/model
```

Khi dung Docker Compose, bien nay co the truyen truc tiep:

```bash
EMBEDDING_MODEL_NAME=/app/model docker compose run --rm app python scripts/build_vector_store.py
```

## 6. Tao fine-tune dataset tu `data/train` va `data/test`

Script nay doc:
- `data/train/questions.txt`
- `data/train/reference_answers.txt`
- `data/test/questons.txt` hoac `data/test/questions.txt`
- `data/test/reference_answers.txt`

Chay:

```bash
docker compose run --rm app python scripts/prepare_qa_finetune_data.py \
  --valid-ratio 0.1 \
  --seed 42
```

Ket qua mong doi voi bo hien tai:

```text
manual_train_pairs_total: 1000
train_pairs: 900
valid_pairs: 100
test_pairs: 250
```

File sinh ra:

```text
data/finetune/train_pairs.jsonl
data/finetune/valid_pairs.jsonl
data/finetune/test_pairs.jsonl
data/finetune/train_triplets.jsonl
data/finetune/valid_triplets.jsonl
data/finetune/test_triplets.jsonl
data/finetune/summary.json
```

Neu muon dung ca 1000 cau train de train va khong tach validation:

```bash
docker compose run --rm app python scripts/prepare_qa_finetune_data.py \
  --valid-ratio 0 \
  --seed 42
```

Khuyen nghi van giu `--valid-ratio 0.1` de co diem validation trong luc train.

## 7. Validate fine-tune dataset

```bash
docker compose run --rm app python scripts/validate_finetune_data.py \
  --model-name dangvantuan/vietnamese-embedding \
  --max-seq-length 256 \
  --output-json data/finetune/validation_report.json
```

Mo file:

```bash
cat data/finetune/validation_report.json
```

Can xem:
- `duplicate_pairs_pct` nen gan `0`.
- `suspicious_questionmark_rows` nen bang `0`.
- `positive_truncated_pct` nen thap, tot nhat duoi `10%`.
- `overlap_source_count` giua train/valid/test nen bang `0`.

## 8. Danh gia baseline truoc fine-tune

Chay model goc tren test set:

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

Chi so can quan tam:
- `recall_at_1`: cau dung nam top 1.
- `recall_at_5`: cau dung nam trong top 5.
- `recall_at_10`: cau dung nam trong top 10.
- `mrr_at_10`: rank trung binh co trong so.
- `ndcg_at_10`: chat luong thu hang top 10.

## 9. Fine-tune model embedding

### 9.1. Fine-tune bang GPU

Khuyen nghi dung service `gpu`:

```bash
docker compose --profile gpu run --rm gpu python scripts/train_embedding.py \
  --model-name dangvantuan/vietnamese-embedding \
  --train-file data/finetune/train_pairs.jsonl \
  --valid-file data/finetune/valid_pairs.jsonl \
  --output-dir data/models/vietnamese-embedding-legal \
  --epochs 3 \
  --batch-size 16 \
  --lr 2e-5 \
  --warmup-ratio 0.1 \
  --max-seq-length 256 \
  --use-amp
```

Neu GPU out-of-memory:
- Giam `--batch-size 16` xuong `12` hoac `8`.
- Giu `--max-seq-length 256` truoc, chi tang len khi can va GPU con du VRAM.

### 9.2. Fine-tune bang CPU

Chi nen dung de smoke test vi rat cham:

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
  --max-seq-length 256
```

Model sau train nam tai:

```text
data/models/vietnamese-embedding-legal/
```

## 10. Danh gia model sau fine-tune

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

Model fine-tuned tot hon khi cac chi so tren tang so voi baseline.

Ngoai ra xem file validation trong qua trinh train:

```text
data/models/vietnamese-embedding-legal/eval/binary_classification_evaluation_valid_binary_results.csv
data/models/vietnamese-embedding-legal/train_summary.json
```

## 11. Build vector store cho chatbot/retrieval

Sau fine-tune, bat buoc build lai vector store bang model moi:

```bash
docker compose run --rm app python scripts/build_vector_store.py \
  --embedding-model data/models/vietnamese-embedding-legal \
  --batch-size 64
```

Neu chua fine-tune va muon build bang model goc:

```bash
docker compose run --rm app python scripts/build_vector_store.py \
  --embedding-model dangvantuan/vietnamese-embedding \
  --batch-size 64
```

Neu server co GPU va muon build nhanh hon:

```bash
docker compose --profile gpu run --rm gpu python scripts/build_vector_store.py \
  --embedding-model data/models/vietnamese-embedding-legal \
  --batch-size 128
```

Kiem tra manifest:

```bash
cat data/vector_store/manifest.json
```

Can thay:
- `embedding_model` dung model vua build.
- `total_chunks` > 0.
- `index_backend` la `faiss` neu `faiss-cpu` cai duoc, hoac `numpy` neu fallback.

## 12. Khai bao hieu luc van ban

File:

```text
metadata/legal_documents_metadata.csv
```

Cot quan trong:
- `source_file`: phai trung voi duong dan trong `data/processed`, vi du `DatDai/Luat_45_2013_QH13.txt`.
- `domain`: linh vuc.
- `expiry_date`: ngay het hieu luc, chap nhan `YYYY-MM-DD`, `DD/MM/YYYY`, `DD-MM-YYYY`.
- `status`: neu dat `expired`, `het_hieu_luc`, `inactive` thi he thong luon coi van ban da het hieu luc.
- `replaced_by`: van ban thay the neu biet.

Vi du danh dau mot van ban het hieu luc:

```csv
source_file,domain,document_title,document_number,issued_date,effective_date,expiry_date,status,replaced_by
DatDai/Luat_45_2013_QH13.txt,DatDai,Luat Dat dai,45/2013/QH13,2013-11-29,2014-07-01,2024-08-01,expired,Luat 31/2024/QH15
```

Khi query:
- Neu tat ca ket qua phu hop deu thuoc van ban het hieu luc, CLI se khong dung cac chunk do.
- CLI in ly do `FALLBACK REQUIRED`.
- Neu Gemini fallback duoc bat, sau do moi goi Gemini.
- Neu Gemini fallback tat, CLI chi thong bao ly do va dung lai.

## 13. Query CLI

Truy van nhanh:

```bash
docker compose run --rm app python scripts/query_cli.py \
  --query "Cong dan can cap doi the can cuoc khi nao?" \
  --domain CCCD \
  --top-k 5
```

Bat Gemini fallback cho mot lan query:

```bash
docker compose run --rm app python scripts/query_cli.py \
  --query "Quy dinh moi nhat ve tach thua dat la gi?" \
  --domain DatDai \
  --top-k 5 \
  --gemini-fallback
```

Neu khong co can cu noi bo hop le, output se co dang:

```text
[FALLBACK REQUIRED]
Khong truy xuat duoc doan tai lieu phu hop...

[GEMINI FALLBACK] Dang goi Gemini vi khong co can cu noi bo hop le.
```

Chay che do hoi-dap terminal:

```bash
docker compose run --rm app python scripts/query_cli.py --top-k 5
```

Hien day du chunk:

```bash
docker compose run --rm app python scripts/query_cli.py \
  --query "Ho so dang ky doanh nghiep gom nhung gi?" \
  --domain DoanhNghiep \
  --top-k 5 \
  --show-full
```

Dieu chinh nguong score noi bo:

```bash
docker compose run --rm app python scripts/query_cli.py \
  --query "Cau hoi can kiem tra" \
  --min-score 0.50 \
  --gemini-fallback
```

Nguyen tac uu tien:
- Neu co ket qua noi bo hop le voi score >= `--min-score` va khong het hieu luc: dung local RAG, khong goi Gemini.
- Neu domain khong ton tai, khong co ket qua, score thap, hoac ket qua deu het hieu luc: moi fallback.
- Gemini khong duoc xem la nguon can cu trong kho noi bo.

## 14. Cap nhat tai lieu moi

Khi them/xoa van ban trong `Multi-Agent`:

```bash
docker compose run --rm app python scripts/convert_docs_to_txt.py --clean-output
docker compose run --rm app python scripts/build_vector_store.py \
  --embedding-model data/models/vietnamese-embedding-legal \
  --batch-size 64
```

Neu chi cap nhat mot linh vuc:

```bash
docker compose run --rm app python scripts/convert_docs_to_txt.py --domain Thue --overwrite
docker compose run --rm app python scripts/update_vector_store.py \
  --domain Thue \
  --scope domain \
  --overwrite \
  --embedding-model data/models/vietnamese-embedding-legal
```

## 15. Chay khong Docker bang virtualenv

Dung khi can debug truc tiep tren server:

```bash
sudo apt-get update
sudo apt-get install -y git python3.10 python3.10-venv python3.10-dev \
  build-essential antiword catdoc tesseract-ocr tesseract-ocr-vie

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Sau do chay cac script giong Docker, bo tien to `docker compose run --rm app`.

Vi du:

```bash
python scripts/prepare_qa_finetune_data.py --valid-ratio 0.1 --seed 42
python scripts/validate_finetune_data.py --model-name dangvantuan/vietnamese-embedding --max-seq-length 256
python scripts/train_embedding.py \
  --model-name dangvantuan/vietnamese-embedding \
  --train-file data/finetune/train_pairs.jsonl \
  --valid-file data/finetune/valid_pairs.jsonl \
  --output-dir data/models/vietnamese-embedding-legal \
  --epochs 3 \
  --batch-size 16 \
  --lr 2e-5 \
  --warmup-ratio 0.1 \
  --max-seq-length 256 \
  --use-amp
```

## 16. Files diem va bao cao

Sau khi chay day du, xem cac file:

```text
data/finetune/summary.json
data/finetune/validation_report.json
data/finetune/retrieval_eval_baseline.json
data/finetune/retrieval_eval_finetuned.json
data/models/vietnamese-embedding-legal/train_summary.json
data/models/vietnamese-embedding-legal/eval/binary_classification_evaluation_valid_binary_results.csv
data/vector_store/manifest.json
```

Y nghia:
- `validation_report.json`: data co bi duplicate, loi encoding, truncate, leakage khong.
- `retrieval_eval_baseline.json`: diem model goc.
- `retrieval_eval_finetuned.json`: diem model sau fine-tune.
- `train_summary.json`: thong tin train.
- `manifest.json`: vector store dang dung model nao, bao nhieu chunk, backend nao.

## 17. Luu y Git va artifact lon

Nhung thu muc sau khong nen commit:

```text
model/
data/models/
checkpoints/
data/vector_store/
data/finetune/
```

Model va vector store co the tao lai tren server bang cac buoc trong README.
