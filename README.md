# Offline Legal RAG (No API)

Du an xu ly van ban phap luat theo huong offline:
- Khong dung OpenAI/Gemini API.
- Embedding dung `sentence-transformers` voi model local/Hugging Face.
- Vector search dung FAISS local neu cai duoc, fallback sang numpy brute-force neu chua co FAISS.
- Fine-tune embedding model tu bo Q/A trong `data/train`, danh gia bang `data/test`.

## 2.3.1 Yeu cau bai toan

Khi nguoi dung nhap mot cau hoi, he thong can tra ve cau tra loi chinh xac va tu nhien dua tren du lieu da tai len:
- Hieu ngon ngu tu nhien tu cau hoi nguoi dung.
- Tim kiem trong co so du lieu vector nhung doan van ban gan nhat ve ngu nghia.
- Ket hop du lieu tim duoc voi mo hinh ngon ngu lon local neu can sinh cau tra loi tu nhien.
- Dam bao cau tra loi khong bia dat, ma gan lien voi du lieu co that trong tai lieu.

## 2.3.2 Giai phap

He thong ap dung kien truc Retrieval-Augmented Generation (RAG):
- Query embedding: chuyen cau hoi thanh vector embedding.
- Vector search: tim `k` doan van ban gan nhat trong FAISS/numpy index.
- Answer generation: dua cac doan van ban va cau hoi vao LLM local neu can sinh cau tra loi.
- Post-processing: dinh dang cau tra loi va bo sung trich dan nguon du lieu.

Phien ban hien tai da hoan thien cac phan offline: convert/OCR du lieu, chunking, build vector store, fine-tune embedding, retrieval evaluation va query CLI. Phan sinh cau tra loi tu nhien bang LLM nen dung model local neu van giu yeu cau khong API.

## Cau truc chinh

```text
Multi-Agent/
  DoanhNghiep/
  HoTich/
  CCCD/
  DatDai/
  Thue/

data/
  processed/      # txt da chuan hoa tu .doc/.docx/.txt/.png/.jpg...
  train/          # questions.txt + reference_answers.txt de fine-tune
  test/           # questons.txt/questions.txt + reference_answers.txt de danh gia
  finetune/       # train/valid/test jsonl + bao cao danh gia
  vector_store/   # faiss.index hoac embeddings.npy + metadata + manifest
  models/         # model fine-tuned local, khong commit len Git

scripts/
  convert_docs_to_txt.py
  convert_images_to_txt_ocr.py
  prepare_qa_finetune_data.py
  bootstrap_finetune_data.py
  train_embedding.py
  validate_finetune_data.py
  evaluate_retrieval.py
  build_vector_store.py
  update_vector_store.py
  query_cli.py
```

## Cai dat local Windows

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip wheel setuptools
pip install -r requirements.txt
```

Neu xu ly file `.doc` cu:
- Linux: cai `antiword` hoac `catdoc`.
- Windows: cai `antiword`/`catdoc` vao PATH.
- Windows co Microsoft Word: `pywin32` co the dung COM fallback.

## OCR anh van ban phap luat

Cai Tesseract OCR:
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-vie
```

Quet tat ca anh trong `Multi-Agent/`:
```bash
python scripts/convert_images_to_txt_ocr.py --overwrite --lang vie+eng --preprocess adaptive
```

OCR theo mot domain:
```bash
python scripts/convert_images_to_txt_ocr.py --domain DatDai --overwrite --lang vie+eng
```

Windows neu `tesseract` khong nam trong PATH:
```bash
python scripts/convert_images_to_txt_ocr.py --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Sau OCR, file txt duoc luu vao `data/processed/<domain>/...`.

## Convert doc/docx sang txt

```bash
python scripts/convert_docs_to_txt.py --clean-output
python scripts/convert_docs_to_txt.py --domain Thue --overwrite
```

## Build / Update vector store

Build moi bang model mac dinh trong `config.py`:
```bash
python scripts/build_vector_store.py
```

Build bang model goc:
```bash
python scripts/build_vector_store.py --embedding-model dangvantuan/vietnamese-embedding
```

Build lai bang model da fine-tune:
```bash
python scripts/build_vector_store.py --embedding-model data/models/vietnamese-embedding-legal
```

Cap nhat lai sau khi them/xoa van ban:
```bash
python scripts/update_vector_store.py --scope full --clean-output
python scripts/update_vector_store.py --domain DatDai --scope domain --overwrite
```

## Query CLI (khong API)

```bash
python scripts/query_cli.py --top-k 5
python scripts/query_cli.py --query "Ho so dang ky doanh nghiep gom gi?" --domain DoanhNghiep
```

`query_cli.py` uu tien model ghi trong `data/vector_store/manifest.json`, vi vay sau khi fine-tune phai build vector store lai bang model fine-tuned.

## Chay tren Linux server (khong API)

Quy trinh nay dam bao model hoc tu `data/train` va test tren `data/test`. Lan dau co the can internet de tai model tu Hugging Face; day khong phai API inference. Neu server offline hoan toan, hay copy san model vao `model/` hoac `data/models/...`.

### 1. Cai system packages

Khuyen nghi Python 3.10 hoac 3.11 de cai FAISS on dinh:
```bash
sudo apt-get update
sudo apt-get install -y git python3.10 python3.10-venv python3.10-dev \
  build-essential antiword catdoc tesseract-ocr tesseract-ocr-vie
```

Clone repo:
```bash
git clone https://github.com/sinh2206/Chatbot_law_vn.git
cd Chatbot_law_vn
```

Tao virtual environment:
```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
```

Neu server co NVIDIA GPU, cai PyTorch CUDA truoc:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Cai dependencies:
```bash
pip install -r requirements.txt
```

Kiem tra GPU:
```bash
python - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
print("device=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

### 2. Kiem tra model `dangvantuan/vietnamese-embedding`

```bash
python - <<'PY'
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("dangvantuan/vietnamese-embedding")
print("embedding_dim=", m.get_sentence_embedding_dimension())
print("max_seq_length=", m.max_seq_length)
PY
```

Neu muon chay offline sau khi tai xong, luu model vao `model/`:
```bash
python - <<'PY'
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("dangvantuan/vietnamese-embedding")
m.save("model")
PY
export EMBEDDING_MODEL_NAME="$PWD/model"
```

### 3. Tao fine-tune JSONL tu `data/train` va `data/test`

Lenh nay doc:
- `data/train/questions.txt`
- `data/train/reference_answers.txt`
- `data/test/questons.txt` hoac `data/test/questions.txt`
- `data/test/reference_answers.txt`

Va ghi ra `data/finetune/*.jsonl`:
```bash
python scripts/prepare_qa_finetune_data.py --valid-ratio 0.1 --seed 42
```

Ket qua mong doi voi bo hien tai:
```text
manual_train_pairs_total: 1000
train_pairs: 900
valid_pairs: 100
test_pairs: 250
```

Neu muon dung ca 1000 cau train de train va khong tach validation:
```bash
python scripts/prepare_qa_finetune_data.py --valid-ratio 0 --seed 42
```

### 4. Validate data truoc khi train

```bash
python scripts/validate_finetune_data.py \
  --model-name dangvantuan/vietnamese-embedding \
  --max-seq-length 256 \
  --output-json data/finetune/validation_report.json
```

Can xem trong `data/finetune/validation_report.json`:
- `duplicate_pairs_pct` nen gan `0`.
- `suspicious_questionmark_rows` nen bang `0`.
- `positive_truncated_pct` nen thap, tot nhat duoi `10%`.
- `overlap_source_count` giua train/valid/test nen bang `0`.

### 5. Chay baseline truoc fine-tune

```bash
python scripts/evaluate_retrieval.py \
  --model-name dangvantuan/vietnamese-embedding \
  --test-file data/finetune/test_pairs.jsonl \
  --train-file data/finetune/train_pairs.jsonl \
  --valid-file data/finetune/valid_pairs.jsonl \
  --output-json data/finetune/retrieval_eval_baseline.json
```

File diem baseline: `data/finetune/retrieval_eval_baseline.json`.

### 6. Fine-tune embedding model

Lenh khuyen nghi cho GPU T4/16GB:
```bash
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

Neu GPU out-of-memory, giam `--batch-size` xuong `8` hoac `12`. Neu train tren CPU qua cham, nen chuyen sang GPU.

### 7. Test model sau fine-tune

```bash
python scripts/evaluate_retrieval.py \
  --model-name data/models/vietnamese-embedding-legal \
  --test-file data/finetune/test_pairs.jsonl \
  --train-file data/finetune/train_pairs.jsonl \
  --valid-file data/finetune/valid_pairs.jsonl \
  --output-json data/finetune/retrieval_eval_finetuned.json
```

So sanh baseline va fine-tuned:
```bash
python - <<'PY'
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

Model fine-tuned tot hon khi `recall_at_k`, `mrr_at_10`, `ndcg_at_10` tang so voi baseline.

### 8. Build lai vector store bang model fine-tuned

Vector store cu khong tu cap nhat sau fine-tune. Phai build lai de query embedding va document embedding cung dimension/cung khong gian vector:
```bash
python scripts/build_vector_store.py \
  --embedding-model data/models/vietnamese-embedding-legal \
  --batch-size 64
```

Kiem tra manifest:
```bash
cat data/vector_store/manifest.json
```

Can thay:
- `embedding_model` la `data/models/vietnamese-embedding-legal`.
- `total_chunks` khop so chunk trong kho tai lieu.
- `index_backend` nen la `faiss` tren Linux/Python 3.10 neu `faiss-cpu` cai thanh cong.

### 9. Chay truy van thu

```bash
python scripts/query_cli.py --top-k 5
python scripts/query_cli.py --query "Cong dan can cap doi the can cuoc khi nao?" --domain CCCD
```

## Google Colab GPU T4

Neu chay tren Colab:
```python
REPO_URL = "https://github.com/sinh2206/Chatbot_law_vn.git"
REPO_DIR = "/content/Chatbot_law_vn"

import os
if not os.path.exists(REPO_DIR):
    !git clone $REPO_URL $REPO_DIR
else:
    !git -C $REPO_DIR pull --ff-only

%cd $REPO_DIR
```

Cai dependencies:
```bash
!apt-get update
!apt-get install -y tesseract-ocr tesseract-ocr-vie
!python -m pip install -U pip wheel setuptools
!pip install torch --index-url https://download.pytorch.org/whl/cu121
!pip install -r requirements.txt
```

Sau do chay tu buoc `3` den buoc `8` trong phan Linux server, them dau `!` truoc moi command shell.

## Nhin vao dau de xac dinh diem model

Retrieval metrics quan trong nhat cho RAG:
- `data/finetune/retrieval_eval_baseline.json`
- `data/finetune/retrieval_eval_finetuned.json`

Chi so can so sanh:
- `recall_at_1`, `recall_at_3`, `recall_at_5`, `recall_at_10`
- `mrr_at_10`
- `ndcg_at_10`

Eval trong qua trinh train:
- `data/models/vietnamese-embedding-legal/eval/binary_classification_evaluation_valid_binary_results.csv`

Cac cot chinh:
- `cosine_ap`, `cosine_f1`, `cosine_accuracy`, `cosine_mcc`

Tong ket train:
- `data/models/vietnamese-embedding-legal/train_summary.json`

## Luu y ky thuat

- `data/models/`, `model/`, `checkpoints/` la artifact lon va da duoc ignore trong Git.
- Model thu nghiem `data/models/smoke-vietnamese-embedding*` khong con can thiet va nen xoa khoi repository.
- Tren Windows/Python 3.12, `faiss-cpu` co the khong cai duoc; script se fallback sang numpy index.
- Tren Linux server nen dung Python 3.10/3.11 de co `faiss-cpu`.
- Sau moi lan thay doi model embedding, phai build lai vector store.
