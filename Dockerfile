FROM python:3.10-slim

ARG DEBIAN_FRONTEND=noninteractive
ARG INSTALL_CUDA_TORCH=0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/root/.cache/huggingface/sentence-transformers

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    antiword \
    build-essential \
    catdoc \
    curl \
    git \
    libglib2.0-0 \
    libgl1 \
    tesseract-ocr \
    tesseract-ocr-vie \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip wheel setuptools \
    && if [ "$INSTALL_CUDA_TORCH" = "1" ]; then \
        python -m pip install torch --index-url https://download.pytorch.org/whl/cu121 ; \
    else \
        python -m pip install torch --index-url https://download.pytorch.org/whl/cpu ; \
    fi \
    && python -m pip install -r /app/requirements.txt

COPY . /app

RUN mkdir -p \
    /app/data/processed \
    /app/data/vector_store \
    /app/data/finetune \
    /app/data/models \
    /app/model \
    /app/checkpoints

EXPOSE 8000

CMD ["bash"]
