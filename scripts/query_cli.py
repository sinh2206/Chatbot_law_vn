from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (  # noqa: E402
    EMBEDDING_MODEL_NAME,
    EMBEDDINGS_NPY_PATH,
    FAISS_INDEX_PATH,
    GEMINI_API_KEY,
    GEMINI_FALLBACK_ENABLED,
    GEMINI_MODEL,
    LEGAL_DOCUMENT_METADATA_PATH,
    MANIFEST_PATH,
    MAX_CANDIDATE_MULTIPLIER,
    METADATA_PATH,
    MIN_RETRIEVAL_SCORE,
    TOP_K,
)
from scripts.gemini_fallback import (  # noqa: E402
    GeminiFallbackRequest,
    generate_gemini_fallback_answer,
)


def setup_stdout_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


@dataclass
class RetrievedItem:
    score: float
    domain: str
    source_file: str
    chunk_id: str
    chunk_index: int
    text: str
    chunk_type: str = ""
    article_id: str = ""
    article_title: str = ""
    clause_id: str = ""
    is_expired: bool = False
    expiry_date: str = ""
    document_title: str = ""
    document_number: str = ""
    status: str = ""


@dataclass
class LegalDocumentStatus:
    source_file: str
    domain: str
    document_title: str = ""
    document_number: str = ""
    issued_date: str = ""
    effective_date: str = ""
    expiry_date: str = ""
    status: str = ""
    replaced_by: str = ""


@dataclass
class RetrievalDecision:
    use_internal: bool
    reason: str
    usable_results: list[RetrievedItem]
    expired_results: list[RetrievedItem]
    low_confidence_results: list[RetrievedItem]


@dataclass(frozen=True)
class RequestedDocument:
    label: str
    source_file: str
    available: bool
    note: str = ""


KNOWN_REQUESTED_DOCUMENTS: list[dict[str, object]] = [
    {
        "label": "Luật Căn cước 2023",
        "source_file": "CCCD/Luat_26_2023_QH15.txt",
        "patterns": [r"\bluat can cuoc 2023\b", r"\bluat so 26/2023/qh15\b"],
    },
    {
        "label": "Luật Hộ tịch 2014",
        "source_file": "HoTich/Luat_60_2014_QH13.txt",
        "patterns": [r"\bluat ho tich 2014\b", r"\bluat so 60/2014/qh13\b"],
    },
    {
        "label": "Luật Doanh nghiệp 2020",
        "source_file": "DoanhNghiep/Luat_59_2020_QH14.txt",
        "patterns": [r"\bluat doanh nghiep 2020\b", r"\bluat so 59/2020/qh14\b"],
    },
    {
        "label": "Nghị định 01/2021/NĐ-CP về đăng ký doanh nghiệp",
        "source_file": "DoanhNghiep/NghiDinh_01_2021_ND-CP.txt",
        "patterns": [
            r"\bnghi dinh(?: so)? 01/2021(?:/nd-cp|/nđ-cp)?\b",
            r"\b01/2021(?:/nd-cp|/nđ-cp)?\b",
        ],
        "replacement_file": "DoanhNghiep/NghiDinh_168_2025_ND-CP.txt",
        "replacement_label": "Nghị định 168/2025/NĐ-CP thay thế Nghị định 01/2021/NĐ-CP",
    },
    {
        "label": "Nghị định 168/2025/NĐ-CP về đăng ký doanh nghiệp",
        "source_file": "DoanhNghiep/NghiDinh_168_2025_ND-CP.txt",
        "patterns": [r"\bnghi dinh(?: so)? 168/2025(?:/nd-cp|/nđ-cp)?\b"],
    },
    {
        "label": "Luật Phòng, chống rửa tiền 2022",
        "source_file": "NganHang/Luat_07_2022_QH15.txt",
        "patterns": [
            r"\bluat phong chong rua tien 2022\b",
            r"\bluat phong, chong rua tien 2022\b",
            r"\bluat so 07/2022/qh15\b",
        ],
    },
    {
        "label": "Thông tư 09/2023/TT-NHNN",
        "source_file": "NganHang/ThongTu_09_2023_TT-NHNN.txt",
        "patterns": [r"\bthong tu(?: so)? 09/2023/tt-nhnn\b", r"\b09/2023/tt-nhnn\b"],
    },
]


INFERRED_DOCUMENT_RULES: list[dict[str, object]] = [
    {
        "required_groups": [
            {
                "cải chính hộ tịch",
                "cải chính",
                "thay đổi hộ tịch",
                "thông tin nhân thân",
                "nhân thân",
            },
            {
                "người đại diện theo pháp luật",
                "đại diện theo pháp luật",
                "hồ sơ doanh nghiệp",
                "đăng ký doanh nghiệp",
                "công ty",
            },
        ],
        "documents": [
            "Luật Hộ tịch 2014",
            "Luật Doanh nghiệp 2020",
            "Nghị định 168/2025/NĐ-CP về đăng ký doanh nghiệp",
        ],
    },
    {
        "required_groups": [
            {
                "cải chính hộ tịch",
                "cải chính",
                "thay đổi thông tin cá nhân",
                "thông tin cá nhân",
                "thông tin nhân thân",
                "nhân thân",
            },
            {
                "căn cước",
                "thông tin căn cước",
                "số định danh",
                "giấy tờ cá nhân",
            },
            {
                "ngân hàng",
                "tài khoản ngân hàng",
                "xác minh danh tính",
                "xác thực danh tính",
            },
        ],
        "documents": [
            "Luật Căn cước 2023",
            "Luật Phòng, chống rửa tiền 2022",
            "Thông tư 09/2023/TT-NHNN",
        ],
    },
]


QUERY_EXPANSION_RULES: list[tuple[set[str], set[str], str]] = [
    (
        {"CCCD", "HoTich", "Thue", "DatDai"},
        {
            "cải chính hộ tịch",
            "thay đổi thông tin cá nhân",
            "thông tin căn cước",
            "tài khoản ngân hàng",
            "xác minh danh tính",
            "mã số thuế",
            "giấy tờ nhà đất",
            "đứng tên",
        },
        (
            " căn cước số định danh cá nhân thông tin trong Cơ sở dữ liệu quốc gia về dân cư "
            "Cơ sở dữ liệu căn cước cập nhật chỉnh sửa thông tin công dân "
            "cấp đổi thẻ căn cước thay đổi thông tin nhân thân thay đổi cải chính hộ tịch "
            "thông tin đăng ký thuế khớp đúng với Cơ sở dữ liệu quốc gia về dân cư "
            "quyền sử dụng đất giấy chứng nhận thông tin người sử dụng đất "
            "xác minh danh tính nhận biết khách hàng ngân hàng phòng chống rửa tiền"
        ),
    ),
    (
        {"HoTich", "DoanhNghiep"},
        {
            "cải chính hộ tịch",
            "thay đổi hộ tịch",
            "thông tin nhân thân",
            "nhân thân",
            "người đại diện theo pháp luật",
            "hồ sơ doanh nghiệp",
            "giấy tờ cá nhân",
        },
        (
            " thay đổi cải chính hộ tịch bổ sung hộ tịch thông tin hộ tịch "
            "Cơ sở dữ liệu hộ tịch Cơ sở dữ liệu quốc gia về dân cư "
            "giấy khai sinh giấy chứng nhận kết hôn trích lục hộ tịch "
            "giấy tờ pháp lý của cá nhân thẻ Căn cước công dân hộ chiếu "
            "họ tên địa chỉ liên lạc quốc tịch số giấy tờ pháp lý của cá nhân "
            "người đại diện theo pháp luật hồ sơ đăng ký doanh nghiệp "
            "thông báo cập nhật bổ sung thông tin đăng ký doanh nghiệp "
            "Cơ sở dữ liệu quốc gia về đăng ký doanh nghiệp"
        ),
    ),
    (
        {"DoanhNghiep"},
        {"góp vốn", "góp đủ", "tài sản góp vốn", "tài sản", "tiền mặt", "thành lập công ty"},
        (
            " tài sản góp vốn Đồng Việt Nam ngoại tệ tự do chuyển đổi vàng "
            "quyền sử dụng đất quyền sở hữu trí tuệ công nghệ bí quyết kỹ thuật "
            "tài sản khác định giá được bằng Đồng Việt Nam "
            "chuyển quyền sở hữu tài sản góp vốn thanh toán xong "
            "quyền sở hữu hợp pháp đối với tài sản góp vốn đã chuyển sang công ty"
        ),
    ),
    (
        {"DatDai"},
        {"không có giấy tờ", "không có giấy tờ rõ ràng", "chưa có giấy chứng nhận", "chưa có sổ đỏ", "công nhận quyền sử dụng đất", "công nhận lần đầu"},
        (
            " cấp Giấy chứng nhận quyền sử dụng đất quyền sở hữu tài sản gắn liền với đất "
            "hộ gia đình cá nhân đang sử dụng đất ổn định không có giấy tờ về quyền sử dụng đất "
            "không vi phạm pháp luật đất đai không thuộc trường hợp đất được giao không đúng thẩm quyền "
            "Ủy ban nhân dân cấp xã xác nhận không có tranh chấp Điều 138 Luật Đất đai"
        ),
    ),
    (
        {"CCCD"},
        {"đổi thẻ", "cập nhật thông tin", "thay đổi thông tin", "nhân thân", "căn cước"},
        (
            " các trường hợp đổi thẻ căn cước thay đổi thông tin họ chữ đệm tên "
            "đặc điểm nhân dạng xác định lại giới tính quê quán sai sót thông tin "
            "cập nhật điều chỉnh thông tin trong Cơ sở dữ liệu quốc gia về dân cư "
            "Cơ sở dữ liệu căn cước"
        ),
    ),
]


def normalize_query_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def normalize_document_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value)
    without_marks = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    without_marks = without_marks.replace("Đ", "D").replace("đ", "d")
    without_marks = without_marks.lower()
    without_marks = without_marks.replace("nđ-cp", "nd-cp")
    return re.sub(r"\s+", " ", without_marks).strip()


def detect_requested_documents(
    question: str,
    metadata: list[dict[str, object]],
) -> list[RequestedDocument]:
    normalized_question = normalize_document_text(question)
    available_sources = {
        normalize_source_file(str(row.get("source_file", "")))
        for row in metadata
        if str(row.get("source_file", "")).strip()
    }
    documents: list[RequestedDocument] = []
    seen: set[str] = set()
    specs_by_label = {str(spec["label"]): spec for spec in KNOWN_REQUESTED_DOCUMENTS}
    candidate_specs: list[dict[str, object]] = []
    for spec in KNOWN_REQUESTED_DOCUMENTS:
        patterns = [str(item) for item in spec.get("patterns", [])]
        if not any(re.search(pattern, normalized_question) for pattern in patterns):
            continue
        candidate_specs.append(spec)

    for rule in INFERRED_DOCUMENT_RULES:
        required_groups = [
            {normalize_document_text(str(item)) for item in group}
            for group in rule.get("required_groups", [])
        ]
        if not all(
            any(trigger in normalized_question for trigger in group)
            for group in required_groups
        ):
            continue
        for label in rule.get("documents", []):
            spec = specs_by_label.get(str(label))
            if spec is not None:
                candidate_specs.append(spec)

    for spec in candidate_specs:
        label = str(spec["label"])
        if label in seen:
            continue
        source_file = normalize_source_file(str(spec["source_file"]))
        available = source_file in available_sources
        note = ""
        replacement_file = normalize_source_file(str(spec.get("replacement_file", "")))
        replacement_label = str(spec.get("replacement_label", "")).strip()
        if not available and replacement_file and replacement_file in available_sources:
            note = f"Kho nội bộ có {replacement_label}."
        documents.append(
            RequestedDocument(
                label=label,
                source_file=source_file,
                available=available,
                note=note,
            )
        )
        seen.add(label)
    return documents


def document_labels(documents: list[RequestedDocument], available: bool) -> list[str]:
    labels: list[str] = []
    for document in documents:
        if document.available != available:
            continue
        value = f"{document.label} ({document.source_file})"
        if document.note:
            value = f"{value}; {document.note}"
        labels.append(value)
    return labels


def recover_internal_decision(
    decision: RetrievalDecision,
    query: str,
    metadata: list[dict[str, object]],
    min_score: float,
) -> RetrievalDecision:
    if decision.use_internal or not decision.low_confidence_results:
        return decision

    requested_documents = detect_requested_documents(query, metadata)
    available_sources = {
        document.source_file for document in requested_documents if document.available
    }
    if not available_sources:
        return decision

    recovered = [
        item
        for item in decision.low_confidence_results
        if normalize_source_file(item.source_file) in available_sources
        and item.score >= min_score * 0.55
    ]
    if not recovered:
        return decision

    recovered.sort(key=lambda item: item.score, reverse=True)
    return RetrievalDecision(
        use_internal=True,
        reason=(
            "Tìm thấy căn cứ một phần trong kho nội bộ từ các văn bản liên quan "
            "được nhận diện theo nội dung câu hỏi."
        ),
        usable_results=recovered[:5],
        expired_results=decision.expired_results,
        low_confidence_results=[
            item for item in decision.low_confidence_results if item not in recovered
        ],
    )


def expand_query(query: str, domain: str | None) -> str:
    normalized = normalize_query_text(query)
    expansions: list[str] = []
    for domains, triggers, expansion in QUERY_EXPANSION_RULES:
        if domain and domain not in domains:
            continue
        if any(trigger in normalized for trigger in triggers):
            expansions.append(expansion)
    if not expansions:
        return query
    return f"{query}\n{' '.join(expansions)}"


def validate_local_model_dir(model_name: str) -> None:
    model_path = Path(model_name)
    if not model_path.exists() or not model_path.is_dir():
        return
    required = [
        model_path / "modules.json",
        model_path / "config_sentence_transformers.json",
    ]
    missing = [str(item.name) for item in required if not item.exists()]
    if missing:
        raise RuntimeError(
            f"Local model directory is incomplete: {model_path}\n"
            f"Missing required files: {', '.join(missing)}\n"
            "This usually means fine-tuning did not finish, or the directory was created "
            "before the model was saved.\n"
            "Fix it by running scripts/train_embedding.py first, then rebuild the vector store. "
            "If you only want to use the base model, pass "
            "--embedding-model dangvantuan/vietnamese-embedding."
        )


def load_faiss_index(index_path: Path):
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "Missing faiss-cpu. Install requirements first. "
            "If you are on Windows/Python 3.12, use Python 3.10/3.11 for FAISS compatibility.\n"
            f"Try: \"{sys.executable}\" -m pip install faiss-cpu"
        ) from exc

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    return faiss.read_index(str(index_path))


def load_numpy_embeddings(embeddings_path: Path) -> np.ndarray:
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Numpy embeddings not found: {embeddings_path}")
    vectors = np.load(embeddings_path)
    if vectors.ndim != 2:
        raise ValueError("Numpy embeddings must be a 2D array")
    return np.asarray(vectors, dtype=np.float32)


def load_metadata(metadata_path: Path) -> list[dict[str, object]]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    rows: list[dict[str, object]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_source_file(value: str) -> str:
    return value.replace("\\", "/").strip().lstrip("./")


def parse_date(value: str) -> date | None:
    value = value.strip()
    if not value:
        return None
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"]
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def is_status_expired(status: str) -> bool:
    normalized = status.lower().strip().replace(" ", "_").replace("-", "_")
    return normalized in {
        "expired",
        "het_hieu_luc",
        "hethieuluc",
        "inactive",
        "khong_con_hieu_luc",
    }


def load_legal_document_status(
    metadata_csv_path: Path,
) -> dict[str, LegalDocumentStatus]:
    if not metadata_csv_path.exists():
        return {}

    statuses: dict[str, LegalDocumentStatus] = {}
    with metadata_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_file = normalize_source_file(str(row.get("source_file", "")))
            if not source_file:
                continue
            status = LegalDocumentStatus(
                source_file=source_file,
                domain=str(row.get("domain", "")).strip(),
                document_title=str(row.get("document_title", "")).strip(),
                document_number=str(row.get("document_number", "")).strip(),
                issued_date=str(row.get("issued_date", "")).strip(),
                effective_date=str(row.get("effective_date", "")).strip(),
                expiry_date=str(row.get("expiry_date", "")).strip(),
                status=str(row.get("status", "")).strip(),
                replaced_by=str(row.get("replaced_by", "")).strip(),
            )
            statuses[source_file] = status
            statuses[Path(source_file).name] = status
    return statuses


def get_document_status(
    source_file: str,
    statuses: dict[str, LegalDocumentStatus],
) -> LegalDocumentStatus | None:
    normalized = normalize_source_file(source_file)
    return statuses.get(normalized) or statuses.get(Path(normalized).name)


def document_is_expired(status: LegalDocumentStatus | None, today: date | None = None) -> bool:
    if status is None:
        return False
    if is_status_expired(status.status):
        return True
    expiry = parse_date(status.expiry_date)
    if expiry is None:
        return False
    today = today or date.today()
    return today > expiry


def attach_document_status(
    results: list[RetrievedItem],
    statuses: dict[str, LegalDocumentStatus],
) -> list[RetrievedItem]:
    today = date.today()
    enriched: list[RetrievedItem] = []
    for item in results:
        status = get_document_status(item.source_file, statuses)
        if status is None:
            enriched.append(item)
            continue
        enriched.append(
            RetrievedItem(
                score=item.score,
                domain=item.domain,
                source_file=item.source_file,
                chunk_id=item.chunk_id,
                chunk_index=item.chunk_index,
                text=item.text,
                chunk_type=item.chunk_type,
                article_id=item.article_id,
                article_title=item.article_title,
                clause_id=item.clause_id,
                is_expired=document_is_expired(status, today=today),
                expiry_date=status.expiry_date,
                document_title=status.document_title,
                document_number=status.document_number,
                status=status.status,
            )
        )
    return enriched


def format_source(item: RetrievedItem) -> str:
    parts = [item.source_file]
    if item.document_number:
        parts.append(f"so_hieu={item.document_number}")
    if item.document_title:
        parts.append(f"ten={item.document_title}")
    if item.expiry_date:
        parts.append(f"het_hieu_luc={item.expiry_date}")
    if item.status:
        parts.append(f"status={item.status}")
    parts.append(f"score={item.score:.4f}")
    return " | ".join(parts)


def decide_retrieval(
    results: list[RetrievedItem],
    metadata: list[dict[str, object]],
    domain: str | None,
    min_score: float,
) -> RetrievalDecision:
    available_domains = {str(row.get("domain", "")).strip() for row in metadata}
    if domain and domain not in available_domains:
        return RetrievalDecision(
            use_internal=False,
            reason=(
                f"Khong tim thay tai lieu thuoc linh vuc '{domain}' trong vector store. "
                f"Cac linh vuc hien co: {', '.join(sorted(available_domains)) or 'N/A'}."
            ),
            usable_results=[],
            expired_results=[],
            low_confidence_results=[],
        )

    if not results:
        scope = f" linh vuc '{domain}'" if domain else ""
        return RetrievalDecision(
            use_internal=False,
            reason=f"Khong truy xuat duoc doan tai lieu phu hop trong kho noi bo{scope}.",
            usable_results=[],
            expired_results=[],
            low_confidence_results=[],
        )

    expired_results = [item for item in results if item.is_expired]
    non_expired = [item for item in results if not item.is_expired]
    if not non_expired:
        return RetrievalDecision(
            use_internal=False,
            reason=(
                "Cac doan truy xuat duoc deu thuoc van ban da het hieu luc "
                "hoac bi danh dau khong con hieu luc trong metadata."
            ),
            usable_results=[],
            expired_results=expired_results,
            low_confidence_results=[],
        )

    usable_results = [item for item in non_expired if item.score >= min_score]
    low_confidence_results = [item for item in non_expired if item.score < min_score]
    if not usable_results:
        best_score = max(item.score for item in non_expired)
        return RetrievalDecision(
            use_internal=False,
            reason=(
                f"Cac doan noi bo co score thap hon nguong tin cay "
                f"({best_score:.4f} < {min_score:.4f})."
            ),
            usable_results=[],
            expired_results=expired_results,
            low_confidence_results=low_confidence_results,
        )

    return RetrievalDecision(
        use_internal=True,
        reason="Tim thay can cu hop le trong kho noi bo.",
        usable_results=usable_results,
        expired_results=expired_results,
        low_confidence_results=low_confidence_results,
    )


def load_embedder(model_name: str):
    validate_local_model_dir(model_name)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Missing sentence-transformers. Install requirements first.\n"
            f"Try: \"{sys.executable}\" -m pip install sentence-transformers"
        ) from exc
    model = SentenceTransformer(model_name)
    tokenizer_limit = getattr(getattr(model, "tokenizer", None), "model_max_length", None)
    config_limit = getattr(getattr(model[0], "auto_model", None), "config", None)
    if config_limit is not None:
        config_limit = getattr(config_limit, "max_position_embeddings", None)
    limits: list[int] = []
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 100000:
        limits.append(int(tokenizer_limit))
    if isinstance(config_limit, int) and 0 < config_limit < 100000:
        limits.append(max(8, int(config_limit) - 2))
    if limits:
        safe_max_seq_length = min(limits)
        if model.max_seq_length != safe_max_seq_length:
            print(
                f"[WARN] Adjusting model.max_seq_length from {model.max_seq_length} "
                f"to safe limit {safe_max_seq_length} for model: {model_name}"
            )
            model.max_seq_length = safe_max_seq_length
    return model


def search(
    backend: str,
    index,
    metadata: list[dict[str, object]],
    embedder,
    query: str,
    top_k: int,
    domain: str | None,
    candidate_multiplier: int,
) -> list[RetrievedItem]:
    if not query.strip():
        return []

    expanded_query = expand_query(query=query, domain=domain)
    query_vec = embedder.encode(
        [expanded_query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    query_vec = np.asarray(query_vec, dtype=np.float32)

    max_candidates = max(top_k, top_k * max(1, candidate_multiplier))
    if backend == "faiss":
        if domain:
            max_candidates = len(metadata)
        scores, indices = index.search(query_vec, max_candidates)
        score_list = scores[0]
        index_list = indices[0]
    else:
        all_scores = np.dot(index, query_vec[0])
        if domain:
            allowed_indices = np.asarray(
                [
                    i
                    for i, row in enumerate(metadata)
                    if str(row.get("domain", "")) == domain
                ],
                dtype=np.int64,
            )
            if allowed_indices.size == 0:
                return []
            allowed_scores = all_scores[allowed_indices]
            sorted_local = np.argsort(-allowed_scores)[:max_candidates]
            sorted_indices = allowed_indices[sorted_local]
        else:
            sorted_indices = np.argsort(-all_scores)[:max_candidates]
        score_list = all_scores[sorted_indices]
        index_list = sorted_indices

    results: list[RetrievedItem] = []
    for score, idx in zip(score_list, index_list):
        if idx < 0 or idx >= len(metadata):
            continue

        row = metadata[idx]
        row_domain = str(row.get("domain", ""))
        if domain and row_domain != domain:
            continue

        results.append(
            RetrievedItem(
                score=float(score),
                domain=row_domain,
                source_file=str(row.get("source_file", "")),
                chunk_id=str(row.get("chunk_id", "")),
                chunk_index=int(row.get("chunk_index", -1)),
                text=str(row.get("text", "")),
                chunk_type=str(row.get("chunk_type", "")),
                article_id=str(row.get("article_id", "")),
                article_title=str(row.get("article_title", "")),
                clause_id=str(row.get("clause_id", "")),
            )
        )
        if len(results) >= top_k:
            break
    return results


def print_results(results: list[RetrievedItem], show_full: bool, snippet_chars: int) -> None:
    if not results:
        print("Khong tim thay ket qua phu hop.")
        return

    for i, item in enumerate(results, start=1):
        print("\n" + "=" * 90)
        print(
            f"[{i}] score={item.score:.4f} | domain={item.domain} | "
            f"source={item.source_file} | chunk={item.chunk_index}"
        )
        if item.document_number or item.document_title or item.expiry_date or item.status:
            print(
                "document_meta="
                f"number={item.document_number or 'N/A'} | "
                f"title={item.document_title or 'N/A'} | "
                f"expiry={item.expiry_date or 'N/A'} | "
                f"status={item.status or 'N/A'} | "
                f"expired={item.is_expired}"
            )
        if show_full:
            print(item.text)
        else:
            snippet = item.text[:snippet_chars].strip()
            if len(item.text) > snippet_chars:
                snippet += " ..."
            print(snippet)


def print_expired_warning(expired_results: list[RetrievedItem]) -> None:
    if not expired_results:
        return
    print("\n[WARN] Da loai bo cac doan thuoc van ban het hieu luc/khong con hieu luc:")
    seen: set[str] = set()
    for item in expired_results:
        key = item.source_file
        if key in seen:
            continue
        seen.add(key)
        print(f"- {format_source(item)}")


def build_fallback_notice(decision: RetrievalDecision) -> str:
    if decision.expired_results:
        item = decision.expired_results[0]
        label = item.document_title or item.document_number or item.source_file
        suffix = f" ({item.expiry_date})" if item.expiry_date else ""
        return f"tài liệu {label}{suffix} hết hạn, cần cập nhật"
    return "không tìm thấy tài liệu làm căn cứ cho câu hỏi trong kho nội bộ"


def run_query(
    query: str,
    backend: str,
    index,
    metadata: list[dict[str, object]],
    embedder,
    domain: str | None,
    top_k: int,
    candidate_multiplier: int,
    document_statuses: dict[str, LegalDocumentStatus],
    min_score: float,
    show_full: bool,
    snippet_chars: int,
    gemini_fallback: bool,
    gemini_model: str,
    gemini_api_key: str,
) -> None:
    results = search(
        backend=backend,
        index=index,
        metadata=metadata,
        embedder=embedder,
        query=query,
        top_k=top_k,
        domain=domain,
        candidate_multiplier=candidate_multiplier,
    )
    results = attach_document_status(results, document_statuses)
    decision = decide_retrieval(
        results=results,
        metadata=metadata,
        domain=domain,
        min_score=min_score,
    )
    decision = recover_internal_decision(
        decision=decision,
        query=query,
        metadata=metadata,
        min_score=min_score,
    )

    if decision.use_internal:
        print("\n[LOCAL RAG] Su dung can cu tu data/processed.")
        print_expired_warning(decision.expired_results)
        print_results(
            decision.usable_results,
            show_full=show_full,
            snippet_chars=snippet_chars,
        )
        requested_documents = detect_requested_documents(query, metadata)
        missing_documents = document_labels(requested_documents, available=False)
        available_documents = document_labels(requested_documents, available=True)
        if not missing_documents:
            return
        print("\n[API SUPPLEMENT REQUIRED]")
        print("Chi goi Gemini de bo sung van ban con thieu:")
        for item in missing_documents:
            print(f"- {item}")
        if not gemini_fallback:
            print("\nGemini fallback dang tat nen khong tra cuu bo sung.")
            return
        try:
            supplement_result = generate_gemini_fallback_answer(
                request=GeminiFallbackRequest(
                    question=query,
                    reason=(
                        "RAG noi bo da co mot phan can cu, nhung cau hoi yeu cau "
                        "mot so van ban chua co trong vector store."
                    ),
                    domain=domain,
                    available_local_documents=available_documents,
                    missing_documents=missing_documents,
                    local_answer=(
                        "Kho noi bo da tra ve can cu. Xem cac ket qua LOCAL RAG o tren."
                    ),
                ),
                api_key=gemini_api_key,
                model_name=gemini_model,
            )
        except Exception as exc:
            print(f"\n[ERROR] Khong the goi Gemini de bo sung tai lieu: {exc}")
            return
        print("\n" + "=" * 90)
        print(supplement_result.answer)
        if supplement_result.sources:
            print("\nTai lieu/can cu tham khao tu Gemini grounding:")
            for source in supplement_result.sources:
                title = source.get("title") or source.get("source") or "N/A"
                url = source.get("url") or ""
                print(f"- {title}{' | ' + url if url else ''}")
        return

    print("\n[FALLBACK REQUIRED]")
    fallback_notice = build_fallback_notice(decision)
    print(fallback_notice)
    print(decision.reason)
    if decision.expired_results:
        print("Tai lieu noi bo bi loai do het hieu luc:")
        for item in decision.expired_results:
            print(f"- {format_source(item)}")
    if decision.low_confidence_results:
        print("Tai lieu noi bo bi loai do score thap:")
        for item in decision.low_confidence_results[:5]:
            print(f"- {format_source(item)}")

    if not gemini_fallback:
        print(
            "\nGemini fallback dang tat. Bat bang --gemini-fallback "
            "hoac GEMINI_FALLBACK_ENABLED=true va thiet lap GEMINI_API_KEY."
        )
        return

    print("\n[GEMINI FALLBACK] Dang goi Gemini vi khong co can cu noi bo hop le.")
    request = GeminiFallbackRequest(
        question=query,
        reason=decision.reason,
        domain=domain,
        fallback_notice=fallback_notice,
        expired_sources=[format_source(item) for item in decision.expired_results],
        low_confidence_sources=[
            format_source(item) for item in decision.low_confidence_results[:5]
        ],
    )
    try:
        fallback_result = generate_gemini_fallback_answer(
            request=request,
            api_key=gemini_api_key,
            model_name=gemini_model,
        )
    except Exception as exc:
        print(f"\n[ERROR] Khong the goi Gemini fallback: {exc}")
        return

    print("\n" + "=" * 90)
    print(fallback_result.answer)
    if fallback_result.sources:
        print("\nTai lieu/can cu tham khao tu Gemini grounding:")
        for source in fallback_result.sources:
            title = source.get("title") or source.get("source") or "N/A"
            url = source.get("url") or ""
            print(f"- {title}{' | ' + url if url else ''}")


def read_manifest(manifest_path: Path) -> dict[str, object]:
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> None:
    setup_stdout_utf8()
    parser = argparse.ArgumentParser(description="Offline RAG query CLI using local vector index.")
    parser.add_argument("--query", default="")
    parser.add_argument("--domain", default="")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--embedding-model",
        default="",
        help="Embedding model path/name. Empty = use manifest embedding_model, then config default.",
    )
    parser.add_argument("--index-path", default=str(FAISS_INDEX_PATH))
    parser.add_argument("--metadata-path", default=str(METADATA_PATH))
    parser.add_argument("--manifest-path", default=str(MANIFEST_PATH))
    parser.add_argument(
        "--legal-metadata-path",
        default=str(LEGAL_DOCUMENT_METADATA_PATH),
        help="CSV metadata with source_file/domain/expiry_date/status for expiry checks.",
    )
    parser.add_argument("--show-full", action="store_true")
    parser.add_argument("--snippet-chars", type=int, default=350)
    parser.add_argument("--candidate-multiplier", type=int, default=MAX_CANDIDATE_MULTIPLIER)
    parser.add_argument(
        "--min-score",
        type=float,
        default=MIN_RETRIEVAL_SCORE,
        help="Minimum cosine score required to trust local retrieval.",
    )
    parser.add_argument(
        "--gemini-fallback",
        action="store_true",
        default=GEMINI_FALLBACK_ENABLED,
        help="Call Gemini only when local RAG cannot provide valid evidence.",
    )
    parser.add_argument(
        "--no-gemini-fallback",
        action="store_false",
        dest="gemini_fallback",
        help="Disable Gemini fallback even if GEMINI_FALLBACK_ENABLED=true.",
    )
    parser.add_argument("--gemini-model", default=GEMINI_MODEL)
    args = parser.parse_args()

    manifest = read_manifest(Path(args.manifest_path).resolve())
    backend = str(manifest.get("index_backend", "faiss")).lower()

    if backend == "numpy":
        embeddings_path = Path(
            str(manifest.get("embeddings_npy_path", EMBEDDINGS_NPY_PATH))
        ).resolve()
        index = load_numpy_embeddings(embeddings_path)
        index_display = str(embeddings_path)
    else:
        backend = "faiss"
        index_path = Path(args.index_path).resolve()
        index = load_faiss_index(index_path)
        index_display = str(index_path)

    metadata = load_metadata(Path(args.metadata_path).resolve())
    legal_metadata_path = Path(args.legal_metadata_path).resolve()
    document_statuses = load_legal_document_status(legal_metadata_path)
    manifest_model = str(manifest.get("embedding_model", "")).strip()
    embedding_model = args.embedding_model.strip() or manifest_model or EMBEDDING_MODEL_NAME
    embedder = load_embedder(embedding_model)

    print("=== LOCAL RAG QUERY CLI ===")
    if manifest:
        print(f"Chunks: {manifest.get('total_chunks', 'N/A')}")
        print(f"Embedding model in manifest: {manifest.get('embedding_model', 'N/A')}")
    print(f"Embedding model in use: {embedding_model}")
    print(f"Index backend: {backend}")
    print(f"Index data: {index_display}")
    print(f"Metadata: {args.metadata_path}")
    print(f"Legal metadata: {legal_metadata_path} ({len(document_statuses)} keys)")
    print(f"Min local score: {args.min_score:.4f}")
    print(
        "Gemini fallback: "
        f"{'enabled' if args.gemini_fallback else 'disabled'} | "
        f"model={args.gemini_model}"
    )

    domain = args.domain.strip() or None

    if args.query.strip():
        run_query(
            query=args.query,
            backend=backend,
            index=index,
            metadata=metadata,
            embedder=embedder,
            top_k=max(1, args.top_k),
            domain=domain,
            candidate_multiplier=max(1, args.candidate_multiplier),
            document_statuses=document_statuses,
            min_score=args.min_score,
            show_full=args.show_full,
            snippet_chars=max(50, args.snippet_chars),
            gemini_fallback=args.gemini_fallback,
            gemini_model=args.gemini_model,
            gemini_api_key=GEMINI_API_KEY,
        )
        return

    print("\nNhap cau hoi (go 'exit' de thoat).")
    while True:
        query = input("\nBan: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Ket thuc.")
            return

        run_query(
            query=query,
            backend=backend,
            index=index,
            metadata=metadata,
            embedder=embedder,
            top_k=max(1, args.top_k),
            domain=domain,
            candidate_multiplier=max(1, args.candidate_multiplier),
            document_statuses=document_statuses,
            min_score=args.min_score,
            show_full=args.show_full,
            snippet_chars=max(50, args.snippet_chars),
            gemini_fallback=args.gemini_fallback,
            gemini_model=args.gemini_model,
            gemini_api_key=GEMINI_API_KEY,
        )


if __name__ == "__main__":
    main()
