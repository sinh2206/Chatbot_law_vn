from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - graceful fallback when dependencies are not installed yet
    def load_dotenv() -> bool:
        return False

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "Multi-Agent")))
VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", str(BASE_DIR / "vector_store")))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_docs")

METADATA_DIR = Path(os.getenv("METADATA_DIR", str(BASE_DIR / "metadata")))
LEGAL_METADATA_FILE = Path(
    os.getenv("LEGAL_METADATA_FILE", str(METADATA_DIR / "legal_documents_metadata.csv"))
)
LOGS_DIR = Path(os.getenv("LOGS_DIR", str(BASE_DIR / "logs")))
ADMIN_ALERT_LOG_FILE = Path(
    os.getenv("ADMIN_ALERT_LOG_FILE", str(LOGS_DIR / "admin_alerts.log"))
)

FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", str(BASE_DIR / "frontend")))
DASHBOARD_DIR = Path(os.getenv("DASHBOARD_DIR", str(BASE_DIR / "dashboard")))
TESTSUITE_DIR = Path(os.getenv("TESTSUITE_DIR", str(BASE_DIR / "testsuite")))
TELEMETRY_DB_FILE = Path(os.getenv("TELEMETRY_DB_FILE", str(LOGS_DIR / "telemetry.db")))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", str(BASE_DIR / "reports")))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", str(TOP_K)))

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "gemini").strip().lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
SENTENCE_TRANSFORMER_MODEL = os.getenv(
    "SENTENCE_TRANSFORMER_MODEL",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
INTENT_TEMPERATURE = float(os.getenv("INTENT_TEMPERATURE", "0.0"))
MERGE_TEMPERATURE = float(os.getenv("MERGE_TEMPERATURE", "0.1"))
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1024"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

ALLOWED_EXTENSIONS = {".txt", ".doc", ".docx"}

WEB_FALLBACK_ENABLED = os.getenv("WEB_FALLBACK_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}
WEB_SEARCH_BACKEND = os.getenv("WEB_SEARCH_BACKEND", "ddgs")
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "8"))
TRUSTED_WEB_DOMAINS = [
    domain.strip().lower()
    for domain in os.getenv(
        "TRUSTED_WEB_DOMAINS",
        "chinhphu.vn,congbao.chinhphu.vn,vbpl.vn,thuvienphapluat.vn,moj.gov.vn"
    ).split(",")
    if domain.strip()
]

EXPIRY_SCAN_INTERVAL_HOURS = int(os.getenv("EXPIRY_SCAN_INTERVAL_HOURS", "24"))

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "")
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_SENDER = os.getenv("SMTP_SENDER", SMTP_USERNAME or "")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").strip().lower() in {"1", "true", "yes"}
SMTP_USE_SSL = os.getenv("SMTP_USE_SSL", "false").strip().lower() in {"1", "true", "yes"}

CORS_ALLOW_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:5500,http://127.0.0.1:5500,http://localhost:8000,http://127.0.0.1:8000,http://localhost:8501,http://127.0.0.1:8501",
    ).split(",")
    if origin.strip()
]

ALERT_MIN_ACCURACY = float(os.getenv("ALERT_MIN_ACCURACY", "75"))
ALERT_MAX_AVG_RESPONSE_MS = int(os.getenv("ALERT_MAX_AVG_RESPONSE_MS", "12000"))
ALERT_MAX_FALLBACK_RATE = float(os.getenv("ALERT_MAX_FALLBACK_RATE", "0.35"))

TEST_DATASET_FILE = Path(
    os.getenv("TEST_DATASET_FILE", str(TESTSUITE_DIR / "test_cases.json"))
)
TEST_PASS_SCORE = float(os.getenv("TEST_PASS_SCORE", "6.0"))
TEST_USE_LLM_GRADER = os.getenv("TEST_USE_LLM_GRADER", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}
TEST_RUNNER_MODE = os.getenv("TEST_RUNNER_MODE", "api").strip().lower()
TEST_API_BASE_URL = os.getenv("TEST_API_BASE_URL", "http://localhost:8000").strip().rstrip("/")
TEST_API_TIMEOUT_SECONDS = int(os.getenv("TEST_API_TIMEOUT_SECONDS", "180"))
TEST_ALLOW_DIRECT_FALLBACK = os.getenv("TEST_ALLOW_DIRECT_FALLBACK", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}
TEST_REPORT_SEND_EMAIL = os.getenv("TEST_REPORT_SEND_EMAIL", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}

DOMAIN_LABELS = {
    "DoanhNghiep": "Doanh nghiệp",
    "HoTich": "Hộ tịch",
    "CCCD": "Căn cước công dân",
    "DatDai": "Đất đai",
    "Thue": "Thuế",
}

DOMAIN_KEYWORDS = {
    "DoanhNghiep": [
        "doanh nghiệp",
        "công ty",
        "đăng ký kinh doanh",
        "giấy chứng nhận đăng ký doanh nghiệp",
    ],
    "HoTich": [
        "hộ tịch",
        "khai sinh",
        "kết hôn",
        "khai tử",
        "cải chính",
    ],
    "CCCD": [
        "căn cước",
        "cccd",
        "chứng minh nhân dân",
        "định danh cá nhân",
        "chip",
    ],
    "DatDai": [
        "đất đai",
        "sổ đỏ",
        "quyền sử dụng đất",
        "thu hồi đất",
        "chuyển mục đích sử dụng đất",
    ],
    "Thue": [
        "thuế",
        "khấu trừ",
        "kê khai",
        "hóa đơn",
        "nghĩa vụ tài chính",
        "thuế giá trị gia tăng",
    ],
}


def ensure_directories() -> None:
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    TESTSUITE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def require_gemini_key() -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Missing Gemini API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env"
        )
    return GEMINI_API_KEY
