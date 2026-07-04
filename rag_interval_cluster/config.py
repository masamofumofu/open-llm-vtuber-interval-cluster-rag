from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DOCS_DIR = BASE_DIR / "docs"
INDEX_DIR = BASE_DIR / "index"
LOGS_DIR = BASE_DIR / "logs"

SUPPORTED_EXTENSIONS = {
    ".sql",
    ".txt",
    ".md",
    ".yaml",
    ".yml",
    ".py",
    ".bat",
}

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MIN_CHUNK_CHARS = 30

TOP_K = 30
CLUSTER_COUNT = 5
MAX_CONTEXT_CHUNKS = 6

LAMBDA_WIDTH = 0.5