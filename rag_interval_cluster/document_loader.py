from dataclasses import dataclass
from pathlib import Path

from .config import DOCS_DIR, SUPPORTED_EXTENSIONS


@dataclass
class Document:
    file_name: str
    file_path: str
    extension: str
    text: str


def read_text_safely(path: Path) -> str:
    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp932",
        "shift_jis",
    ]

    last_error = None

    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Could not decode {path}: {last_error}",
    )


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    return text


def load_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    docs_dir = Path(docs_dir)

    if not docs_dir.exists():
        raise FileNotFoundError(f"docs_dir not found: {docs_dir}")

    documents: list[Document] = []

    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = read_text_safely(path)
        text = normalize_text(text)

        if not text.strip():
            continue

        documents.append(
            Document(
                file_name=path.name,
                file_path=str(path),
                extension=path.suffix.lower(),
                text=text,
            )
        )

    return documents