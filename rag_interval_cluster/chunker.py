from dataclasses import dataclass

from .config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_CHARS
from .document_loader import Document


@dataclass
class Chunk:
    chunk_id: str
    file_name: str
    file_path: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int


def split_text_by_chars(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[tuple[int, int, str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[tuple[int, int, str]] = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = text[start:end].strip()

        if len(chunk_text) >= MIN_CHUNK_CHARS:
            chunks.append((start, end, chunk_text))

        if end >= text_length:
            break

        start = end - chunk_overlap

    return chunks


def count_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def is_yaml_key_line(line: str, max_indent: int = 6) -> bool:
    """
    YAMLのキー行らしきものを判定する。

    max_indent=6 にすることで、トップレベルだけでなく、
    llm_configs: や ollama_llm: のような中階層も拾う。
    """
    if not line.strip():
        return False

    if line.lstrip().startswith("#"):
        return False

    if line.lstrip().startswith("-"):
        return False

    indent = count_indent(line)

    if indent > max_indent:
        return False

    stripped = line.strip()

    if ":" not in stripped:
        return False

    key = stripped.split(":", 1)[0].strip()

    if not key:
        return False

    # URLなどを誤ってキー扱いしないための簡易除外
    if "://" in stripped:
        return False

    return True


def is_yaml_comment_header(line: str) -> bool:
    """
    conf.yaml 内の大きなコメント見出しを分割点として扱う。

    例:
      # ============== Prompts ==============
      # =================== LLM Backend Settings ===================
      # =================== TTS Settings ===================
    """
    stripped = line.strip()

    if not stripped.startswith("#"):
        return False

    header_markers = [
        "====",
        "----",
        "LLM Backend",
        "TTS",
        "ASR",
        "Agent",
        "Prompts",
        "Translate",
    ]

    return any(marker in stripped for marker in header_markers)


def should_start_new_yaml_chunk(line: str) -> bool:
    """
    YAMLチャンクの開始候補を判定する。
    """
    return is_yaml_comment_header(line) or is_yaml_key_line(line, max_indent=6)


def split_long_yaml_section(
    start_char: int,
    end_char: int,
    section_text: str,
) -> list[tuple[int, int, str]]:
    """
    1セクションが長すぎる場合は、通常の文字数分割で補助分割する。
    """
    if len(section_text) <= CHUNK_SIZE * 2:
        return [(start_char, end_char, section_text)]

    sub_chunks = split_text_by_chars(section_text)

    results: list[tuple[int, int, str]] = []

    for sub_start, sub_end, sub_text in sub_chunks:
        results.append(
            (
                start_char + sub_start,
                start_char + sub_end,
                sub_text,
            )
        )

    return results


def split_yaml_by_semantic_sections(text: str) -> list[tuple[int, int, str]]:
    """
    YAMLを設定ファイル向けに意味単位で分割する。

    分割点:
      - トップレベルキー
      - インデント6以内の中階層キー
      - 大きなコメント見出し

    これにより、llm_configs や ollama_llm 周辺を
    文字数分割より見つけやすくする。
    """
    lines = text.splitlines(keepends=True)

    if not lines:
        return []

    sections: list[tuple[int, int, str]] = []

    current_start_char = 0
    current_lines: list[str] = []
    char_pos = 0

    for line in lines:
        line_start = char_pos
        line_end = char_pos + len(line)

        starts_new = should_start_new_yaml_chunk(line)

        if starts_new and current_lines:
            section_text = "".join(current_lines).strip()

            if len(section_text) >= MIN_CHUNK_CHARS:
                sections.extend(
                    split_long_yaml_section(
                        start_char=current_start_char,
                        end_char=line_start,
                        section_text=section_text,
                    )
                )

            current_lines = [line]
            current_start_char = line_start
        else:
            if not current_lines:
                current_start_char = line_start

            current_lines.append(line)

        char_pos = line_end

    if current_lines:
        section_text = "".join(current_lines).strip()

        if len(section_text) >= MIN_CHUNK_CHARS:
            sections.extend(
                split_long_yaml_section(
                    start_char=current_start_char,
                    end_char=char_pos,
                    section_text=section_text,
                )
            )

    if len(sections) <= 1:
        return split_text_by_chars(text)

    return sections


def split_document_text(document: Document) -> list[tuple[int, int, str]]:
    """
    ファイル種別に応じてチャンク分割方法を変える。
    """
    if document.extension in {".yaml", ".yml"}:
        return split_yaml_by_semantic_sections(document.text)

    return split_text_by_chars(document.text)


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []

    for document in documents:
        split_chunks = split_document_text(document)

        for index, (start, end, text) in enumerate(split_chunks):
            chunk_id = f"{document.file_name}::chunk_{index:04d}"

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    file_name=document.file_name,
                    file_path=document.file_path,
                    chunk_index=index,
                    text=text,
                    start_char=start,
                    end_char=end,
                )
            )

    return chunks