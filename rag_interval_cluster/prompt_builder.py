from dataclasses import dataclass

from .chunker import Chunk
from .hierarchical_cluster import ClusterSummary


@dataclass
class PromptBuildResult:
    query: str
    prompt: str
    context_text: str
    used_cluster_count: int
    used_chunk_count: int


def confidence_label(score: float, width: float) -> str:
    """
    距離スコアと区間幅から、簡易的な信頼度ラベルを返す。

    TF-IDF版では距離が全体的に高めに出るため、
    厳密な絶対値ではなく、まずは目安として使う。
    """
    if width <= 0.05:
        return "高"
    if width <= 0.10:
        return "中"
    return "補助"


def shorten_text(text: str, max_chars: int) -> str:
    text = text.strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "\n...（以下省略）"


def get_neighbor_chunks(
    all_chunks: list[Chunk],
    center_chunk: Chunk,
    neighbor_before: int = 1,
    neighbor_after: int = 6,
) -> list[Chunk]:
    """
    代表チャンクの前後チャンクを取得する。

    YAML設定では、見出しチャンクの直後に詳細設定が続くことが多いため、
    後続チャンクを多めに含める。
    """
    same_file_chunks = [
        chunk
        for chunk in all_chunks
        if chunk.file_path == center_chunk.file_path
    ]

    same_file_chunks.sort(key=lambda chunk: chunk.chunk_index)

    center_pos = None

    for index, chunk in enumerate(same_file_chunks):
        if chunk.chunk_id == center_chunk.chunk_id:
            center_pos = index
            break

    if center_pos is None:
        return [center_chunk]

    start = max(0, center_pos - neighbor_before)
    end = min(len(same_file_chunks), center_pos + neighbor_after + 1)

    return same_file_chunks[start:end]


def build_expanded_chunk_text(
    chunks: list[Chunk],
    max_chars_per_cluster: int,
) -> str:
    """
    複数チャンクを、プロンプト用にまとめる。
    """
    blocks: list[str] = []

    for chunk in chunks:
        block = f"""--- {chunk.chunk_id} / 文字範囲: {chunk.start_char} - {chunk.end_char} ---
{chunk.text.strip()}
"""
        blocks.append(block)

    merged_text = "\n".join(blocks)

    return shorten_text(
        merged_text,
        max_chars=max_chars_per_cluster,
    )


def build_context_text(
    summaries: list[ClusterSummary],
    all_chunks: list[Chunk] | None = None,
    max_chars_per_cluster: int = 2500,
    neighbor_before: int = 1,
    neighbor_after: int = 6,
) -> tuple[str, int]:
    """
    クラスタ代表チャンクを、LLMに渡しやすい根拠テキストへ整形する。

    all_chunks が渡された場合は、代表チャンクの前後も含める。
    """
    blocks: list[str] = []
    used_chunk_ids: set[str] = set()

    for index, summary in enumerate(summaries, start=1):
        rep = summary.representative
        interval = rep.interval
        chunk = rep.chunk

        confidence = confidence_label(
            score=interval.score,
            width=interval.width,
        )

        if all_chunks is not None:
            expanded_chunks = get_neighbor_chunks(
                all_chunks=all_chunks,
                center_chunk=chunk,
                neighbor_before=neighbor_before,
                neighbor_after=neighbor_after,
            )
        else:
            expanded_chunks = [chunk]

        for expanded_chunk in expanded_chunks:
            used_chunk_ids.add(expanded_chunk.chunk_id)

        chunk_text = build_expanded_chunk_text(
            chunks=expanded_chunks,
            max_chars_per_cluster=max_chars_per_cluster,
        )

        expanded_ids = ", ".join(
            expanded_chunk.chunk_id
            for expanded_chunk in expanded_chunks
        )

        block = f"""【根拠クラスタ{index}】
クラスタID: {summary.cluster_id}
クラスタ内チャンク数: {summary.size}
信頼度: {confidence}
距離区間: [{interval.low:.3f}, {interval.high:.3f}]
区間幅: {interval.width:.3f}
スコア: {interval.score:.3f}
ファイル: {chunk.file_name}
代表チャンクID: {chunk.chunk_id}
展開チャンクID: {expanded_ids}
代表文字範囲: {chunk.start_char} - {chunk.end_char}

内容:
{chunk_text}
"""

        blocks.append(block)

    return "\n".join(blocks), len(used_chunk_ids)


def build_rag_prompt(
    query: str,
    summaries: list[ClusterSummary],
    all_chunks: list[Chunk] | None = None,
    max_chars_per_cluster: int = 2500,
    neighbor_before: int = 1,
    neighbor_after: int = 6,
) -> PromptBuildResult:
    """
    LLMへ渡す最終プロンプトを作る。
    """
    context_text, used_chunk_count = build_context_text(
        summaries=summaries,
        all_chunks=all_chunks,
        max_chars_per_cluster=max_chars_per_cluster,
        neighbor_before=neighbor_before,
        neighbor_after=neighbor_after,
    )

    if context_text.strip():
        prompt = f"""あなたは、与えられた根拠情報をもとに回答するアシスタントです。

以下のルールを守ってください。

- 根拠情報に書かれている内容を優先してください。
- 根拠情報にない内容は推測で断定しないでください。
- 不明な場合は「根拠情報からは確認できません」と答えてください。
- ファイル名、設定名、項目名が分かる場合は明示してください。
- 回答は日本語で、簡潔かつ具体的に書いてください。

# 根拠情報

{context_text}

# ユーザー質問

{query}

# 回答
"""
    else:
        prompt = f"""あなたは日本語で回答するアシスタントです。

次の質問に回答してください。
ただし、参照できる根拠情報は見つかりませんでした。
不明な点は推測で断定しないでください。

# ユーザー質問

{query}

# 回答
"""

    return PromptBuildResult(
        query=query,
        prompt=prompt,
        context_text=context_text,
        used_cluster_count=len(summaries),
        used_chunk_count=used_chunk_count,
    )