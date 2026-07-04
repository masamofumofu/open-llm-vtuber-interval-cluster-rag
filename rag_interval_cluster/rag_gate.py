from dataclasses import dataclass


@dataclass
class RagGateResult:
    use_rag: bool
    reason: str
    matched_keywords: list[str]


RAG_KEYWORDS = [
    "conf.yaml",
    "設定",
    "config",
    "ollama_llm",
    "llm_configs",
    "base_url",
    "model",
    "temperature",
    "keep_alive",
    "unload_at_exit",
    "MioTTS",
    "miotts",
    "VOICEVOX",
    "Open-LLM-VTuber",
    "TinySwallow",
    "tinyswallow",
    "Live2D",
    "model_dict",
    "エラー",
    "ログ",
    "ファイル",
    "どこにある",
    "何が設定されている",
    "確認して",
    "調べて",
]


NON_RAG_PATTERNS = [
    "おはよう",
    "こんにちは",
    "こんばんは",
    "ありがとう",
    "おやすみ",
    "雑談",
    "話そう",
    "元気",
]


def should_use_rag(query: str) -> RagGateResult:
    """
    ユーザー入力に対して、RAG検索を使うべきかを簡易判定する。

    目的:
      - 設定確認、ファイル確認、ログ確認などはRAGへ回す
      - 雑談や通常会話はRAGへ回さない
    """
    normalized = query.strip()
    normalized_lower = normalized.lower()

    if not normalized:
        return RagGateResult(
            use_rag=False,
            reason="empty query",
            matched_keywords=[],
        )

    for pattern in NON_RAG_PATTERNS:
        if pattern.lower() in normalized_lower:
            return RagGateResult(
                use_rag=False,
                reason=f"non-rag pattern matched: {pattern}",
                matched_keywords=[pattern],
            )

    matched_keywords: list[str] = []

    for keyword in RAG_KEYWORDS:
        if keyword.lower() in normalized_lower:
            matched_keywords.append(keyword)

    if matched_keywords:
        return RagGateResult(
            use_rag=True,
            reason="rag keyword matched",
            matched_keywords=matched_keywords,
        )

    return RagGateResult(
        use_rag=False,
        reason="no rag keyword matched",
        matched_keywords=[],
    )