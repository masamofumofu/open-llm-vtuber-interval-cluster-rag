import re


def extract_setting_value(answer: str, key: str) -> str | None:
    """
    RAG回答の箇条書きから設定値を取り出す。

    例:
      - model: tinyswallow-vtuber:latest
    """
    pattern = rf"^\s*-\s*{re.escape(key)}\s*:\s*(.+?)\s*$"

    for line in answer.splitlines():
        match = re.match(pattern, line)
        if match:
            value = match.group(1).strip()
            value = value.strip("'\"")
            return value

    return None


def shorten_for_tts(text: str, max_chars: int = 40) -> str:
    """
    MioTTSへ渡すため、短く安全な1文にする。
    """
    text = text.strip()

    if len(text) <= max_chars:
        return text

    cut = text[:max_chars]

    last_period = max(
        cut.rfind("。"),
        cut.rfind("！"),
        cut.rfind("？"),
        cut.rfind("."),
        cut.rfind("!"),
        cut.rfind("?"),
    )

    if last_period >= 8:
        return cut[: last_period + 1]

    return cut.rstrip("、，,. ") + "。"


def build_rag_tts_text(query: str, answer: str, max_chars: int = 40) -> str:
    """
    RAG回答全文から、音声読み上げ用の短文を作る。

    方針:
      - 画面にはRAG回答全文を表示する
      - 音声では短い確認文だけ読む
    """
    query_text = str(query or "")
    answer_text = str(answer or "")

    if not answer_text.strip():
        return ""

    if "根拠情報からは確認できません" in answer_text:
        return "根拠情報からは確認できませんでした。"

    if "ollama_llm" in query_text.lower() or "ollama_llm" in answer_text.lower():
        model = extract_setting_value(answer_text, "model")

        if model:
            return shorten_for_tts(
                f"モデルは{model}です。",
                max_chars=max_chars,
            )

        return "ollama_llm設定を確認しました。"

    if "設定" in query_text or "config" in query_text.lower():
        return "設定を確認しました。"

    if "ログ" in query_text or "エラー" in query_text:
        return "ログ情報を確認しました。"

    return shorten_for_tts(
        "RAGで確認しました。画面に結果を表示しています。",
        max_chars=max_chars,
    )