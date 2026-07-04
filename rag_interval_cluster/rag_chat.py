from dataclasses import dataclass
from pathlib import Path

from rag_interval_cluster.rag_answer import answer_with_rag
from rag_interval_cluster.rag_gate import RagGateResult, should_use_rag


@dataclass
class RagChatResult:
    query: str
    answer: str | None
    use_rag: bool
    gate_result: RagGateResult
    source: str
    error: str | None = None


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def expand_query_for_rag(query: str) -> str:
    """
    短いユーザー質問を、RAG検索と回答生成に向いた質問へ補強する。

    例:
      conf.yamlのollama_llm設定は？
    を
      conf.yamlのollama_llm設定は？ base_url、model、temperature、keep_alive、unload_at_exit を答えてください。
    に変換する。
    """
    query_text = str(query).strip()
    query_lower = query_text.lower()

    if "ollama_llm" in query_lower:
        required_terms = [
            "base_url",
            "model",
            "temperature",
            "keep_alive",
            "unload_at_exit",
        ]

        missing_terms = [
            term for term in required_terms
            if term.lower() not in query_lower
        ]

        if missing_terms:
            return (
                query_text
                + " "
                + "base_url、model、temperature、keep_alive、unload_at_exit を答えてください。"
            )

    return query_text


def chat_with_optional_rag(
    query: str,
    model: str = "tinyswallow-vtuber:latest",
    base_url: str = "http://localhost:11434/v1",
    temperature: float = 0.2,
    max_tokens: int = 300,
    save_logs: bool = True,
) -> RagChatResult:
    """
    Open-LLM-VTuber組み込み前の会話入口。

    RAGが必要な質問:
      answer_with_rag() で回答する

    RAGが不要な質問:
      ここでは通常LLM処理に渡す前提で answer=None を返す
      実際のOpen-LLM-VTuber組み込み時は、通常のLLM経路に流す
    """
    gate_result = should_use_rag(query)

    if not gate_result.use_rag:
        result = RagChatResult(
            query=query,
            answer=None,
            use_rag=False,
            gate_result=gate_result,
            source="normal_llm",
            error=None,
        )

        if save_logs:
            save_text(
                Path("rag_interval_cluster/logs/last_rag_chat_route.txt"),
                (
                    "source=normal_llm\n"
                    f"query={query}\n"
                    f"reason={gate_result.reason}\n"
                    f"matched_keywords={gate_result.matched_keywords}\n"
                ),
            )

        return result

    try:
        expanded_query = expand_query_for_rag(query)

        rag_answer = answer_with_rag(
            query=expanded_query,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            save_logs=save_logs,
        )

        result = RagChatResult(
            query=query,
            answer=rag_answer.answer,
            use_rag=True,
            gate_result=gate_result,
            source="rag",
            error=None,
        )

        if save_logs:
            save_text(
                Path("rag_interval_cluster/logs/last_rag_chat_route.txt"),
                (
                    "source=rag\n"
                    f"query={query}\n"
                    f"expanded_query={expanded_query}\n"
                    f"reason={gate_result.reason}\n"
                    f"matched_keywords={gate_result.matched_keywords}\n"
                    f"top_representative_chunk_id={rag_answer.top_representative_chunk_id}\n"
                ),
            )

        return result

    except Exception as exc:
        # RAG側で失敗しても、Open-LLM-VTuber全体を止めないための保険。
        # 組み込み時は、この場合に通常LLM経路へフォールバックさせる。
        result = RagChatResult(
            query=query,
            answer=None,
            use_rag=True,
            gate_result=gate_result,
            source="rag_error",
            error=str(exc),
        )

        if save_logs:
            save_text(
                Path("rag_interval_cluster/logs/last_rag_chat_route.txt"),
                (
                    "source=rag_error\n"
                    f"query={query}\n"
                    f"reason={gate_result.reason}\n"
                    f"matched_keywords={gate_result.matched_keywords}\n"
                    f"error={exc}\n"
                ),
            )

        return result