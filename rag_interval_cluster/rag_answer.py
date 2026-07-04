import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_interval_cluster.ollama_client import OllamaChatClient
from rag_interval_cluster.retriever import IntervalClusterRetriever


@dataclass
class RagAnswerResult:
    query: str
    answer: str
    raw_answer: str
    prompt: str
    prompt_chars: int
    used_cluster_count: int
    used_chunk_count: int
    top_representative_chunk_id: str | None
    retrieval_score: float | None
    retrieval_reliable: bool


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_answer_prompt(original_prompt: str) -> str:
    """
    RAGプロンプトに、回答形式の制約を追加する。

    重要:
    build_rag_prompt() が末尾に "# 回答" を持っているため、
    追加ルールは "# 回答" の前に挿入する。
    """

    format_rule = """# 回答形式の追加ルール

- JSON形式で回答しないでください。
- コードブロックで回答しないでください。
- Pythonの辞書形式で回答しないでください。
- 箇条書きの日本語説明で回答してください。
- 設定名と設定値を明確に対応させてください。
- 根拠情報にある値だけを使ってください。
- 次の形式に合わせてください。

conf.yaml の ollama_llm 設定は以下の通りです。

- base_url: ...
- model: ...
- temperature: ...
- keep_alive: ...
- unload_at_exit: ...
"""

    marker = "# 回答"

    if marker in original_prompt:
        return original_prompt.replace(marker, format_rule + "\n\n" + marker, 1)

    return original_prompt + "\n\n" + format_rule


def strip_code_fence(text: str) -> str:
    """
    ```json ... ``` や ``` ... ``` を除去する。
    """
    text = text.strip()

    fence_pattern = r"^```(?:json|python|yaml|yml)?\s*(.*?)\s*```$"
    match = re.match(fence_pattern, text, flags=re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()

    return text


def try_parse_json_like(text: str) -> dict[str, Any] | None:
    """
    LLMがJSON風・Python辞書風で返した場合に、できる範囲でdict化する。
    """
    cleaned = strip_code_fence(text)

    # まず通常JSONとして読む
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Python風の True / False / None やシングルクォートを雑にJSON寄せする
    normalized = cleaned
    normalized = normalized.replace("'", '"')
    normalized = re.sub(r"\bFalse\b", "false", normalized)
    normalized = re.sub(r"\bTrue\b", "true", normalized)
    normalized = re.sub(r"\bNone\b", "null", normalized)

    try:
        parsed = json.loads(normalized)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None

    return None


def format_dict_as_japanese_answer(data: dict[str, Any]) -> str:
    """
    設定値dictを、音声化しやすい日本語説明へ変換する。
    """
    preferred_keys = [
        "base_url",
        "model",
        "temperature",
        "keep_alive",
        "unload_at_exit",
    ]

    lines = ["conf.yaml の ollama_llm 設定は以下の通りです。", ""]

    used_keys: set[str] = set()

    for key in preferred_keys:
        if key in data:
            lines.append(f"- {key}: {data[key]}")
            used_keys.add(key)

    for key, value in data.items():
        if key not in used_keys:
            lines.append(f"- {key}: {value}")

    return "\n".join(lines).strip()


def normalize_answer_for_vtuber(answer: str) -> str:
    """
    Open-LLM-VTuber / TTS に渡しやすい形へ後処理する。

    目的:
    - コードブロックを除去
    - JSON風回答を日本語説明へ変換
    - 余分な前置きを削る
    """
    answer = answer.strip()

    parsed = try_parse_json_like(answer)

    if parsed is not None:
        return format_dict_as_japanese_answer(parsed)

    answer = strip_code_fence(answer)

    # JSON形式でなくても、コードブロックだけは消す
    answer = answer.replace("```", "").strip()

    return answer


def answer_with_rag(
    query: str,
    model: str = "tinyswallow-vtuber:latest",
    base_url: str = "http://localhost:11434/v1",
    temperature: float = 0.2,
    max_tokens: int = 300,
    save_logs: bool = True,
    max_reliable_score: float = 0.85,
) -> RagAnswerResult:
    """
    RAG検索からOllama回答生成までを1関数で実行する。

    Open-LLM-VTuber本体へ組み込む前の共通入口として使う。
    """
    retriever = IntervalClusterRetriever(
        top_k=10,
        candidate_count=20,
        cluster_count=5,
        max_context_chunks=1,
        max_chars_per_cluster=4000,
        neighbor_before=1,
        neighbor_after=6,
    )

    result = retriever.retrieve(query)

    top_representative_chunk_id = None
    retrieval_score = None
    retrieval_reliable = False

    if result.cluster_summaries:
        top_summary = result.cluster_summaries[0]
        top_rep = top_summary.representative
        top_representative_chunk_id = top_rep.chunk.chunk_id
        retrieval_score = top_rep.interval.score
        retrieval_reliable = retrieval_score <= max_reliable_score

    if not retrieval_reliable:
        fallback_answer = (
            "根拠情報からは確認できません。"
            "対象のログファイルやエラー内容が docs に入っていない可能性があります。"
        )

        prompt = result.prompt_result.prompt

        if save_logs:
            save_text(
                Path("rag_interval_cluster/logs/last_rag_answer_prompt.txt"),
                prompt,
            )
            save_text(
                Path("rag_interval_cluster/logs/last_rag_answer_raw.txt"),
                fallback_answer,
            )
            save_text(
                Path("rag_interval_cluster/logs/last_rag_answer.txt"),
                fallback_answer,
            )

        return RagAnswerResult(
            query=query,
            answer=fallback_answer,
            raw_answer=fallback_answer,
            prompt=prompt,
            prompt_chars=len(prompt),
            used_cluster_count=result.prompt_result.used_cluster_count,
            used_chunk_count=result.prompt_result.used_chunk_count,
            top_representative_chunk_id=top_representative_chunk_id,
            retrieval_score=retrieval_score,
            retrieval_reliable=False,
        )

    prompt = build_answer_prompt(result.prompt_result.prompt)

    client = OllamaChatClient(
        base_url=base_url,
        model=model,
        timeout_sec=180,
    )

    answer_result = client.chat(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    raw_answer = answer_result.content
    normalized_answer = normalize_answer_for_vtuber(raw_answer)

    if result.cluster_summaries:
        top_representative_chunk_id = (
            result.cluster_summaries[0]
            .representative
            .chunk
            .chunk_id
        )

    if save_logs:
        save_text(
            Path("rag_interval_cluster/logs/last_rag_answer_prompt.txt"),
            prompt,
        )
        save_text(
            Path("rag_interval_cluster/logs/last_rag_answer_raw.txt"),
            raw_answer,
        )
        save_text(
            Path("rag_interval_cluster/logs/last_rag_answer.txt"),
            normalized_answer,
        )

    return RagAnswerResult(
        query=query,
        answer=normalized_answer,
        raw_answer=raw_answer,
        prompt=prompt,
        prompt_chars=len(prompt),
        used_cluster_count=result.prompt_result.used_cluster_count,
        used_chunk_count=result.prompt_result.used_chunk_count,
        top_representative_chunk_id=top_representative_chunk_id,
        retrieval_score=retrieval_score,
        retrieval_reliable=retrieval_reliable,
    )