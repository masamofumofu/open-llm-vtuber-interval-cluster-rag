# Implementation Notes

このドキュメントでは、Open-LLM-VTuber に階層的区間クラスタリングRAGを組み込むために行った差し込み箇所と、追加コードの概要を説明する。

本リポジトリでは Open-LLM-VTuber 本体を同梱しない。  
また、`single_conversation.py` は利用環境ごとに差異が出やすいため、本体ファイルを丸ごと公開せず、手動での差し込み手順として記録する。

---

## 1. 組み込み方針

Open-LLM-VTuber の通常会話処理では、ユーザー入力を受け取った後、最終的に `context.agent_engine.chat(batch_input)` が呼び出される。

本実装では、その直前に RAG 判定を差し込む。

挙動は以下の通りである。

```text
enabled=False
  → 従来通り通常LLMへ渡す

enabled=True かつ RAG対象外
  → 従来通り通常LLMへ渡す

enabled=True かつ RAG対象
  → 通常LLMをスキップ
  → RAG回答を生成
  → 回答全文を画面に表示
  → TTS用短文だけをMioTTSへ渡す
  → 会話終了通知を送信
```

この構成により、雑談などの通常会話は従来動作を維持しつつ、設定ファイルやログ確認などの根拠参照が必要な質問だけRAG経路に切り替える。

---

## 2. 前提ファイル

Open-LLM-VTuber のプロジェクトルート直下に、以下のようなRAGモジュールを配置する。

```text
Open-LLM-VTuber/
├─ rag_interval_cluster/
│  ├─ __init__.py
│  ├─ chunker.py
│  ├─ document_loader.py
│  ├─ embedder_tfidf.py
│  ├─ interval_distance.py
│  ├─ hierarchical_cluster.py
│  ├─ prompt_builder.py
│  ├─ retriever.py
│  ├─ ollama_client.py
│  ├─ rag_answer.py
│  ├─ rag_gate.py
│  ├─ rag_chat.py
│  ├─ rag_tts_text.py
│  └─ rag_runtime_config.py
└─ src/
   └─ open_llm_vtuber/
      └─ conversations/
         └─ single_conversation.py
```

`rag_interval_cluster/docs/` には検索対象文書を配置する。  
ただし、GitHub公開用リポジトリには実際の `conf.yaml` やログファイルは含めない。

---

## 3. ランタイム設定

RAGのON/OFFや利用モデルは `rag_runtime_config.json` で管理する。

例:

```json
{
  "enabled": true,
  "model": "tinyswallow-vtuber:latest",
  "base_url": "http://localhost:11434/v1",
  "temperature": 0.2,
  "max_tokens": 300
}
```

`enabled` を `false` にすると、RAG判定ログだけを出し、通常LLM経路に戻せる。

---

## 4. `single_conversation.py` への import 追加

`src/open_llm_vtuber/conversations/single_conversation.py` の import 群に、以下を追加する。

```python
import json
import asyncio

from rag_interval_cluster.rag_gate import should_use_rag
from rag_interval_cluster.rag_runtime_config import load_runtime_config
from rag_interval_cluster.rag_chat import chat_with_optional_rag
from rag_interval_cluster.rag_tts_text import build_rag_tts_text
from ..utils.stream_audio import prepare_audio_payload
```

既に `json` や `asyncio` が import 済みの場合は重複させない。

---

## 5. 差し込み箇所

`process_single_conversation()` 内で、次の通常LLM呼び出しを探す。

```python
# agent.chat yields Union[SentenceOutput, Dict[str, Any]]
agent_output_stream = context.agent_engine.chat(batch_input)
```

この直前に、RAG分岐を追加する。

---

## 6. 追加するRAG分岐コード

以下のコードを `agent_output_stream = context.agent_engine.chat(batch_input)` の直前に追加する。

```python
# Interval RAG integration:
# enabled=True かつ should_use_rag=True の場合だけ、
# 通常LLMをスキップしてRAG回答を使う。
try:
    rag_runtime_config = load_runtime_config()
    rag_gate_result = should_use_rag(str(user_input))

    logger.info(
        "[IntervalRAG] "
        f"enabled={rag_runtime_config.enabled}, "
        f"use_rag={rag_gate_result.use_rag}, "
        f"reason={rag_gate_result.reason}, "
        f"matched_keywords={rag_gate_result.matched_keywords}, "
        f"user_input={user_input}"
    )

    if rag_runtime_config.enabled and rag_gate_result.use_rag:
        logger.info("[IntervalRAG] RAG answer mode enabled. Skipping normal agent chat.")

        rag_result = await asyncio.to_thread(
            chat_with_optional_rag,
            query=str(user_input),
            model=rag_runtime_config.model,
            base_url=rag_runtime_config.base_url,
            temperature=rag_runtime_config.temperature,
            max_tokens=rag_runtime_config.max_tokens,
            save_logs=True,
        )

        if rag_result.answer:
            full_response = rag_result.answer

            rag_tts_text = build_rag_tts_text(
                query=str(user_input),
                answer=full_response,
                max_chars=40,
            )

            try:
                # 画面にはRAG回答全文を表示する
                await websocket_send(
                    json.dumps(
                        {
                            "type": "full-text",
                            "text": full_response,
                            "name": context.character_config.character_name,
                            "avatar": context.character_config.avatar,
                        },
                        ensure_ascii=False,
                    )
                )

                # 音声は短文だけ読み上げる
                if rag_tts_text:
                    logger.info(f"[IntervalRAG] RAG TTS text: {rag_tts_text}")

                    audio_file_path = await context.tts_engine.async_generate_audio(
                        text=rag_tts_text,
                        file_name_no_ext=f"rag_{client_uid}",
                    )

                    payload = prepare_audio_payload(
                        audio_path=audio_file_path,
                        display_text={
                            "text": rag_tts_text,
                            "name": context.character_config.character_name,
                            "avatar": context.character_config.avatar,
                        },
                        actions=None,
                    )

                    await websocket_send(
                        json.dumps(
                            payload,
                            ensure_ascii=False,
                        )
                    )

                    try:
                        context.tts_engine.remove_file(audio_file_path)
                    except Exception as remove_error:
                        logger.warning(
                            f"[IntervalRAG] failed to remove temp audio: {remove_error}"
                        )

                await websocket_send(
                    json.dumps(
                        {
                            "type": "backend-synth-complete",
                        },
                        ensure_ascii=False,
                    )
                )

                await websocket_send(
                    json.dumps(
                        {
                            "type": "force-new-message",
                        },
                        ensure_ascii=False,
                    )
                )

            except Exception as websocket_error:
                logger.error(f"[IntervalRAG] websocket or TTS send failed: {websocket_error}")

            logger.info(f"[IntervalRAG] RAG response: {full_response}")

            if context.history_uid:
                store_message(
                    conf_uid=context.character_config.conf_uid,
                    history_uid=context.history_uid,
                    role="ai",
                    content=full_response,
                    name=context.character_config.character_name,
                    avatar=context.character_config.avatar,
                )

            return full_response

        logger.warning(
            "[IntervalRAG] RAG selected but answer was empty. "
            "Falling back to normal agent chat."
        )

except Exception as rag_check_error:
    logger.error(f"[IntervalRAG] check or answer failed: {rag_check_error}")
```

このコードの直後に、既存の通常LLM処理を残す。

```python
# agent.chat yields Union[SentenceOutput, Dict[str, Any]]
agent_output_stream = context.agent_engine.chat(batch_input)
```

---

## 7. この実装で行っていること

### 7.1 RAG使用判定

```python
rag_gate_result = should_use_rag(str(user_input))
```

ユーザー入力に対して、RAG対象かどうかを判定する。

例:

```text
こんにちは。
  → use_rag=False

conf.yamlのollama_llm設定は？
  → use_rag=True
```

### 7.2 RAGが無効な場合

```python
if rag_runtime_config.enabled and rag_gate_result.use_rag:
```

この条件に入らない場合、既存の `context.agent_engine.chat(batch_input)` がそのまま呼ばれる。

これにより、通常会話の挙動を維持できる。

### 7.3 RAGが有効な場合

RAG対象質問では、通常LLMを呼ばずに以下を実行する。

```python
rag_result = await asyncio.to_thread(
    chat_with_optional_rag,
    query=str(user_input),
    model=rag_runtime_config.model,
    base_url=rag_runtime_config.base_url,
    temperature=rag_runtime_config.temperature,
    max_tokens=rag_runtime_config.max_tokens,
    save_logs=True,
)
```

`asyncio.to_thread()` を使うことで、同期的なRAG処理をイベントループ内で安全に実行する。

### 7.4 画面表示

RAG回答全文は `full-text` としてフロントへ送信する。

```python
await websocket_send(
    json.dumps(
        {
            "type": "full-text",
            "text": full_response,
            "name": context.character_config.character_name,
            "avatar": context.character_config.avatar,
        },
        ensure_ascii=False,
    )
)
```

### 7.5 TTS短文化

RAG回答全文をそのまま読み上げると長くなりやすいため、音声用には短文を作る。

```python
rag_tts_text = build_rag_tts_text(
    query=str(user_input),
    answer=full_response,
    max_chars=40,
)
```

例:

```text
RAG回答全文:
conf.yaml の ollama_llm 設定は以下の通りです。

- base_url: http://localhost:11434/v1
- model: tinyswallow-vtuber:latest
- temperature: 0.6
- keep_alive: -1
- unload_at_exit: False

TTS短文:
モデルはtinyswallow-vtuber:latestです。
```

### 7.6 音声送信

TTS短文だけを `context.tts_engine.async_generate_audio()` に渡し、生成された音声を `prepare_audio_payload()` でWebSocket送信用payloadに変換する。

```python
audio_file_path = await context.tts_engine.async_generate_audio(
    text=rag_tts_text,
    file_name_no_ext=f"rag_{client_uid}",
)

payload = prepare_audio_payload(
    audio_path=audio_file_path,
    display_text={
        "text": rag_tts_text,
        "name": context.character_config.character_name,
        "avatar": context.character_config.avatar,
    },
    actions=None,
)
```

### 7.7 会話終了通知

RAG回答後も通常会話へ戻れるように、以下を送信する。

```python
await websocket_send(
    json.dumps(
        {
            "type": "backend-synth-complete",
        },
        ensure_ascii=False,
    )
)

await websocket_send(
    json.dumps(
        {
            "type": "force-new-message",
        },
        ensure_ascii=False,
    )
)
```

これにより、RAG回答後に次の入力を受け付けられる状態に戻る。

---

## 8. 動作確認

### 8.1 構文チェック

```powershell
cd C:\work\Open-LLM-VTuber

uv run python -m py_compile .\src\open_llm_vtuber\conversations\single_conversation.py
```

何も出力されなければ構文チェック成功。

---

### 8.2 RAG対象外の確認

入力:

```text
こんにちは。
```

期待ログ:

```text
[IntervalRAG] enabled=True, use_rag=False
```

期待動作:

```text
通常LLM経路で応答する
MioTTSで通常どおり読み上げる
```

---

### 8.3 RAG対象質問の確認

入力:

```text
conf.yamlのollama_llm設定は？
```

期待ログ:

```text
[IntervalRAG] enabled=True, use_rag=True
[IntervalRAG] RAG answer mode enabled. Skipping normal agent chat.
[IntervalRAG] RAG TTS text: モデルはtinyswallow-vtuber:latestです。
[IntervalRAG] RAG response: conf.yaml の ollama_llm 設定は以下の通りです。
```

期待回答:

```text
conf.yaml の ollama_llm 設定は以下の通りです。

- base_url: http://localhost:11434/v1
- model: tinyswallow-vtuber:latest
- temperature: 0.6
- keep_alive: -1
- unload_at_exit: False
```

期待TTS:

```text
モデルはtinyswallow-vtuber:latestです。
```

---

## 9. 確認済みの結果

以下を確認済みである。

```text
- RAG対象外の通常会話が可能
- RAG対象質問で通常LLMをスキップできる
- RAG回答の値が正しい
- RAG回答全文を画面に表示できる
- RAG回答からTTS短文を生成できる
- MioTTSで短文を読み上げできる
- RAG回答後に通常会話へ戻れる
```

---

## 10. 注意点

### 10.1 `conf.yaml` は公開しない

検索対象として使用した `conf.yaml` はローカル環境固有の設定を含むため、GitHubにはアップロードしない。

公開リポジトリには、設定例として `examples/rag_runtime_config.example.json` のみを含める。

### 10.2 Open-LLM-VTuber本体は同梱しない

本実装は Open-LLM-VTuber への追加実験である。  
Open-LLM-VTuber本体のコードは含めず、差し込み手順のみを記載する。

### 10.3 バージョン差異

Open-LLM-VTuber のバージョンによって、`single_conversation.py` の構造や関数名が異なる可能性がある。

その場合は、以下の通常LLM呼び出しに相当する箇所を探し、その直前にRAG分岐を差し込む。

```python
context.agent_engine.chat(batch_input)
```

---

## 11. 今後の改善候補

- RAG回答生成の高速化
- 設定値問い合わせ時の直接抽出ルート追加
- 対象文書の差分更新
- embeddingモデルへの差し替え
- YAML / SQL / Markdown / log それぞれに適したチャンク分割
- RAG対象判定キーワードの調整
- 信頼度しきい値の自動調整
- RAGログの可視化
