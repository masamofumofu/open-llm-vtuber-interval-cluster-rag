\# Test Results



\## 1. RAG単体テスト



\### 入力



```text

conf.yamlのollama\_llm設定は？

```



\### 期待結果



`conf.yaml` 内の `ollama\_llm` 設定を検索し、以下を回答する。



```text

\- base\_url: http://localhost:11434/v1

\- model: tinyswallow-vtuber:latest

\- temperature: 0.6

\- keep\_alive: -1

\- unload\_at\_exit: False

```



\### 確認結果



以下の回答が得られた。



```text

conf.yaml の ollama\_llm 設定は以下の通りです。



\- base\_url: http://localhost:11434/v1

\- model: tinyswallow-vtuber:latest

\- temperature: 0.6

\- keep\_alive: -1

\- unload\_at\_exit: False

```



\## 2. 短い質問へのクエリ補強



\### 入力



```text

conf.yamlのollama\_llm設定は？

```



\### 課題



短い質問では、検索対象項目が不足し、回答が不安定になる場合があった。



\### 対応



内部で以下の項目を補強した。



```text

base\_url

model

temperature

keep\_alive

unload\_at\_exit

```



\### 結果



短い質問でも、正しい `ollama\_llm` 設定を回答できるようになった。



\## 3. RAG対象外の通常会話



\### 入力



```text

こんにちは。

```



\### 確認ログ



```text

\[IntervalRAG] enabled=True, use\_rag=False

```



\### 結果



RAG経路には入らず、従来どおり通常LLMとMioTTSで応答した。



\## 4. RAG対象質問のOpen-LLM-VTuber統合テスト



\### 入力



```text

conf.yamlのollama\_llm設定は？

```



\### 確認ログ



```text

\[IntervalRAG] enabled=True, use\_rag=True

\[IntervalRAG] RAG answer mode enabled. Skipping normal agent chat.

\[IntervalRAG] RAG TTS text: モデルはtinyswallow-vtuber:latestです。

\[IntervalRAG] RAG response: conf.yaml の ollama\_llm 設定は以下の通りです。

```



\### 結果



以下を確認した。



```text

\- RAG対象質問で通常LLMをスキップできた

\- RAG回答全文を画面に表示できた

\- TTS用短文を生成できた

\- MioTTSで短文を読み上げできた

```



\## 5. RAG回答後の通常会話復帰



\### 手順



1\. RAG対象質問を送信

2\. RAG回答を確認

3\. 続けて通常会話を送信



\### 入力



```text

こんにちは。

```



\### 確認ログ



```text

\[IntervalRAG] enabled=True, use\_rag=False

Conversation Chain completed!

```



\### 結果



RAG回答後も、通常会話へ戻れることを確認した。



\## 6. RAG TTS短文化テスト



\### 入力



```text

conf.yamlのollama\_llm設定は？

```



\### RAG回答



```text

conf.yaml の ollama\_llm 設定は以下の通りです。



\- base\_url: http://localhost:11434/v1

\- model: tinyswallow-vtuber:latest

\- temperature: 0.6

\- keep\_alive: -1

\- unload\_at\_exit: False

```



\### TTS短文



```text

モデルはtinyswallow-vtuber:latestです。

```



\### 結果



RAG回答全文を読み上げず、短文だけをMioTTSへ渡すことに成功した。



\## 7. 未対応情報への低信頼度ガード



\### 入力



```text

ログのエラーを確認して

```



\### 結果



検索対象に該当するログがない場合、以下を返す。



```text

根拠情報からは確認できませんでした。

```



これにより、根拠のない推測回答を抑制できる。



\## 8. 現在の評価



現時点で以下を確認済みである。



```text

\- RAG検索が動作する

\- 区間距離ランキングが動作する

\- 階層クラスタリングが動作する

\- RAG回答が生成できる

\- Open-LLM-VTuberに統合できる

\- RAG対象質問で通常LLMをスキップできる

\- RAG回答を画面表示できる

\- RAG回答を短文TTS化できる

\- RAG回答後に通常会話へ戻れる

```



\## 9. 残課題



```text

\- RAG回答生成の高速化

\- 設定値問い合わせ時の直接抽出

\- 検索対象ファイルの追加

\- ファイル更新時の差分処理

\- embeddingモデルへの移行

\- RAG対象判定の精度向上

```



