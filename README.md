# Hierarchical Interval Cluster RAG for Open-LLM-VTuber

## 概要

本リポジトリは、Open-LLM-VTuber に検索結果後処理型の階層的区間クラスタリングRAGを追加するための実験実装と技術解説です。

通常のRAGでは、検索上位チャンクをそのままLLMへ渡します。  
本実装では、検索結果に対して以下の後処理を行います。

1. TF-IDFによる候補検索
2. クエリ変種に基づく距離区間の計算
3. 区間距離による再ランキング
4. 階層的クラスタリング
5. 代表チャンクと近隣チャンクの展開
6. 根拠付きプロンプト生成
7. Ollama OpenAI互換APIによる回答生成
8. Open-LLM-VTuber上での回答表示
9. RAG回答の短文TTS化

## まず読む

HIC-RAGの考え方を理解するには、まず以下の順に読むことを推奨します。

1. [構成図](docs/architecture_overview.md)  
   Open-LLM-VTuberにHIC-RAGをどう組み込んでいるかを図で確認できます。

2. [階層的区間クラスタリングRAGの数学的定式化](docs/hierarchical_interval_clustering_rag_paper_ja.md)  
   HIC-RAGの理論的背景、区間距離、階層クラスタリング、信頼度ゲート、根拠制約付き生成について説明しています。

3. [Open-LLM-VTuberへの組み込み手順](docs/implementation_notes.md)  
   実際に `single_conversation.py` へ差し込む箇所と追加コードを説明しています。

## 背景

ローカルLLM環境では、モデルサイズ、VRAM、CPU性能などの制約があります。  
そのため、LLMの内部知識だけに頼ると、設定ファイルやログの内容について幻覚が発生しやすくなります。

本実装では、ローカル文書を検索し、根拠に基づいて応答することで、軽量ローカルLLMの信頼性を補強します。

## 構成図

全体構成は以下を参照してください。

- [docs/architecture_overview.md](docs/architecture_overview.md)

## 対象環境

検証環境は以下です。

- Open-LLM-VTuber v1.2.1
- Ollama
- TinySwallow系ローカルモデル
- MioTTS
- Windows 11
- Python / uv

ただし、本リポジトリは実験実装であり、環境差異に応じた調整が必要です。

## 特徴

- Open-LLM-VTuberへの最小差し込み
- RAG対象質問だけ通常LLMをスキップ
- 雑談は従来の会話経路を維持
- RAG回答全文は画面表示
- 音声は短文だけMioTTSへ渡す
- 根拠不足時は「根拠情報からは確認できません」と返答
- 実際の `conf.yaml` やログファイルは同梱しない安全構成

## 構成

```text
open-llm-vtuber-interval-cluster-rag/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ docs/
│  ├─ implementation_notes.md
│  ├─ architecture.md
│  └─ test_results.md
├─ examples/
│  └─ rag_runtime_config.example.json
└─ rag_interval_cluster/
   ├─ __init__.py
   ├─ config.py
   ├─ chunker.py
   ├─ document_loader.py
   ├─ embedder_tfidf.py
   ├─ interval_distance.py
   ├─ hierarchical_cluster.py
   ├─ prompt_builder.py
   ├─ retriever.py
   ├─ ollama_client.py
   ├─ rag_answer.py
   ├─ rag_gate.py
   ├─ rag_chat.py
   ├─ rag_tts_text.py
   └─ rag_runtime_config.py
```

## Open-LLM-VTuberへの組み込み

`single_conversation.py` は利用環境ごとに差異が出やすいため、本リポジトリでは本体ファイルを同梱していません。

組み込み手順は以下を参照してください。

- [docs/implementation_notes.md](docs/implementation_notes.md)

## アーキテクチャと理論背景

技術構成と理論背景は以下を参照してください。

- [構成図](docs/architecture_overview.md)
- [アーキテクチャ詳細](docs/architecture.md)
- [階層的区間クラスタリングRAGの数学的定式化](docs/hierarchical_interval_clustering_rag_paper_ja.md)

## 動作確認結果

検証結果は以下を参照してください。

- [docs/test_results.md](docs/test_results.md)

## 動作例

質問:

```text
conf.yamlのollama_llm設定は？
```

RAG回答:

```text
conf.yaml の ollama_llm 設定は以下の通りです。

- base_url: http://localhost:11434/v1
- model: tinyswallow-vtuber:latest
- temperature: 0.6
- keep_alive: -1
- unload_at_exit: False
```

TTS短文:

```text
モデルはtinyswallow-vtuber:latestです。
```

## 処理フロー

```text
ユーザー入力
→ RAG使用判定
→ RAG対象外なら通常LLM
→ RAG対象なら文書検索
→ 区間距離による再ランキング
→ 階層クラスタリング
→ 根拠プロンプト生成
→ Ollama回答生成
→ 回答整形
→ 画面へ全文表示
→ MioTTSへ短文だけ送信
```

## ランタイム設定例

`examples/rag_runtime_config.example.json`:

```json
{
  "enabled": true,
  "model": "tinyswallow-vtuber:latest",
  "base_url": "http://localhost:11434/v1",
  "temperature": 0.2,
  "max_tokens": 300
}
```

`enabled` を `false` にすると、RAG判定ログだけを出し、従来の通常LLM経路へ戻せます。

## 注意

本リポジトリには以下を含めません。

- 実際の `conf.yaml`
- ローカルログ
- 音声ファイル
- モデルファイル
- Open-LLM-VTuber本体一式
- 個人環境固有の設定値
- APIキー、トークン、パスワード等の認証情報

## 現在の到達点

以下を確認済みです。

- RAG検索が動作する
- 区間距離ランキングが動作する
- 階層クラスタリングが動作する
- RAG回答が生成できる
- Open-LLM-VTuberに統合できる
- RAG対象質問で通常LLMをスキップできる
- RAG回答を画面表示できる
- RAG回答を短文TTS化できる
- RAG回答後に通常会話へ戻れる

## 今後の課題

- RAG回答生成の高速化
- 設定値問い合わせ時の直接抽出ルート追加
- 対象文書の差分更新
- embeddingモデルへの差し替え
- YAML / SQL / Markdown / log ごとのチャンク分割最適化
- RAG対象判定キーワードの調整
- 信頼度しきい値の自動調整
- RAGログの可視化

## License

MIT License
