\# Architecture



\## 1. 基本方針



本実装は、Open-LLM-VTuber の通常会話経路に対して、RAG判定を前段で差し込む構成である。



通常会話は従来どおり `agent\_engine.chat()` に渡し、設定ファイルやログ確認のような根拠参照が必要な質問だけRAG経路へ切り替える。



```text

ユーザー入力

→ RAG使用判定

→ RAG対象外なら通常LLM

→ RAG対象ならRAG回答

```



\## 2. 通常RAGとの差分



一般的なRAGは以下の構成である。



```text

query

→ retrieve top-k

→ prompt

→ LLM answer

```



本実装では、検索結果をそのまま渡さず、次の後処理を行う。



```text

query

→ query variants

→ TF-IDF search

→ interval distance ranking

→ hierarchical clustering

→ representative chunk selection

→ neighbor expansion

→ prompt

→ LLM answer

```



\## 3. 区間距離



単一のクエリだけでチャンク距離を評価すると、表現ゆれに弱くなる。

そこで、ユーザー質問から複数のクエリ変種を作り、それぞれに対する距離を計算する。



各チャンクに対して、距離の最小値と最大値から距離区間を作る。



```text

chunk\_i:

&#x20; interval\_i = \[low\_i, high\_i]

```



区間幅が小さいチャンクは、クエリ変種に対して安定して近いとみなす。

区間幅が大きいチャンクは、特定の表現には近いが、別の表現では遠い可能性がある。



スコアは、中心と区間幅を組み合わせて計算する。



```text

center = (low + high) / 2

width  = high - low

score  = center + lambda \* width

```



\## 4. 階層的クラスタリング



検索候補チャンク同士の距離を計算し、階層的クラスタリングを行う。



目的は、似た根拠をまとめ、同じ内容のチャンクばかりをプロンプトへ入れないことである。



クラスタごとに代表チャンクを選び、必要に応じて前後の近隣チャンクも展開する。



```text

検索候補

→ クラスタリング

→ クラスタ代表チャンク

→ 近隣チャンク展開

→ 根拠情報

```



\## 5. プロンプト生成



LLMへ渡すプロンプトには、以下を含める。



\* 回答ルール

\* 根拠クラスタ

\* ファイル名

\* チャンクID

\* 距離区間

\* 信頼度

\* ユーザー質問



根拠にない内容は推測しないように指示する。



\## 6. Open-LLM-VTuber統合



Open-LLM-VTuberの個別会話処理に対し、通常LLM呼び出しの直前でRAG判定を行う。



```text

enabled=False

&#x20; → 通常LLM



enabled=True かつ RAG対象外

&#x20; → 通常LLM



enabled=True かつ RAG対象

&#x20; → 通常LLMをスキップ

&#x20; → RAG回答

```



\## 7. 画面表示とTTSの分離



RAG回答は箇条書きや設定値を含むため、そのまま音声化すると長く、不自然になりやすい。



そのため、画面表示と音声読み上げを分離する。



```text

画面表示:

&#x20; RAG回答全文



TTS:

&#x20; 短く整形した1文

```



例:



```text

画面表示:

conf.yaml の ollama\_llm 設定は以下の通りです。



\- base\_url: http://localhost:11434/v1

\- model: tinyswallow-vtuber:latest

\- temperature: 0.6

\- keep\_alive: -1

\- unload\_at\_exit: False



TTS:

モデルはtinyswallow-vtuber:latestです。

```



\## 8. 信頼度ガード



検索結果のスコアが低信頼と判断される場合、LLM生成を行わず、以下のように返す。



```text

根拠情報からは確認できません。

```



これにより、根拠のないログや未登録ファイルについて、LLMが推測で回答することを抑制する。



\## 9. 現在の制約



現在の実装には以下の制約がある。



\* TF-IDFベースのため、意味検索能力は限定的

\* 初回回答生成に時間がかかる場合がある

\* 検索対象ファイルの更新差分処理は未実装

\* 設定値問い合わせでも一部LLM生成を行う

\* Open-LLM-VTuber本体のバージョン差異に注意が必要



\## 10. 今後の改善



今後の改善候補は以下である。



\* embeddingモデルの導入

\* 設定値問い合わせの直接抽出

\* ファイル更新時の差分インデックス更新

\* クラスタリング条件の自動調整

\* YAML / SQL / Markdown / log 別のチャンク分割

\* RAG処理時間の短縮



