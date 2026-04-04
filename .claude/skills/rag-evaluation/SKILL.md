---
name: rag-evaluation
description: RAGシステムの精度評価・診断・改善策提案を支援するskill。「RAGの精度が悪い」「回答がおかしい」「検索がヒットしない」「幻覚が起きる」「recallを上げたい」「precisionを改善したい」「RAGをどこから改善すればいいかわからない」といった場面で必ず使用すること。Self-RAG・Agentic RAG・RAG Reasoningなどの高度な生成フェーズ改善も対象。「評価」「精度」「改善」「recall」「precision」「ハルシネーション」などのキーワードが出たら積極的にこのskillを参照すること。
---

# RAG Evaluation & Improvement Skill

RAGシステムの精度評価と改善を支援するskill。  
設計・構築フェーズについては `rag-design` skillを参照。

---

## RAG精度の4軸評価

| 指標 | 意味 | 改善のヒント |
|---|---|---|
| **Precision（正確さ）** | 回答に含まれる情報がどれだけ正確か | 不要チャンクを減らす（Rerank・ChunkRAG） |
| **Recall（再現性）** | 必要な情報をどれだけ取りこぼさずに検索できているか | Hybrid Search・クエリ拡張 |
| **一貫性** | 同じ質問に対して回答がブレないか | Self-RAGのCritiqueステップ |
| **説明可能性** | 回答の根拠が示せるか | GraphRAG・引用付き回答生成 |

---

## 問題診断フローチャート

```
RAGの精度が悪い
├─ 検索がそもそもヒットしていない（Recall低下）
│   ├─ キーワードが完全一致しないと拾えない → Hybrid Search（Sparse追加）
│   ├─ ユーザの質問が曖昧・短い → Query Rewrite（HyDE）
│   └─ チャンクが小さすぎて文脈が失われている → チャンクサイズ見直し or Parent Page Retrieval
│
├─ 検索はできているが回答がおかしい（Precision低下）
│   ├─ 無関係なチャンクが混入している → Rerank / LLM Rerank / ChunkRAG
│   ├─ チャンクが大きすぎてノイズが多い → チャンクサイズを小さく
│   └─ チャンクに文脈がない → Contextual Retrieval
│
├─ 複数文書・複雑な関係が絡む質問に弱い
│   └─ → GraphRAG / RAPL / Agentic RAG
│
└─ 回答一貫性・自己矛盾がある
    └─ → Self-RAG（Critiqueステップ）
```

---

## 改善ステップ別対応表

RAGの改善は「前処理・検索・生成」どのフェーズに手を入れるかで分類できる。

### フェーズ1：前処理（Data Parsing & Indexing）での改善

| 問題 | 改善手法 |
|---|---|
| 表データの回答精度が低い | HTMLフォーマットで保存 / LanceDB活用 |
| ノイズが多い | 正規化処理（ヘッダ/フッタ・不可視記号の除去） |
| チャンク境界が意味的におかしい | Semantic Chunking |

### フェーズ2：検索（Retrieval）での改善

| 問題 | 改善手法 | コスト |
|---|---|---|
| Recall低下 | Hybrid Search / Query Rewrite | 低〜中 |
| Precision低下 | Rerank / LLM Rerank | 中 |
| 無関係チャンクの混入 | ChunkRAG | 中 |
| 文脈の欠落 | Parent Page Retrieval | 中 |
| エンティティ関係が重要 | GraphRAG / RAPL | 高（低速） |

### フェーズ3：生成（Generation）での改善

詳細は `references/generation-methods.md` を参照。

| 手法 | 概要 | 効果 |
|---|---|---|
| **Self-RAG** | 検索要否・自己批評・根拠評価を組み込む | 一貫性・説明可能性向上 |
| **Agentic RAG** | 複数データソースを使い分けるエージェント構成 | 複雑なクエリへの対応力向上 |
| **RAG Reasoning** | 推論とRAGを組み合わせる発展的手法 | 高度な推論を要するクエリに有効 |

---

## 改善優先度ガイド

コスト対効果の観点から以下の順で試すことを推奨：

1. **Query Rewrite（HyDE）**：実装コストが低く、即効性がある
2. **Hybrid Search**：キーワード検索の追加は効果が出やすい
3. **Rerank**：候補を出した後の精度向上
4. **チャンクサイズの見直し**：根本的な改善になりやすい
5. **Parent Page Retrieval / ChunkRAG**：構造化文書に特に有効
6. **Self-RAG**：一貫性・根拠の質を上げたい場合
7. **GraphRAG / Agentic RAG**：複雑な構成が必要な場合（コスト大）

---

## 参照ファイル

- `references/generation-methods.md`：Self-RAG・Agentic RAG・RAG Reasoningの詳細
