---
name: ai-model-knowledge
description: |
  AI APIを使ったコードを書くとき、またはAIモデルのモデルID・スペックを指定する必要があるときに必ず使うskill。OpenAI（GPT/o系）、Anthropic（Claude）、Google（Gemini）、Meta（Llama）、Mistral等、主要LLM全般が対象。

  【必ずトリガーすべき場面】
  - OpenAI API、Anthropic API、Gemini API、Llama APIなどを使ったコードを書くとき
  - コード中にモデルID文字列（例："gpt-4o"、"claude-3-5-sonnet"等）を指定するとき
  - 「どのモデルを使えばいい？」「このAPIの最新モデルは？」と聞かれたとき
  - 既存コードのモデル名を更新・見直したいとき
  - ユーザーが「そのモデルIDは古い／違う」と指摘してきたとき

  AIモデルの情報は頻繁に更新される。コードに古いモデルIDを埋め込むと動作しなくなるリスクがあるため、このskillを使って必ず最新情報を確認してから実装すること。
---

# AI Model Knowledge Skill

## 概要

Claude Codeのトレーニングデータのカットオフ後、AIモデルは継続的にリリース・更新・廃止されている。特にAPI経由でモデルを指定するコードでは、古いモデルIDを使うとAPIエラーや意図しない動作につながる。このskillは、AI APIを使ったコーディング時に最新の正式モデルIDを確認・使用することを徹底させる。

---

## 必須ワークフロー

### Step 1: 公式ドキュメントでモデルIDを確認する（必須）

コードを書く前に、使用するプロバイダーの公式モデル一覧ページを**必ず確認すること**。記憶のモデルIDをそのまま使ってはいけない。

#### プロバイダー別・公式モデル一覧URL（直接fetchを推奨）

| プロバイダー | 公式モデル一覧 |
|---|---|
| OpenAI | `https://platform.openai.com/docs/models` |
| Anthropic | `https://docs.anthropic.com/en/docs/about-claude/models/overview` |
| Google Gemini | `https://ai.google.dev/gemini-api/docs/models` |
| Meta Llama | `https://llama.meta.com` |
| Mistral | `https://docs.mistral.ai/getting-started/models/models_overview/` |

#### 検索クエリの例（URLが取得できない場合）
```
OpenAI latest models API 2025
Anthropic Claude models API string 2025
Gemini API model list 2025
Llama latest model release 2025
```

### Step 2: コーディングに必要な情報を確認する

公式ドキュメントから以下を必ず確認してからコードを書く：

- [ ] **APIで使う正式なモデルID文字列**（例：`gpt-4o-2024-11-20`）
- [ ] **推奨モデル vs レガシーモデルの区別**
- [ ] **非推奨（deprecated）になったモデルがないか**
- [ ] **コンテキストウィンドウサイズ**（処理するデータ量に影響）
- [ ] **対応モダリティ**（テキスト/画像/音声など、必要な機能が使えるか）
- [ ] **料金・レート制限**（本番環境での考慮が必要な場合）

### Step 3: 確認したモデルIDでコードを実装する

```python
# ❌ 悪い例：記憶のまま古いモデルIDを使う
# → APIエラーや意図しない動作のリスクあり
response = client.chat.completions.create(model="gpt-4")

# ✅ 良い例：公式ドキュメントで確認した最新の正式モデルIDを使う
response = client.chat.completions.create(model="gpt-4o-2024-11-20")
```

```python
# Anthropic の例
# ❌ 悪い例
client.messages.create(model="claude-3-opus")

# ✅ 良い例（公式ドキュメントで確認したID）
client.messages.create(model="claude-opus-4-5")
```


---

## 特に注意が必要なケース

以下のモデル・状況では知識が古い可能性が特に高い：

- **OpenAI o系モデル**：o1・o3・o4など推論特化シリーズは頻繁に更新される
- **「mini」「nano」「turbo」「preview」サフィックス付きモデル**：バリアント名は変わりやすい
- **バージョン日付付きモデル**（例：`gpt-4o-2024-11-20`）：新しい日付版がリリースされていることがある
- **廃止されたモデルを参照しているレガシーコード**のリファクタリング時

---

## よくある誤りと対処法

| 誤りのパターン | 対処法 |
|---|---|
| `"gpt-4"` をそのまま書く | 公式で現在の推奨IDを確認する |
| `"claude-3-opus"` など古いIDを使う | Anthropic公式の最新モデル一覧を確認する |
| 存在しないモデルIDをハルシネーションで生成する | 必ず公式ドキュメントのモデル一覧と照合する |
| コンテキスト長を誤って実装する | 公式スペックを確認してから実装する |
| 廃止モデルを使い続ける | deprecatedリストを確認し、後継モデルに移行する |

---

## ユーザーへの説明テンプレート

コードにモデルIDを入れる前に、ユーザーに伝える：

> 「AIモデルは頻繁に更新されるため、公式ドキュメントで最新のモデルIDを確認してから実装します。」

確認後のコメント例（コード内）：
```python
# モデルID: 公式ドキュメント確認済み（YYYY-MM-DD時点）
# 参照: https://platform.openai.com/docs/models
model = "gpt-4o-2024-11-20"
```
