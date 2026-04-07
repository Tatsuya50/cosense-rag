import os
from pathlib import Path

import chromadb
import openai
import streamlit as st

from query import (
    CHROMA_DIR,
    COLLECTION_NAME,
    KEYWORD_WEIGHT,
    MODEL_NAME,
    SNIPPET_CHARS,
    TOP_N,
    VECTOR_WEIGHT,
    QueryEmbeddingFunction,
    hybrid_search,
)

# app.py の場所を基準にワーキングディレクトリを固定（chroma_db/ の相対パス解決）
os.chdir(Path(__file__).parent)

# ── 定数 ──────────────────────────────────────────────────────────────────────
AVAILABLE_MODELS = ["gpt-4o-mini", "gpt-4o"]
MAX_CONTEXT_CHARS = 4000
PRECISION_THRESHOLD = 0.5

SYSTEM_PROMPT = """\
あなたは「ML & Customer Success 知識ベース」の専門アシスタントです。
ユーザーの質問に対して、提供されたコンテキスト（知識ベースから検索した内容）をもとに、
日本語で正確・簡潔に回答してください。

回答のルール:
1. コンテキストに含まれる情報のみを根拠として使用してください。
2. コンテキストに答えがない場合は「知識ベースにはその情報が見つかりませんでした」と明示してください。
3. 情報が部分的な場合は、分かる範囲で答えつつ不確かな部分を明示してください。
4. 回答はMarkdown形式で読みやすく構造化してください。
5. コンテキストのタイトル（例: 【1. ページ名】）を適宜引用して出典を示してください。
"""


# ── ChromaDB 接続（アプリ起動時に1回だけロード） ──────────────────────────────
@st.cache_resource
def get_collection():
    if not Path(CHROMA_DIR).exists():
        st.error(f"{CHROMA_DIR}/ が見つかりません。先に ingest.py を実行してください。")
        st.stop()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embedding_fn = QueryEmbeddingFunction(MODEL_NAME)

    try:
        return client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
    except Exception:
        st.error(f"コレクション '{COLLECTION_NAME}' が見つかりません。先に ingest.py を実行してください。")
        st.stop()


# ── コンテキスト構築 ───────────────────────────────────────────────────────────
def build_context(results: list) -> str:
    parts = []
    total_chars = 0
    for rank, (_, _b, _v, _id, document, metadata) in enumerate(results, 1):
        title = metadata.get("title", f"ソース{rank}")
        chunk_info = ""
        if metadata.get("total_chunks", 1) > 1:
            chunk_info = f" (チャンク {metadata['chunk_index'] + 1}/{metadata['total_chunks']})"
        section = f"【{rank}. {title}{chunk_info}】\n{document}"
        if total_chars + len(section) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining > 100:
                parts.append(section[:remaining] + "...")
            break
        parts.append(section)
        total_chars += len(section)
    return "\n\n".join(parts)


# ── GPT 回答生成 ───────────────────────────────────────────────────────────────
def generate_answer(query: str, results: list, model: str, api_key: str) -> str:
    client = openai.OpenAI(api_key=api_key)
    context = build_context(results)
    user_message = f"# 質問\n{query}\n\n# 知識ベースから検索したコンテキスト\n{context}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except openai.AuthenticationError:
        st.error("APIキーが無効です。正しい OpenAI API Key を入力してください。")
        st.stop()
    except openai.RateLimitError:
        st.error("OpenAI APIのレート制限に達しました。しばらく待ってから再試行してください。")
        st.stop()
    except Exception as e:
        st.error(f"GPT呼び出しエラー: {e}")
        st.stop()


# ── ソース表示 ─────────────────────────────────────────────────────────────────
def render_sources(results: list) -> None:
    for rank, (combined, bm25_score, vector_score, _id, document, metadata) in enumerate(results, 1):
        title = metadata.get("title", f"ソース{rank}")
        chunk_label = ""
        if metadata.get("total_chunks", 1) > 1:
            chunk_label = f" [チャンク {metadata['chunk_index'] + 1}/{metadata['total_chunks']}]"

        lines = document.split("\n")
        content_lines = lines[1:] if lines and lines[0] == title else lines
        snippet = " ".join(line.strip() for line in content_lines if line.strip())
        if len(snippet) > SNIPPET_CHARS:
            snippet = snippet[:SNIPPET_CHARS] + "..."

        relevance = "関連あり" if combined >= PRECISION_THRESHOLD else "関連なし"
        label = f"#{rank} {title}{chunk_label}  —  合計: {combined:.3f} ({relevance})"

        with st.expander(label):
            col1, col2, col3 = st.columns(3)
            col1.metric("合計スコア", f"{combined:.3f}")
            col2.metric(f"ベクトル (x{VECTOR_WEIGHT})", f"{vector_score:.3f}")
            col3.metric(f"キーワード (x{KEYWORD_WEIGHT})", f"{bm25_score:.3f}")
            st.markdown(snippet)


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cosense RAG",
    page_icon=":mag:",
    layout="wide",
)

# サイドバー
with st.sidebar:
    st.header("設定")

    api_key = st.text_input(
        "OpenAI API Key",
        value=os.environ.get("OPENAI_API_KEY", ""),
        type="password",
        help="環境変数 OPENAI_API_KEY が設定されていれば自動入力されます",
    )

    st.divider()

    model = st.selectbox(
        "GPT モデル",
        options=AVAILABLE_MODELS,
        index=0,
        help="gpt-4o-mini: 高速・低コスト / gpt-4o: 高精度",
    )

    top_n = st.slider("取得チャンク数", min_value=1, max_value=10, value=TOP_N)

    with st.expander("検索重み調整"):
        vector_weight = st.slider("ベクトル検索", 0.0, 1.0, VECTOR_WEIGHT, 0.05)
        keyword_weight = st.slider("キーワード検索", 0.0, 1.0, KEYWORD_WEIGHT, 0.05)
        st.caption(f"合計重み: {vector_weight + keyword_weight:.2f}")

    st.divider()
    st.caption(f"ナレッジベース: {COLLECTION_NAME}")
    collection = get_collection()
    st.caption(f"チャンク数: {collection.count()}")

# メイン画面
st.title("Cosense RAG")
st.caption("ML & Customer Success 知識ベースへの質問応答システム")

query = st.text_input(
    "質問を入力してください",
    placeholder="例: アップリフトモデリングとは何ですか？",
)

if st.button("検索・回答生成", type="primary", disabled=not query):
    if not api_key:
        st.error("サイドバーから OpenAI API Key を入力してください。")
        st.stop()

    with st.spinner("知識ベースを検索中..."):
        results = hybrid_search(
            collection,
            query,
            n_results=top_n,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
        )

    with st.spinner(f"{model} が回答を生成中..."):
        answer = generate_answer(query, results, model, api_key)

    st.subheader("回答")
    st.markdown(answer)

    st.subheader(f"参照ソース（上位 {len(results)} 件）")
    render_sources(results)
