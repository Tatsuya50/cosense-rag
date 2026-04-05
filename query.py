import re
import sys
from pathlib import Path

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ── 定数 ──────────────────────────────────────────────────────────────────────
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "cosense"
MODEL_NAME = "intfloat/multilingual-e5-small"
TOP_N = 5
SNIPPET_CHARS = 300
PRECISION_THRESHOLD = 0.5  # この類似度以上を「関連あり」とみなす
VECTOR_WEIGHT = 0.7        # ベクトル検索の重み
KEYWORD_WEIGHT = 0.3       # キーワード検索（BM25）の重み


# ── 埋め込み関数（クエリ側: "query: " プレフィックス） ───────────────────────────
class QueryEmbeddingFunction(EmbeddingFunction):
    """
    クエリ検索用の埋め込み関数。
    multilingual-e5 はクエリに "query: " プレフィックスが必須（文書側の "passage: " と異なる）。
    """

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        texts = ["query: " + text for text in input]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )
        return embeddings.tolist()


# ── トークナイザ（BM25用） ──────────────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    """
    日本語・英語混在テキストのトークン化。
    - 英数字: 単語単位
    - CJK文字: 2文字のbigram（形態素解析なしで日本語BM25を近似）
    """
    tokens = []
    # 英数字トークン（小文字化）
    tokens.extend(re.findall(r"[a-zA-Z0-9]+", text.lower()))
    # CJK文字をbigram化
    cjk_chars = re.findall(
        r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\u3400-\u4dbf]", text
    )
    tokens.extend(cjk_chars[i] + cjk_chars[i + 1] for i in range(len(cjk_chars) - 1))
    return tokens


# ── ハイブリッド検索 ────────────────────────────────────────────────────────────
def hybrid_search(
    collection,
    query_text: str,
    n_results: int,
    vector_weight: float = VECTOR_WEIGHT,
    keyword_weight: float = KEYWORD_WEIGHT,
) -> list[tuple[float, float, float, str, str, dict]]:
    """
    BM25（キーワード）とベクトル検索を組み合わせたハイブリッド検索。

    スコア計算:
        combined = keyword_weight * bm25_normalized + vector_weight * vector_similarity

    Returns: [(combined, bm25_norm, vector_sim, id, document, metadata), ...]
    """
    # ── 全チャンクを取得（BM25インデックス構築用） ──────────────────────────────
    all_data = collection.get(include=["documents", "metadatas"])
    all_ids: list[str] = all_data["ids"]
    all_docs: list[str] = all_data["documents"]
    all_metas: list[dict] = all_data["metadatas"]

    # ── BM25インデックス構築 ────────────────────────────────────────────────────
    corpus_tokenized = [tokenize(doc) for doc in all_docs]
    bm25 = BM25Okapi(corpus_tokenized)

    # ── BM25スコア計算・正規化 ──────────────────────────────────────────────────
    query_tokens = tokenize(query_text)
    bm25_scores = bm25.get_scores(query_tokens)  # shape: (n_docs,)
    max_bm25 = bm25_scores.max()
    bm25_norm = bm25_scores / max_bm25 if max_bm25 > 0 else bm25_scores

    # ── ベクトル検索（候補を多めに取得） ────────────────────────────────────────
    k = min(n_results * 3, len(all_ids))
    vector_results = collection.query(
        query_texts=[query_text],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    # id -> vector similarity のマップ
    vector_sim_map: dict[str, float] = {
        vid: 1.0 - dist
        for vid, dist in zip(vector_results["ids"][0], vector_results["distances"][0])
    }

    # ── 候補をBM25上位とベクトル上位の和集合で構築 ──────────────────────────────
    id_to_idx = {doc_id: i for i, doc_id in enumerate(all_ids)}
    bm25_top_ids = {all_ids[i] for i in bm25_norm.argsort()[::-1][:k]}
    candidate_ids = bm25_top_ids | set(vector_results["ids"][0])

    # ── スコア合成 ──────────────────────────────────────────────────────────────
    scored = []
    for doc_id in candidate_ids:
        idx = id_to_idx[doc_id]
        b_score = float(bm25_norm[idx])
        v_score = vector_sim_map.get(doc_id, 0.0)
        combined = keyword_weight * b_score + vector_weight * v_score
        scored.append((combined, b_score, v_score, doc_id, all_docs[idx], all_metas[idx]))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:n_results]


# ── 結果表示 ───────────────────────────────────────────────────────────────────
def format_result(
    rank: int,
    combined: float,
    bm25_score: float,
    vector_score: float,
    metadata: dict,
    document: str,
) -> str:
    title = metadata.get("title", "(タイトルなし)")

    chunk_info = ""
    total = metadata.get("total_chunks", 1)
    if total > 1:
        chunk_info = f"  [チャンク {metadata['chunk_index'] + 1}/{total}]"

    # スニペット: チャンク先頭のタイトル行を除いた内容
    lines = document.split("\n")
    content_lines = lines[1:] if lines and lines[0] == title else lines
    snippet = " ".join(line.strip() for line in content_lines if line.strip())
    if len(snippet) > SNIPPET_CHARS:
        snippet = snippet[:SNIPPET_CHARS] + "..."

    relevance_mark = "[o]" if combined >= PRECISION_THRESHOLD else "[x]"

    return (
        f"\n{'─' * 60}\n"
        f"#{rank}  {title}{chunk_info}\n"
        f"    合計スコア: {combined:.3f}  {relevance_mark} {'関連あり' if combined >= PRECISION_THRESHOLD else '関連なし'} (閾値: {PRECISION_THRESHOLD})\n"
        f"    ベクトル: {vector_score:.3f} (x{VECTOR_WEIGHT})  キーワード: {bm25_score:.3f} (x{KEYWORD_WEIGHT})\n"
        f"    {snippet}"
    )


# ── メイン ─────────────────────────────────────────────────────────────────────
def main():
    # クエリ取得
    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:])
    else:
        query_text = input("クエリを入力してください: ").strip()

    if not query_text:
        print("クエリが入力されていません。")
        sys.exit(1)

    # ChromaDB が存在するか確認
    if not Path(CHROMA_DIR).exists():
        print(f"エラー: {CHROMA_DIR}/ が見つかりません。先に ingest.py を実行してください。")
        sys.exit(1)

    # ChromaDB 接続
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embedding_fn = QueryEmbeddingFunction(MODEL_NAME)

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
    except Exception:
        print(f"エラー: コレクション '{COLLECTION_NAME}' が見つかりません。先に ingest.py を実行してください。")
        sys.exit(1)

    # ハイブリッド検索
    print(f'\nクエリ: "{query_text}"')
    print(f"上位 {TOP_N} 件（全 {collection.count()} チャンク中）")
    print(f"検索方式: ハイブリッド（ベクトル x{VECTOR_WEIGHT} + キーワード x{KEYWORD_WEIGHT}）\n")

    results = hybrid_search(collection, query_text, TOP_N)

    for rank, (combined, b_score, v_score, _, doc, meta) in enumerate(results, start=1):
        print(format_result(rank, combined, b_score, v_score, meta, doc))

    # Precision@N
    relevant_count = sum(1 for combined, *_ in results if combined >= PRECISION_THRESHOLD)
    precision = relevant_count / len(results)

    print(f"\n{'─' * 60}")
    print(f"Precision@{TOP_N}: {relevant_count}/{len(results)} = {precision:.2f}  (合計スコア >= {PRECISION_THRESHOLD} を「関連あり」と判定)")


if __name__ == "__main__":
    main()
