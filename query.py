import sys
from pathlib import Path

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

# ── 定数 ──────────────────────────────────────────────────────────────────────
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "cosense"
MODEL_NAME = "intfloat/multilingual-e5-small"
TOP_N = 5
SNIPPET_CHARS = 300
PRECISION_THRESHOLD = 0.5  # この類似度以上を「関連あり」とみなす


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


# ── 結果表示 ───────────────────────────────────────────────────────────────────
def format_result(rank: int, distance: float, metadata: dict, document: str) -> str:
    # cosine距離 → 類似度スコア（1.0 = 完全一致、0.0 = 直交）
    score = 1.0 - distance
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

    relevance_mark = "[o]" if score >= PRECISION_THRESHOLD else "[x]"

    return (
        f"\n{'─' * 60}\n"
        f"#{rank}  {title}{chunk_info}\n"
        f"    類似度: {score:.3f}  {relevance_mark} {'関連あり' if score >= PRECISION_THRESHOLD else '関連なし'} (閾値: {PRECISION_THRESHOLD})\n"
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

    # ベクトル検索
    results = collection.query(
        query_texts=[query_text],
        n_results=TOP_N,
        include=["documents", "metadatas", "distances"],
    )

    # 結果表示
    print(f'\nクエリ: "{query_text}"')
    print(f"上位 {TOP_N} 件（全 {collection.count()} チャンク中）:\n")

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        print(format_result(rank, dist, meta, doc))

    # Precision@N: 関連ありチャンク数 / 取得件数
    scores = [1.0 - d for d in distances]
    relevant_count = sum(1 for s in scores if s >= PRECISION_THRESHOLD)
    precision = relevant_count / len(scores)

    print(f"\n{'─' * 60}")
    print(f"Precision@{TOP_N}: {relevant_count}/{len(scores)} = {precision:.2f}  (類似度 >= {PRECISION_THRESHOLD} を「関連あり」と判定)")


if __name__ == "__main__":
    main()
