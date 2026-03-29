import json
import re
import sys
from pathlib import Path

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

# ── 定数 ──────────────────────────────────────────────────────────────────────
DATA_FILE = Path("tipsofmlandcs.json")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "cosense"
MODEL_NAME = "intfloat/multilingual-e5-small"
MAX_CHARS_PER_CHUNK = 1500
BATCH_SIZE = 32


# ── 埋め込み関数 ───────────────────────────────────────────────────────────────
class PassageEmbeddingFunction(EmbeddingFunction):
    """
    文書保存用の埋め込み関数。
    multilingual-e5 は文書に "passage: " プレフィックスが必須。
    """

    def __init__(self, model_name: str):
        print(f"  モデルをロード中: {model_name}")
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        texts = ["passage: " + text for text in input]
        embeddings = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()


# ── テキスト処理 ───────────────────────────────────────────────────────────────
def clean_line(line: str) -> str:
    """アイコン記法（ノイズ）を除去し、他のマークアップはそのまま保持。"""
    return re.sub(r"\[[^\]]*\.icon\]", "", line).rstrip()


def page_to_chunks(page: dict) -> list[tuple[str, dict]]:
    """
    Scrapboxページを1つ以上の (テキスト, メタデータ) タプルに変換する。

    - lines[0] は常に title と同一なのでスキップし、タイトルを先頭に明示的に付与。
    - 1500文字超のページは分割し、各チャンクの先頭にタイトルを付与。
    """
    title = page["title"]
    page_id = page["id"]

    # lines[0]（タイトルの重複）をスキップし、残りを整形
    content_lines = [clean_line(line) for line in page["lines"][1:]]

    # 前後の空行を除去
    while content_lines and content_lines[0] == "":
        content_lines.pop(0)
    while content_lines and content_lines[-1] == "":
        content_lines.pop()

    full_text = title + "\n" + "\n".join(content_lines)

    if len(full_text) <= MAX_CHARS_PER_CHUNK:
        metadata = {
            "title": title,
            "page_id": page_id,
            "chunk_index": 0,
            "total_chunks": 1,
            "updated": page.get("updated", 0),
        }
        return [(full_text, metadata)]

    # 1500文字超 → 行単位で分割、各チャンクにタイトルを付与
    chunks: list[str] = []
    chunk_lines = [title]
    for line in content_lines:
        chunk_lines.append(line)
        if len("\n".join(chunk_lines)) >= MAX_CHARS_PER_CHUNK:
            chunks.append("\n".join(chunk_lines))
            chunk_lines = [title]

    if len(chunk_lines) > 1:
        chunks.append("\n".join(chunk_lines))

    total = len(chunks)
    return [
        (
            text,
            {
                "title": title,
                "page_id": page_id,
                "chunk_index": i,
                "total_chunks": total,
                "updated": page.get("updated", 0),
            },
        )
        for i, text in enumerate(chunks)
    ]


# ── メイン ─────────────────────────────────────────────────────────────────────
def main():
    # データ読み込み
    print(f"{DATA_FILE} を読み込み中...")
    with DATA_FILE.open(encoding="utf-8") as f:
        data = json.load(f)
    pages = data["pages"]
    print(f"  {len(pages)} ページ読み込み完了")

    # チャンク生成
    all_chunks: list[tuple[str, dict]] = []
    for page in pages:
        all_chunks.extend(page_to_chunks(page))
    print(f"  {len(all_chunks)} チャンク生成（大きいページを分割済み）")

    # ChromaDB 初期化
    CHROMA_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # 既存コレクションを削除（再実行時の冪等性確保）
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  既存コレクション '{COLLECTION_NAME}' を削除")
    except Exception:
        pass

    embedding_fn = PassageEmbeddingFunction(MODEL_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB 用データ準備
    ids = []
    documents = []
    metadatas = []
    for text, meta in all_chunks:
        chunk_id = f"{meta['page_id']}_{meta['chunk_index']}"
        ids.append(chunk_id)
        documents.append(text)
        metadatas.append(meta)

    # バッチ処理でインジェスト
    print(f"\n{len(documents)} チャンクを埋め込み・保存中...")
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )
        done = min(i + batch_size, len(documents))
        print(f"  [{done:3d}/{len(documents)}] 完了", end="\r")

    print(f"\n完了。{collection.count()} ドキュメントを {CHROMA_DIR}/ に保存しました。")


if __name__ == "__main__":
    main()
