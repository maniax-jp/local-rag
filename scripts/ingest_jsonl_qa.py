#!/usr/bin/env python3
"""
JSONL形式のQ&Aデータをドキュメントとして登録するスクリプト
"""

import json
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from langchain_core.documents import Document
from models.embeddings import create_embeddings
from vector_store.qdrant_client import QdrantVectorStoreManager
from utils.text_splitter import create_text_splitter


def load_jsonl_qa(file_path: str) -> list[Document]:
    """
    JSONL形式のQ&Aファイルを読み込んでDocumentオブジェクトに変換

    Args:
        file_path: JSONLファイルのパス

    Returns:
        Documentオブジェクトのリスト
    """
    documents = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                messages = data.get('messages', [])

                # ユーザーの質問とアシスタントの回答を抽出
                question = None
                answer = None

                for msg in messages:
                    if msg.get('role') == 'user':
                        question = msg.get('content')
                    elif msg.get('role') == 'assistant':
                        answer = msg.get('content')

                if question and answer:
                    # Q&A形式のテキストを作成
                    content = f"質問: {question}\n回答: {answer}"

                    # Documentオブジェクトを作成
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": Path(file_path).name,
                            "line_number": line_no,
                            "question": question,
                            "answer": answer
                        }
                    )
                    documents.append(doc)

            except json.JSONDecodeError as e:
                print(f"警告: {line_no}行目のJSONパースに失敗しました: {e}")
                continue

    print(f"\n{len(documents)}件のQ&Aを読み込みました")
    return documents


def main():
    """メイン処理"""
    # 引数チェック
    if len(sys.argv) < 2:
        print("使用法: python ingest_jsonl_qa.py <jsonl_file_path> [collection_name]")
        sys.exit(1)

    jsonl_file = sys.argv[1]
    collection_name = sys.argv[2] if len(sys.argv) > 2 else "takaichi_sanae_qa"

    # ファイル存在チェック
    if not Path(jsonl_file).exists():
        print(f"エラー: ファイルが見つかりません: {jsonl_file}")
        sys.exit(1)

    print(f"\n=== JSONL Q&Aデータの登録 ===")
    print(f"ファイル: {jsonl_file}")
    print(f"コレクション名: {collection_name}")

    # 1. JSONLファイルの読み込み
    print("\n[1] JSONLファイルを読み込み中...")
    documents = load_jsonl_qa(jsonl_file)

    if not documents:
        print("エラー: 有効なQ&Aデータが見つかりませんでした")
        sys.exit(1)

    # 2. テキスト分割（短いQ&Aなのでそのまま使用）
    print("\n[2] ドキュメントを分割中...")
    text_splitter = create_text_splitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)

    # 3. 埋め込みモデルの初期化
    print("\n[3] 埋め込みモデルを初期化中...")
    embeddings = create_embeddings()

    # 4. Qdrantクライアントの初期化
    print("\n[4] Qdrantクライアントを初期化中...")
    vector_store_manager = QdrantVectorStoreManager(
        collection_name=collection_name,
        embeddings=embeddings
    )
    vector_store_manager.initialize()

    # 5. コレクションの作成（既存の場合は削除して再作成）
    print("\n[5] コレクションを作成中...")
    vector_store_manager.create_collection(force=True)

    # 6. ドキュメントの追加
    print("\n[6] ドキュメントをベクターストアに追加中...")
    batch_size = 50
    total_added = 0

    for i in range(0, len(split_docs), batch_size):
        batch = split_docs[i:i + batch_size]
        ids = vector_store_manager.add_documents(batch)
        total_added += len(ids)
        print(f"  進捗: {total_added}/{len(split_docs)}")

    # 7. 登録結果の確認
    print("\n[7] 登録結果を確認中...")
    info = vector_store_manager.get_collection_info()
    print(f"\nコレクション情報:")
    print(f"  - 名前: {info['name']}")
    print(f"  - ポイント数: {info['points_count']}")
    print(f"  - ステータス: {info['status']}")

    print("\n✅ 登録が完了しました！")


if __name__ == "__main__":
    main()
