#!/usr/bin/env python3
"""
RAGシステムに対してクエリを実行するスクリプト
"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from models.embeddings import create_embeddings
from models.llm import create_llm
from vector_store.qdrant_client import QdrantVectorStoreManager
from prompts.templates import format_documents, create_prompt_with_context


def query_rag(query: str, collection_name: str = "takaichi_sanae_qa", top_k: int = 5):
    """
    RAGシステムにクエリを実行

    Args:
        query: 検索クエリ
        collection_name: コレクション名
        top_k: 取得する類似ドキュメント数
    """
    print(f"\n=== RAG推論実行 ===")
    print(f"コレクション: {collection_name}")
    print(f"質問: {query}")
    print(f"取得ドキュメント数: {top_k}")

    # 1. 埋め込みモデルの初期化
    print("\n[1] 埋め込みモデルを初期化中...")
    embeddings = create_embeddings()

    # 2. Qdrantクライアントの初期化
    print("[2] Qdrantクライアントを初期化中...")
    vector_store_manager = QdrantVectorStoreManager(
        collection_name=collection_name,
        embeddings=embeddings
    )
    vector_store_manager.initialize()

    # 3. 類似度検索
    print(f"[3] 類似ドキュメントを検索中（top_{top_k}）...")
    try:
        results = vector_store_manager.similarity_search_with_score(query, k=top_k)
    except Exception as e:
        print(f"エラー: {e}")
        return

    if not results:
        print("\n⚠️ 関連するドキュメントが見つかりませんでした")
        return

    # 4. 検索結果の表示
    print(f"\n[4] 検索結果:")
    documents = []
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n  --- ドキュメント {i} (スコア: {score:.4f}) ---")
        print(f"  {doc.page_content[:200]}...")
        documents.append(doc)

    # 5. コンテキストの作成
    print("\n[5] コンテキストを作成中...")
    context = format_documents(documents)

    # 6. プロンプトの作成
    print("[6] プロンプトを作成中...")
    prompt = create_prompt_with_context(context, query)

    # 7. LLMの初期化
    print("[7] LLMを初期化中...")
    llm = create_llm()

    # 8. 回答生成
    print("[8] 回答を生成中...\n")
    print("="*70)
    answer = llm.generate(prompt)
    print("\n📝 回答:")
    print("-"*70)
    print(answer)
    print("="*70)

    # 9. 参照情報の表示
    print("\n📚 参照したドキュメント:")
    for i, (doc, score) in enumerate(results, 1):
        question = doc.metadata.get('question', 'N/A')
        answer_text = doc.metadata.get('answer', 'N/A')
        print(f"\n  [{i}] スコア: {score:.4f}")
        print(f"      質問: {question}")
        print(f"      回答: {answer_text}")


def interactive_mode(collection_name: str = "takaichi_sanae_qa"):
    """対話モード"""
    print("\n=== RAG対話モード ===")
    print("質問を入力してください（終了: quit, exit）\n")

    while True:
        try:
            query = input("\n💬 質問 > ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 終了します")
                break

            query_rag(query, collection_name)

        except KeyboardInterrupt:
            print("\n\n👋 終了します")
            break
        except Exception as e:
            print(f"\nエラー: {e}")


def main():
    """メイン処理"""
    # 引数チェック
    if len(sys.argv) < 2:
        print("使用法:")
        print("  単発質問: python query_rag.py <query> [collection_name] [top_k]")
        print("  対話モード: python query_rag.py -i [collection_name]")
        sys.exit(1)

    # 対話モード
    if sys.argv[1] == "-i":
        collection_name = sys.argv[2] if len(sys.argv) > 2 else "takaichi_sanae_qa"
        interactive_mode(collection_name)
        return

    # 単発質問モード
    query = sys.argv[1]
    collection_name = sys.argv[2] if len(sys.argv) > 2 else "takaichi_sanae_qa"
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    query_rag(query, collection_name, top_k)


if __name__ == "__main__":
    main()
