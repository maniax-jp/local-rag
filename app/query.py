"""
RAG推論スクリプト
質問に対してRAG推論を実行
"""

import argparse
import sys

from config import config
from models.llm import create_llm
from models.embeddings import create_embeddings
from vector_store.qdrant_client import QdrantVectorStoreManager
from prompts.templates import format_documents, create_prompt_with_context


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="質問に対してRAG推論を実行します"
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="質問文"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help=f"Qdrantコレクション名（デフォルト: {config.qdrant.collection_name}）"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=f"取得するコンテキスト数（デフォルト: {config.rag.top_k}）"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=f"LLM温度パラメータ（デフォルト: {config.rag.temperature}）"
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="取得したコンテキストを表示"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("RAG推論を実行します")
    print("=" * 60)

    try:
        # 1. 埋め込みモデル初期化
        print("\n[1/5] 埋め込みモデルを初期化しています...")
        embeddings = create_embeddings()

        # 2. Qdrantクライアント初期化
        print("[2/5] Qdrantに接続しています...")
        vector_store_manager = QdrantVectorStoreManager(
            collection_name=args.collection,
            embeddings=embeddings
        )
        vector_store_manager.initialize()

        # コレクションの存在確認
        info = vector_store_manager.get_collection_info()
        if not info or info.get('points_count', 0) == 0:
            print(f"\nエラー: コレクション '{args.collection or config.qdrant.collection_name}' にデータがありません")
            print("まずingest.pyでドキュメントを取り込んでください")
            sys.exit(1)

        print(f"コレクション: {info.get('name')} ({info.get('points_count')}件)")

        # 3. 類似度検索
        print(f"[3/5] 類似ドキュメントを検索しています...")
        print(f"質問: {args.question}")

        top_k = args.top_k or config.rag.top_k
        results = vector_store_manager.similarity_search_with_score(
            query=args.question,
            k=top_k
        )

        if not results:
            print("\n関連するドキュメントが見つかりませんでした")
            sys.exit(0)

        print(f"見つかったドキュメント: {len(results)}件")

        # コンテキスト表示
        if args.show_context:
            print("\n" + "-" * 60)
            print("取得したコンテキスト:")
            print("-" * 60)
            for i, (doc, score) in enumerate(results, 1):
                print(f"\n[{i}] スコア: {score:.4f}")
                print(f"出典: {doc.metadata.get('file_name', '不明')}")
                print(f"内容: {doc.page_content[:200]}...")
            print("-" * 60)

        # 4. LLM初期化
        print("\n[4/5] LLMを初期化しています...")
        llm = create_llm(temperature=args.temperature)
        print(f"モデル: {config.ollama.llm_model}")

        # 5. プロンプト生成と推論
        print("[5/5] 回答を生成しています...")

        # ドキュメントをコンテキストに変換
        docs_only = [doc for doc, _ in results]
        context = format_documents(docs_only)

        # プロンプト生成
        prompt = create_prompt_with_context(context, args.question)

        # LLM推論
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)

        # 結果表示
        print("\n" + "=" * 60)
        print("回答:")
        print("=" * 60)
        print(answer)
        print("=" * 60)

        # 参照ドキュメント
        print("\n参照したドキュメント:")
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get('file_name', '不明')
            print(f"  [{i}] {source} (スコア: {score:.4f})")

    except KeyboardInterrupt:
        print("\n\n処理が中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
