"""
メインエントリーポイント
対話型RAGインターフェース
"""

import argparse
import sys

from app.config import config
from app.models.llm import create_llm
from app.models.embeddings import create_embeddings
from app.vector_store.qdrant_client import QdrantVectorStoreManager
from app.prompts.templates import format_documents, create_prompt_with_context


def interactive_mode(collection_name: str = None, top_k: int = None, temperature: float = None):
    """
    対話型モード

    Args:
        collection_name: コレクション名
        top_k: 取得するコンテキスト数
        temperature: LLM温度パラメータ
    """
    print("=" * 60)
    print("対話型RAGシステム")
    print("=" * 60)
    print("\nコマンド:")
    print("  - 質問を入力してEnterキー")
    print("  - 'exit' または 'quit' で終了")
    print("  - 'info' でシステム情報表示")
    print("=" * 60)

    try:
        # 初期化
        print("\nシステムを初期化しています...")

        embeddings = create_embeddings()
        print(f"✓ 埋め込みモデル: {config.ollama.embed_model}")

        vector_store_manager = QdrantVectorStoreManager(
            collection_name=collection_name,
            embeddings=embeddings
        )
        vector_store_manager.initialize()

        info = vector_store_manager.get_collection_info()
        if not info or info.get('points_count', 0) == 0:
            print(f"\n警告: コレクション '{collection_name or config.qdrant.collection_name}' にデータがありません")
            print("まずingest.pyでドキュメントを取り込んでください")
            sys.exit(1)

        print(f"✓ コレクション: {info.get('name')} ({info.get('points_count')}件)")

        llm = create_llm(temperature=temperature)
        print(f"✓ LLM: {config.ollama.llm_model}")

        k = top_k or config.rag.top_k
        print(f"✓ Top-K: {k}")

        print("\n準備完了! 質問を入力してください。\n")

        # 対話ループ
        while True:
            try:
                # ユーザー入力
                question = input("質問> ").strip()

                if not question:
                    continue

                # 終了コマンド
                if question.lower() in ['exit', 'quit', 'q']:
                    print("\n終了します。")
                    break

                # 情報表示コマンド
                if question.lower() == 'info':
                    print("\nシステム情報:")
                    print(f"  LLM: {config.ollama.llm_model}")
                    print(f"  埋め込みモデル: {config.ollama.embed_model}")
                    print(f"  コレクション: {info.get('name')}")
                    print(f"  ドキュメント数: {info.get('points_count')}")
                    print(f"  Top-K: {k}")
                    print(f"  温度: {temperature or config.rag.temperature}")
                    print()
                    continue

                # RAG推論実行
                print("\n検索中...")
                results = vector_store_manager.similarity_search_with_score(
                    query=question,
                    k=k
                )

                if not results:
                    print("関連するドキュメントが見つかりませんでした。\n")
                    continue

                print(f"見つかったドキュメント: {len(results)}件")
                print("回答を生成中...\n")

                # コンテキスト生成
                docs_only = [doc for doc, _ in results]
                context = format_documents(docs_only)

                # プロンプト生成
                prompt = create_prompt_with_context(context, question)

                # LLM推論
                response = llm.invoke(prompt)
                answer = response.content if hasattr(response, 'content') else str(response)

                # 回答表示
                print("-" * 60)
                print("回答:")
                print("-" * 60)
                print(answer)
                print("-" * 60)

                # 参照ドキュメント
                print("\n参照:")
                for i, (doc, score) in enumerate(results, 1):
                    source = doc.metadata.get('file_name', '不明')
                    print(f"  [{i}] {source} (スコア: {score:.4f})")
                print()

            except KeyboardInterrupt:
                print("\n\n終了します。")
                break
            except Exception as e:
                print(f"\nエラー: {str(e)}\n")
                continue

    except Exception as e:
        print(f"\n初期化エラー: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Local RAG Application"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="対話型モードで起動"
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

    args = parser.parse_args()

    if args.interactive:
        interactive_mode(
            collection_name=args.collection,
            top_k=args.top_k,
            temperature=args.temperature
        )
    else:
        print("Local RAG Application")
        print("\n使用方法:")
        print("  対話モード:         python main.py --interactive")
        print("  ドキュメント取り込み: python ingest.py --source /documents")
        print("  質問実行:          python query.py --question '質問内容'")
        print("\n詳細は --help を参照してください")


if __name__ == "__main__":
    main()
