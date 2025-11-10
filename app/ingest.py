"""
ドキュメント取り込みスクリプト
ドキュメントをロード、分割、ベクトル化してQdrantに保存
"""

import argparse
import sys
from pathlib import Path

from app.config import config
from app.models.embeddings import create_embeddings
from app.vector_store.qdrant_client import QdrantVectorStoreManager
from app.loaders.document_loader import DocumentLoaderManager
from app.utils.text_splitter import create_text_splitter


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="ドキュメントを取り込んでQdrantに保存します"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="ドキュメントファイルまたはディレクトリのパス"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help=f"Qdrantコレクション名（デフォルト: {config.qdrant.collection_name}）"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=f"チャンクサイズ（デフォルト: {config.rag.chunk_size}）"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help=f"チャンクオーバーラップ（デフォルト: {config.rag.chunk_overlap}）"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="既存のコレクションを削除して再作成"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ドキュメント取り込み処理を開始します")
    print("=" * 60)

    try:
        # 1. ドキュメント読み込み
        print("\n[1/5] ドキュメントを読み込んでいます...")
        loader = DocumentLoaderManager()
        source_path = Path(args.source)

        if source_path.is_file():
            documents = loader.load_document(str(source_path))
        elif source_path.is_dir():
            documents = loader.load_directory(str(source_path))
        else:
            print(f"エラー: パスが見つかりません: {args.source}")
            sys.exit(1)

        if not documents:
            print("エラー: 読み込むドキュメントがありません")
            sys.exit(1)

        # 2. テキスト分割
        print("\n[2/5] テキストを分割しています...")
        text_splitter = create_text_splitter(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        split_docs = text_splitter.split_documents(documents)

        if not split_docs:
            print("エラー: 分割されたドキュメントがありません")
            sys.exit(1)

        # 3. 埋め込みモデル初期化
        print("\n[3/5] 埋め込みモデルを初期化しています...")
        embeddings = create_embeddings()
        print(f"埋め込みモデル: {config.ollama.embed_model}")

        # 4. Qdrantクライアント初期化
        print("\n[4/5] Qdrantに接続しています...")
        vector_store_manager = QdrantVectorStoreManager(
            collection_name=args.collection,
            embeddings=embeddings
        )
        vector_store_manager.initialize()

        # コレクション作成
        vector_store_manager.create_collection(force=args.force)

        # 5. ドキュメント追加
        print("\n[5/5] ドキュメントをQdrantに保存しています...")
        print(f"保存するチャンク数: {len(split_docs)}")

        # バッチで処理（大量データ対応）
        batch_size = 100
        total = len(split_docs)

        for i in range(0, total, batch_size):
            batch = split_docs[i:i + batch_size]
            end = min(i + batch_size, total)
            print(f"  処理中: {i + 1}～{end}/{total}")
            vector_store_manager.add_documents(batch)

        # 完了メッセージ
        print("\n" + "=" * 60)
        print("取り込み完了!")
        print("=" * 60)

        # コレクション情報表示
        info = vector_store_manager.get_collection_info()
        if info:
            print(f"\nコレクション情報:")
            print(f"  名前: {info.get('name')}")
            print(f"  ベクトル数: {info.get('vectors_count')}")
            print(f"  ポイント数: {info.get('points_count')}")

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
