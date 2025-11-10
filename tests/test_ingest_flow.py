"""
ドキュメント取り込みフローの統合テスト
※これらのテストは実際のOllama/Qdrantサービスが必要です
"""

import pytest
from pathlib import Path
from app.models.embeddings import create_embeddings
from app.vector_store.qdrant_client import QdrantVectorStoreManager
from app.loaders.document_loader import DocumentLoaderManager
from app.utils.text_splitter import create_text_splitter


@pytest.mark.integration
class TestIngestFlow:
    """ドキュメント取り込みフロー統合テスト"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self, test_collection_name):
        """テスト前後の処理"""
        # テスト後のクリーンアップ
        yield
        # テスト用コレクションの削除（エラーは無視）
        try:
            embeddings = create_embeddings()
            vector_store_manager = QdrantVectorStoreManager(
                collection_name=test_collection_name,
                embeddings=embeddings
            )
            vector_store_manager.initialize()
            vector_store_manager.delete_collection()
        except:
            pass

    def test_e2e_ingest_single_document(self, sample_txt_path, test_collection_name):
        """E2E: 単一ドキュメントの取り込みフロー"""
        # 1. ドキュメント読み込み
        loader = DocumentLoaderManager()
        documents = loader.load_document(sample_txt_path)
        assert len(documents) > 0

        # 2. テキスト分割
        text_splitter = create_text_splitter(chunk_size=200, chunk_overlap=50)
        split_docs = text_splitter.split_documents(documents)
        assert len(split_docs) > 0

        # 3. 埋め込みモデル初期化
        embeddings = create_embeddings()
        assert embeddings is not None

        # 4. Qdrantクライアント初期化
        vector_store_manager = QdrantVectorStoreManager(
            collection_name=test_collection_name,
            embeddings=embeddings
        )
        vector_store_manager.initialize()

        # 5. コレクション作成
        vector_store_manager.create_collection(force=True)

        # 6. ドキュメント追加
        ids = vector_store_manager.add_documents(split_docs)
        assert len(ids) == len(split_docs)

        # 7. コレクション情報確認
        info = vector_store_manager.get_collection_info()
        assert info["points_count"] > 0

    def test_e2e_ingest_directory(self, fixtures_dir, test_collection_name):
        """E2E: ディレクトリ一括取り込みフロー"""
        # 1. ディレクトリ読み込み
        loader = DocumentLoaderManager()
        documents = loader.load_directory(str(fixtures_dir), recursive=False)
        assert len(documents) > 0

        # 2. テキスト分割
        text_splitter = create_text_splitter()
        split_docs = text_splitter.split_documents(documents)
        assert len(split_docs) > 0

        # 3. 埋め込みとベクトルストア
        embeddings = create_embeddings()
        vector_store_manager = QdrantVectorStoreManager(
            collection_name=test_collection_name,
            embeddings=embeddings
        )
        vector_store_manager.initialize()
        vector_store_manager.create_collection(force=True)

        # 4. ドキュメント追加
        ids = vector_store_manager.add_documents(split_docs)
        assert len(ids) > 0

    def test_mixed_format_ingest(self, fixtures_dir, test_collection_name):
        """混合形式ドキュメントの取り込みテスト"""
        loader = DocumentLoaderManager()

        # 各形式のドキュメントを個別に読み込み
        all_documents = []

        for file_path in Path(fixtures_dir).iterdir():
            if file_path.suffix in loader.SUPPORTED_EXTENSIONS:
                try:
                    docs = loader.load_document(str(file_path))
                    all_documents.extend(docs)
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")

        assert len(all_documents) > 0

        # テキスト分割
        text_splitter = create_text_splitter()
        split_docs = text_splitter.split_documents(all_documents)

        # ベクトルストアに保存
        embeddings = create_embeddings()
        vector_store_manager = QdrantVectorStoreManager(
            collection_name=test_collection_name,
            embeddings=embeddings
        )
        vector_store_manager.initialize()
        vector_store_manager.create_collection(force=True)
        vector_store_manager.add_documents(split_docs)

        # 保存確認
        info = vector_store_manager.get_collection_info()
        assert info["points_count"] > 0

    def test_ingest_with_metadata_preservation(self, sample_txt_path, test_collection_name):
        """メタデータが保持されることを確認"""
        # ドキュメント読み込み
        loader = DocumentLoaderManager()
        documents = loader.load_document(sample_txt_path)

        # メタデータの確認
        assert "file_name" in documents[0].metadata
        assert "file_extension" in documents[0].metadata

        # 分割後もメタデータが保持されることを確認
        text_splitter = create_text_splitter(chunk_size=100, chunk_overlap=20)
        split_docs = text_splitter.split_documents(documents)

        for doc in split_docs:
            assert "file_name" in doc.metadata
            assert doc.metadata["file_name"] == "sample.txt"

    @pytest.mark.slow
    def test_large_document_ingest(self, sample_long_text, test_collection_name):
        """大きなドキュメントの取り込みテスト"""
        from langchain_core.documents import Document

        # 大きなドキュメントを作成
        large_doc = Document(
            page_content=sample_long_text,
            metadata={"source": "large_test"}
        )

        # テキスト分割
        text_splitter = create_text_splitter(chunk_size=500, chunk_overlap=100)
        split_docs = text_splitter.split_documents([large_doc])
        assert len(split_docs) > 5  # 複数のチャンクに分割される

        # ベクトルストアに保存
        embeddings = create_embeddings()
        vector_store_manager = QdrantVectorStoreManager(
            collection_name=test_collection_name,
            embeddings=embeddings
        )
        vector_store_manager.initialize()
        vector_store_manager.create_collection(force=True)

        # バッチで追加
        batch_size = 10
        for i in range(0, len(split_docs), batch_size):
            batch = split_docs[i:i + batch_size]
            vector_store_manager.add_documents(batch)

        # 保存確認
        info = vector_store_manager.get_collection_info()
        assert info["points_count"] == len(split_docs)
