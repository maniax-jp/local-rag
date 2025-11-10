"""
RAG推論フローの統合テスト
※これらのテストは実際のOllama/Qdrantサービスとデータが必要です
"""

import pytest
from langchain_core.documents import Document
from models.llm import create_llm
from models.embeddings import create_embeddings
from vector_store.qdrant_client import QdrantVectorStoreManager
from prompts.templates import format_documents, create_prompt_with_context


@pytest.mark.integration
class TestQueryFlow:
    """RAG推論フロー統合テスト"""

    @pytest.fixture(scope="class")
    def setup_test_data(self, test_collection_name):
        """テストデータのセットアップ"""
        # テスト用ドキュメントを準備
        test_documents = [
            Document(
                page_content="東京タワーの高さは333メートルです。1958年に完成しました。",
                metadata={"source": "tokyo_tower.txt"}
            ),
            Document(
                page_content="富士山は日本最高峰の山で、標高は3,776メートルです。",
                metadata={"source": "fujisan.txt"}
            ),
            Document(
                page_content="スカイツリーの高さは634メートルで、東京にあります。",
                metadata={"source": "skytree.txt"}
            )
        ]

        # ベクトルストアに保存
        embeddings = create_embeddings()
        vector_store_manager = QdrantVectorStoreManager(
            collection_name=test_collection_name,
            embeddings=embeddings
        )
        vector_store_manager.initialize()
        vector_store_manager.create_collection(force=True)
        vector_store_manager.add_documents(test_documents)

        yield vector_store_manager

        # クリーンアップ
        try:
            vector_store_manager.delete_collection()
        except:
            pass

    def test_e2e_rag_inference(self, setup_test_data):
        """E2E: RAG推論フロー全体のテスト"""
        vector_store_manager = setup_test_data

        # 1. クエリをベクトル化して類似検索
        query = "東京タワーの高さは？"
        results = vector_store_manager.similarity_search(query, k=2)

        assert len(results) > 0
        assert any("東京タワー" in doc.page_content for doc in results)

        # 2. コンテキストを生成
        context = format_documents(results)
        assert "東京タワー" in context

        # 3. プロンプトを生成
        prompt = create_prompt_with_context(context, query)
        assert query in prompt
        assert context in prompt

        # 4. LLMで推論（実際のLLM呼び出しはスキップ可能）
        # llm = create_llm()
        # response = llm.invoke(prompt)
        # assert response is not None

    def test_similarity_search_with_score(self, setup_test_data):
        """スコア付き類似度検索のテスト"""
        vector_store_manager = setup_test_data

        query = "富士山について"
        results = vector_store_manager.similarity_search_with_score(query, k=3)

        assert len(results) > 0

        # スコア付き結果の確認
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)
            assert score >= 0

        # 最も関連性の高いドキュメントを確認
        top_doc, top_score = results[0]
        assert "富士山" in top_doc.page_content

    def test_context_not_found(self, test_collection_name):
        """関連コンテキストが見つからない場合のテスト"""
        # 空のコレクションを作成
        embeddings = create_embeddings()
        vector_store_manager = QdrantVectorStoreManager(
            collection_name=test_collection_name + "_empty",
            embeddings=embeddings
        )
        vector_store_manager.initialize()
        vector_store_manager.create_collection(force=True)

        # 検索実行
        query = "存在しない情報について"
        results = vector_store_manager.similarity_search(query, k=3)

        # 結果が空であることを確認
        assert len(results) == 0

        # コンテキストフォーマット
        context = format_documents(results)
        assert "関連する情報が見つかりませんでした" in context

        # クリーンアップ
        try:
            vector_store_manager.delete_collection()
        except:
            pass

    def test_top_k_parameter(self, setup_test_data):
        """Top-Kパラメータのテスト"""
        vector_store_manager = setup_test_data

        query = "タワーについて"

        # K=1の場合
        results_1 = vector_store_manager.similarity_search(query, k=1)
        assert len(results_1) == 1

        # K=2の場合
        results_2 = vector_store_manager.similarity_search(query, k=2)
        assert len(results_2) == 2

        # K=10の場合（データより多い）
        results_10 = vector_store_manager.similarity_search(query, k=10)
        assert len(results_10) <= 3  # テストデータは3件

    def test_japanese_query_handling(self, setup_test_data):
        """日本語クエリの処理テスト"""
        vector_store_manager = setup_test_data

        # 様々な日本語クエリ
        queries = [
            "東京タワーの高さは何メートルですか？",
            "富士山について教えてください",
            "一番高いのはどれですか",
        ]

        for query in queries:
            results = vector_store_manager.similarity_search(query, k=2)
            assert len(results) > 0
            # 日本語が含まれていることを確認
            assert any(
                any(char >= '\u3040' and char <= '\u309F' or  # ひらがな
                    char >= '\u30A0' and char <= '\u30FF' or  # カタカナ
                    char >= '\u4E00' and char <= '\u9FFF'     # 漢字
                    for char in doc.page_content)
                for doc in results
            )

    def test_metadata_in_search_results(self, setup_test_data):
        """検索結果にメタデータが含まれることを確認"""
        vector_store_manager = setup_test_data

        query = "東京タワー"
        results = vector_store_manager.similarity_search(query, k=1)

        assert len(results) > 0
        doc = results[0]
        assert "source" in doc.metadata
        assert doc.metadata["source"] is not None

    @pytest.mark.slow
    def test_prompt_template_integration(self, setup_test_data):
        """プロンプトテンプレート統合テスト"""
        vector_store_manager = setup_test_data

        # 検索
        query = "東京タワーと富士山を比較してください"
        results = vector_store_manager.similarity_search(query, k=3)

        # コンテキスト生成
        context = format_documents(results)

        # プロンプト生成
        prompt = create_prompt_with_context(context, query)

        # プロンプトの内容確認
        assert query in prompt
        assert "東京タワー" in prompt or "富士山" in prompt
        assert "コンテキスト" in prompt
        assert "質問" in prompt

    def test_empty_query_handling(self, setup_test_data):
        """空クエリの処理テスト"""
        vector_store_manager = setup_test_data

        # 空文字列のクエリ
        with pytest.raises(Exception):
            vector_store_manager.similarity_search("", k=1)
