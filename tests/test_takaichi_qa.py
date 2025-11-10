"""
高市早苗Q&Aデータセットに対するE2Eテスト
"""

import pytest
from models.embeddings import create_embeddings
from models.llm import SwallowLLM
from vector_store.qdrant_client import QdrantVectorStoreManager
from prompts.templates import format_documents, create_prompt_with_context


@pytest.mark.integration
class TestTakaichiQA:
    """高市早苗Q&AデータセットのE2Eテスト"""

    @pytest.fixture(scope="class")
    def setup_qa_system(self):
        """Q&Aシステムのセットアップ"""
        collection_name = "takaichi_qa"

        # 埋め込みモデルの初期化
        embeddings = create_embeddings()

        # Qdrantクライアントの初期化
        vector_store_manager = QdrantVectorStoreManager(
            collection_name=collection_name,
            embeddings=embeddings
        )
        vector_store_manager.initialize()

        # LLMインスタンスの作成
        llm = SwallowLLM()
        llm.initialize()

        return {
            "vector_store_manager": vector_store_manager,
            "llm": llm
        }

    def test_query_birth_date(self, setup_qa_system):
        """生年月日に関する質問テスト"""
        system = setup_qa_system
        query = "高市早苗さんの生年月日は？"

        # 類似度検索
        results = system["vector_store_manager"].similarity_search(query, k=3)
        assert len(results) > 0, "検索結果が空です"

        # コンテキスト作成
        context = format_documents(results)
        assert len(context) > 0, "コンテキストが空です"

        # プロンプト作成
        prompt = create_prompt_with_context(context, query)
        assert query in prompt, "プロンプトに質問が含まれていません"

        # LLM推論
        answer = system["llm"].generate(prompt)
        assert len(answer) > 0, "回答が空です"
        assert "1961" in answer or "昭和36" in answer, "正しい生年月が含まれていません"

        print(f"\n質問: {query}")
        print(f"回答: {answer}")

    def test_query_university(self, setup_qa_system):
        """出身大学に関する質問テスト"""
        system = setup_qa_system
        query = "高市早苗さんの出身大学は？"

        # 類似度検索
        results = system["vector_store_manager"].similarity_search(query, k=3)
        assert len(results) > 0

        # コンテキスト作成と推論
        context = format_documents(results)
        prompt = create_prompt_with_context(context, query)
        answer = system["llm"].generate(prompt)

        assert len(answer) > 0
        assert "神戸大学" in answer, "正しい大学名が含まれていません"

        print(f"\n質問: {query}")
        print(f"回答: {answer}")

    def test_query_political_party(self, setup_qa_system):
        """所属政党に関する質問テスト"""
        system = setup_qa_system
        query = "高市早苗さんは何党ですか？"

        # 類似度検索
        results = system["vector_store_manager"].similarity_search(query, k=3)
        assert len(results) > 0

        # コンテキスト作成と推論
        context = format_documents(results)
        prompt = create_prompt_with_context(context, query)
        answer = system["llm"].generate(prompt)

        assert len(answer) > 0
        assert "自民" in answer or "自由民主党" in answer, "正しい政党名が含まれていません"

        print(f"\n質問: {query}")
        print(f"回答: {answer}")

    def test_query_minister_role(self, setup_qa_system):
        """大臣経験に関する質問テスト"""
        system = setup_qa_system
        query = "高市早苗さんはどの大臣を経験しましたか？"

        # 類似度検索
        results = system["vector_store_manager"].similarity_search(query, k=5)
        assert len(results) > 0

        # コンテキスト作成と推論
        context = format_documents(results)
        prompt = create_prompt_with_context(context, query)
        answer = system["llm"].generate(prompt)

        assert len(answer) > 0
        assert "総務大臣" in answer or "経済安全保障" in answer, "大臣職が含まれていません"

        print(f"\n質問: {query}")
        print(f"回答: {answer}")

    def test_query_election_district(self, setup_qa_system):
        """選挙区に関する質問テスト"""
        system = setup_qa_system
        query = "高市早苗さんの選挙区はどこですか？"

        # 類似度検索
        results = system["vector_store_manager"].similarity_search(query, k=3)
        assert len(results) > 0

        # コンテキスト作成と推論
        context = format_documents(results)
        prompt = create_prompt_with_context(context, query)
        answer = system["llm"].generate(prompt)

        assert len(answer) > 0
        assert "奈良" in answer and "2区" in answer, "正しい選挙区情報が含まれていません"

        print(f"\n質問: {query}")
        print(f"回答: {answer}")

    def test_similarity_search_with_score(self, setup_qa_system):
        """スコア付き類似度検索のテスト"""
        system = setup_qa_system
        query = "高市早苗さんの政治経歴について教えてください"

        # スコア付き検索
        results = system["vector_store_manager"].similarity_search_with_score(query, k=5)
        assert len(results) > 0, "検索結果が空です"

        # スコアの確認
        for doc, score in results:
            assert score >= 0, "スコアが負の値です"
            assert doc.page_content is not None, "ドキュメント内容が空です"

        # スコアでソートされていることを確認
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True), "スコアでソートされていません"

        print(f"\n質問: {query}")
        print("\n検索結果:")
        for i, (doc, score) in enumerate(results, 1):
            print(f"  [{i}] スコア: {score:.4f}")
            print(f"      内容: {doc.page_content[:100]}...")

    def test_multiple_queries(self, setup_qa_system):
        """複数の質問に対する連続推論テスト"""
        system = setup_qa_system

        queries = [
            "高市早苗さんは何年生まれですか？",
            "高市早苗さんの出身地はどこですか？",
            "高市早苗さんは何期目の議員ですか？"
        ]

        for query in queries:
            # 類似度検索
            results = system["vector_store_manager"].similarity_search(query, k=3)
            assert len(results) > 0, f"質問「{query}」で検索結果が空です"

            # コンテキスト作成と推論
            context = format_documents(results)
            prompt = create_prompt_with_context(context, query)
            answer = system["llm"].generate(prompt)

            assert len(answer) > 0, f"質問「{query}」で回答が空です"

            print(f"\n質問: {query}")
            print(f"回答: {answer}")

    def test_context_relevance(self, setup_qa_system):
        """コンテキストの関連性テスト"""
        system = setup_qa_system
        query = "高市早苗さんの総務大臣としての業績は？"

        # 類似度検索
        results = system["vector_store_manager"].similarity_search(query, k=5)
        assert len(results) > 0

        # 各ドキュメントに関連キーワードが含まれていることを確認
        relevant_keywords = ["総務大臣", "総務", "大臣", "業績", "高市"]
        for doc in results:
            content = doc.page_content.lower()
            has_keyword = any(keyword.lower() in content for keyword in relevant_keywords)
            assert has_keyword, f"関連キーワードが含まれていません: {doc.page_content[:100]}"

        print(f"\n質問: {query}")
        print("\n関連ドキュメント:")
        for i, doc in enumerate(results, 1):
            print(f"  [{i}] {doc.page_content[:150]}...")
