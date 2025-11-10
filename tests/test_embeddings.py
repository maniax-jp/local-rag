"""
埋め込みモデルモジュールのテスト
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from models.embeddings import OllamaEmbeddings, create_embeddings


class TestOllamaEmbeddings:
    """OllamaEmbeddingsクラスのテスト"""

    def test_init_with_defaults(self):
        """デフォルト値での初期化テスト"""
        embeddings = OllamaEmbeddings()

        assert embeddings.model is not None
        assert embeddings.base_url is not None
        assert "http" in embeddings.base_url

    def test_init_with_custom_model(self):
        """カスタムモデルでの初期化テスト"""
        embeddings = OllamaEmbeddings(model="custom-embed-model")

        assert embeddings.model == "custom-embed-model"

    @patch('models.embeddings.LangChainOllamaEmbeddings')
    def test_initialize_success(self, mock_langchain_embeddings):
        """埋め込みモデル初期化成功のテスト"""
        mock_instance = MagicMock()
        mock_langchain_embeddings.return_value = mock_instance

        embeddings = OllamaEmbeddings()
        result = embeddings.initialize()

        assert result == mock_instance
        assert embeddings._embeddings == mock_instance
        mock_langchain_embeddings.assert_called_once()

    @patch('models.embeddings.LangChainOllamaEmbeddings')
    def test_initialize_failure(self, mock_langchain_embeddings):
        """埋め込みモデル初期化失敗のテスト"""
        mock_langchain_embeddings.side_effect = Exception("Connection failed")

        embeddings = OllamaEmbeddings()

        with pytest.raises(Exception) as exc_info:
            embeddings.initialize()

        assert "埋め込みモデルの初期化に失敗しました" in str(exc_info.value)

    def test_embed_documents_without_initialization(self):
        """初期化前のembed_documents呼び出しテスト"""
        embeddings = OllamaEmbeddings()

        with pytest.raises(ValueError) as exc_info:
            embeddings.embed_documents(["test"])

        assert "埋め込みモデルが初期化されていません" in str(exc_info.value)

    @patch('models.embeddings.LangChainOllamaEmbeddings')
    def test_embed_documents_success(self, mock_langchain_embeddings):
        """複数ドキュメントの埋め込み生成成功テスト"""
        mock_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        mock_instance = MagicMock()
        mock_instance.embed_documents.return_value = mock_vectors
        mock_langchain_embeddings.return_value = mock_instance

        embeddings = OllamaEmbeddings()
        embeddings.initialize()
        result = embeddings.embed_documents(["text1", "text2"])

        assert result == mock_vectors
        assert len(result) == 2
        mock_instance.embed_documents.assert_called_once_with(["text1", "text2"])

    @patch('models.embeddings.LangChainOllamaEmbeddings')
    def test_embed_documents_failure(self, mock_langchain_embeddings):
        """ドキュメント埋め込み生成失敗テスト"""
        mock_instance = MagicMock()
        mock_instance.embed_documents.side_effect = Exception("Embedding failed")
        mock_langchain_embeddings.return_value = mock_instance

        embeddings = OllamaEmbeddings()
        embeddings.initialize()

        with pytest.raises(Exception) as exc_info:
            embeddings.embed_documents(["test"])

        assert "ドキュメントの埋め込み生成に失敗しました" in str(exc_info.value)

    def test_embed_query_without_initialization(self):
        """初期化前のembed_query呼び出しテスト"""
        embeddings = OllamaEmbeddings()

        with pytest.raises(ValueError) as exc_info:
            embeddings.embed_query("test query")

        assert "埋め込みモデルが初期化されていません" in str(exc_info.value)

    @patch('models.embeddings.LangChainOllamaEmbeddings')
    def test_embed_query_success(self, mock_langchain_embeddings):
        """クエリ埋め込み生成成功テスト"""
        mock_vector = [0.1, 0.2, 0.3, 0.4, 0.5]

        mock_instance = MagicMock()
        mock_instance.embed_query.return_value = mock_vector
        mock_langchain_embeddings.return_value = mock_instance

        embeddings = OllamaEmbeddings()
        embeddings.initialize()
        result = embeddings.embed_query("test query")

        assert result == mock_vector
        assert len(result) == 5
        mock_instance.embed_query.assert_called_once_with("test query")

    @patch('models.embeddings.LangChainOllamaEmbeddings')
    def test_embed_query_failure(self, mock_langchain_embeddings):
        """クエリ埋め込み生成失敗テスト"""
        mock_instance = MagicMock()
        mock_instance.embed_query.side_effect = Exception("Embedding failed")
        mock_langchain_embeddings.return_value = mock_instance

        embeddings = OllamaEmbeddings()
        embeddings.initialize()

        with pytest.raises(Exception) as exc_info:
            embeddings.embed_query("test")

        assert "クエリの埋め込み生成に失敗しました" in str(exc_info.value)

    def test_embeddings_property_without_initialization(self):
        """初期化前のembeddingsプロパティアクセステスト"""
        embeddings = OllamaEmbeddings()

        with pytest.raises(ValueError) as exc_info:
            _ = embeddings.embeddings

        assert "埋め込みモデルが初期化されていません" in str(exc_info.value)

    @patch('models.embeddings.LangChainOllamaEmbeddings')
    def test_embeddings_property_after_initialization(self, mock_langchain_embeddings):
        """初期化後のembeddingsプロパティアクセステスト"""
        mock_instance = MagicMock()
        mock_langchain_embeddings.return_value = mock_instance

        embeddings = OllamaEmbeddings()
        embeddings.initialize()

        assert embeddings.embeddings == mock_instance


class TestCreateEmbeddings:
    """create_embeddings関数のテスト"""

    @patch('models.embeddings.OllamaEmbeddings')
    def test_create_embeddings_default(self, mock_ollama_embeddings):
        """デフォルト値でのcreate_embeddingsテスト"""
        mock_instance = MagicMock()
        mock_initialized = MagicMock()
        mock_instance.initialize.return_value = mock_initialized
        mock_ollama_embeddings.return_value = mock_instance

        result = create_embeddings()

        assert result == mock_initialized
        mock_ollama_embeddings.assert_called_once_with(model=None)
        mock_instance.initialize.assert_called_once()

    @patch('models.embeddings.OllamaEmbeddings')
    def test_create_embeddings_custom(self, mock_ollama_embeddings):
        """カスタムモデルでのcreate_embeddingsテスト"""
        mock_instance = MagicMock()
        mock_initialized = MagicMock()
        mock_instance.initialize.return_value = mock_initialized
        mock_ollama_embeddings.return_value = mock_instance

        result = create_embeddings(model="custom-model")

        assert result == mock_initialized
        mock_ollama_embeddings.assert_called_once_with(model="custom-model")
