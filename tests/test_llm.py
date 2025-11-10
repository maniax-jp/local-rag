"""
LLM初期化モジュールのテスト
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from app.models.llm import SwallowLLM, create_llm


class TestSwallowLLM:
    """SwallowLLMクラスのテスト"""

    def test_init_with_defaults(self, mock_config):
        """デフォルト値での初期化テスト"""
        llm = SwallowLLM()

        assert llm.model == mock_config.ollama.llm_model
        assert llm.temperature == mock_config.rag.temperature
        assert llm.max_tokens == mock_config.rag.max_tokens
        assert llm.base_url == mock_config.ollama.base_url

    def test_init_with_custom_values(self):
        """カスタム値での初期化テスト"""
        llm = SwallowLLM(
            model="custom-model",
            temperature=0.5,
            max_tokens=1000
        )

        assert llm.model == "custom-model"
        assert llm.temperature == 0.5
        assert llm.max_tokens == 1000

    @patch('app.models.llm.ChatOllama')
    def test_initialize_success(self, mock_chat_ollama):
        """LLM初期化成功のテスト"""
        mock_instance = MagicMock()
        mock_chat_ollama.return_value = mock_instance

        llm = SwallowLLM()
        result = llm.initialize()

        assert result == mock_instance
        assert llm._llm == mock_instance
        mock_chat_ollama.assert_called_once()

    @patch('app.models.llm.ChatOllama')
    def test_initialize_failure(self, mock_chat_ollama):
        """LLM初期化失敗のテスト"""
        mock_chat_ollama.side_effect = Exception("Connection failed")

        llm = SwallowLLM()

        with pytest.raises(Exception) as exc_info:
            llm.initialize()

        assert "LLMの初期化に失敗しました" in str(exc_info.value)

    def test_generate_without_initialization(self):
        """初期化前のgenerate呼び出しテスト"""
        llm = SwallowLLM()

        with pytest.raises(ValueError) as exc_info:
            llm.generate("test prompt")

        assert "LLMが初期化されていません" in str(exc_info.value)

    @patch('app.models.llm.ChatOllama')
    def test_generate_success(self, mock_chat_ollama):
        """回答生成成功のテスト"""
        mock_response = MagicMock()
        mock_response.content = "これはテスト回答です"

        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_response
        mock_chat_ollama.return_value = mock_instance

        llm = SwallowLLM()
        llm.initialize()
        result = llm.generate("テストプロンプト")

        assert result == "これはテスト回答です"
        mock_instance.invoke.assert_called_once_with("テストプロンプト")

    @patch('app.models.llm.ChatOllama')
    def test_generate_failure(self, mock_chat_ollama):
        """回答生成失敗のテスト"""
        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = Exception("Generation failed")
        mock_chat_ollama.return_value = mock_instance

        llm = SwallowLLM()
        llm.initialize()

        with pytest.raises(Exception) as exc_info:
            llm.generate("test")

        assert "回答生成に失敗しました" in str(exc_info.value)

    def test_llm_property_without_initialization(self):
        """初期化前のllmプロパティアクセステスト"""
        llm = SwallowLLM()

        with pytest.raises(ValueError) as exc_info:
            _ = llm.llm

        assert "LLMが初期化されていません" in str(exc_info.value)

    @patch('app.models.llm.ChatOllama')
    def test_llm_property_after_initialization(self, mock_chat_ollama):
        """初期化後のllmプロパティアクセステスト"""
        mock_instance = MagicMock()
        mock_chat_ollama.return_value = mock_instance

        llm = SwallowLLM()
        llm.initialize()

        assert llm.llm == mock_instance


class TestCreateLLM:
    """create_llm関数のテスト"""

    @patch('app.models.llm.SwallowLLM')
    def test_create_llm_default(self, mock_swallow_llm):
        """デフォルト値でのcreate_llmテスト"""
        mock_instance = MagicMock()
        mock_initialized = MagicMock()
        mock_instance.initialize.return_value = mock_initialized
        mock_swallow_llm.return_value = mock_instance

        result = create_llm()

        assert result == mock_initialized
        mock_swallow_llm.assert_called_once_with(
            model=None,
            temperature=None,
            max_tokens=None
        )
        mock_instance.initialize.assert_called_once()

    @patch('app.models.llm.SwallowLLM')
    def test_create_llm_custom(self, mock_swallow_llm):
        """カスタム値でのcreate_llmテスト"""
        mock_instance = MagicMock()
        mock_initialized = MagicMock()
        mock_instance.initialize.return_value = mock_initialized
        mock_swallow_llm.return_value = mock_instance

        result = create_llm(
            model="custom",
            temperature=0.3,
            max_tokens=500
        )

        assert result == mock_initialized
        mock_swallow_llm.assert_called_once_with(
            model="custom",
            temperature=0.3,
            max_tokens=500
        )
