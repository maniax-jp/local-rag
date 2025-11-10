"""
LLM初期化モジュール
Ollama経由でLlama-3-Swallowを使用
"""

from typing import Optional
from langchain_ollama import ChatOllama
from config import config


class SwallowLLM:
    """Llama-3-Swallow LLMのラッパークラス"""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        初期化

        Args:
            model: モデル名（Noneの場合は設定から取得）
            temperature: 温度パラメータ（Noneの場合は設定から取得）
            max_tokens: 最大トークン数（Noneの場合は設定から取得）
        """
        self.model = model or config.ollama.llm_model
        self.temperature = temperature if temperature is not None else config.rag.temperature
        self.max_tokens = max_tokens or config.rag.max_tokens
        self.base_url = config.ollama.base_url
        self._llm: Optional[ChatOllama] = None

    def initialize(self) -> ChatOllama:
        """
        LLMを初期化して返す

        Returns:
            ChatOllamaインスタンス

        Raises:
            Exception: 初期化に失敗した場合
        """
        try:
            self._llm = ChatOllama(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                num_predict=self.max_tokens
            )
            return self._llm
        except Exception as e:
            raise Exception(f"LLMの初期化に失敗しました: {str(e)}")

    def generate(self, prompt: str) -> str:
        """
        プロンプトから回答を生成

        Args:
            prompt: 入力プロンプト

        Returns:
            生成された回答テキスト

        Raises:
            ValueError: LLMが初期化されていない場合
            Exception: 生成に失敗した場合
        """
        if self._llm is None:
            raise ValueError("LLMが初期化されていません。initialize()を先に呼び出してください。")

        try:
            # プロンプトをHumanMessageに変換
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=prompt)]
            response = self._llm.invoke(messages)
            return response.content
        except Exception as e:
            raise Exception(f"回答生成に失敗しました: {str(e)}")

    @property
    def llm(self) -> ChatOllama:
        """初期化済みのLLMインスタンスを取得"""
        if self._llm is None:
            raise ValueError("LLMが初期化されていません。initialize()を先に呼び出してください。")
        return self._llm


def create_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> ChatOllama:
    """
    LLMインスタンスを作成して返すヘルパー関数

    Args:
        model: モデル名
        temperature: 温度パラメータ
        max_tokens: 最大トークン数

    Returns:
        初期化済みのChatOllamaインスタンス
    """
    swallow = SwallowLLM(model=model, temperature=temperature, max_tokens=max_tokens)
    return swallow.initialize()
