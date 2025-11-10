"""
埋め込みモデルモジュール
Ollama経由でnomic-embed-textを使用
"""

from typing import List, Optional
from langchain_ollama import OllamaEmbeddings as LangChainOllamaEmbeddings
from app.config import config


class OllamaEmbeddings:
    """Ollama埋め込みモデルのラッパークラス"""

    def __init__(self, model: Optional[str] = None):
        """
        初期化

        Args:
            model: 埋め込みモデル名（Noneの場合は設定から取得）
        """
        self.model = model or config.ollama.embed_model
        self.base_url = config.ollama.base_url
        self._embeddings: Optional[LangChainOllamaEmbeddings] = None

    def initialize(self) -> LangChainOllamaEmbeddings:
        """
        埋め込みモデルを初期化して返す

        Returns:
            OllamaEmbeddingsインスタンス

        Raises:
            Exception: 初期化に失敗した場合
        """
        try:
            self._embeddings = LangChainOllamaEmbeddings(
                model=self.model,
                base_url=self.base_url
            )
            return self._embeddings
        except Exception as e:
            raise Exception(f"埋め込みモデルの初期化に失敗しました: {str(e)}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        複数のテキストを埋め込みベクトルに変換

        Args:
            texts: テキストのリスト

        Returns:
            埋め込みベクトルのリスト

        Raises:
            ValueError: 埋め込みモデルが初期化されていない場合
            Exception: 埋め込み生成に失敗した場合
        """
        if self._embeddings is None:
            raise ValueError("埋め込みモデルが初期化されていません。initialize()を先に呼び出してください。")

        try:
            return self._embeddings.embed_documents(texts)
        except Exception as e:
            raise Exception(f"ドキュメントの埋め込み生成に失敗しました: {str(e)}")

    def embed_query(self, text: str) -> List[float]:
        """
        単一のクエリテキストを埋め込みベクトルに変換

        Args:
            text: クエリテキスト

        Returns:
            埋め込みベクトル

        Raises:
            ValueError: 埋め込みモデルが初期化されていない場合
            Exception: 埋め込み生成に失敗した場合
        """
        if self._embeddings is None:
            raise ValueError("埋め込みモデルが初期化されていません。initialize()を先に呼び出してください。")

        try:
            return self._embeddings.embed_query(text)
        except Exception as e:
            raise Exception(f"クエリの埋め込み生成に失敗しました: {str(e)}")

    @property
    def embeddings(self) -> LangChainOllamaEmbeddings:
        """初期化済みの埋め込みモデルインスタンスを取得"""
        if self._embeddings is None:
            raise ValueError("埋め込みモデルが初期化されていません。initialize()を先に呼び出してください。")
        return self._embeddings


def create_embeddings(model: Optional[str] = None) -> LangChainOllamaEmbeddings:
    """
    埋め込みモデルインスタンスを作成して返すヘルパー関数

    Args:
        model: 埋め込みモデル名

    Returns:
        初期化済みのOllamaEmbeddingsインスタンス
    """
    embeddings = OllamaEmbeddings(model=model)
    return embeddings.initialize()
