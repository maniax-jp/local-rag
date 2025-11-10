"""
設定管理モジュール
環境変数からシステム設定を読み込み、管理する
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()


@dataclass
class OllamaConfig:
    """Ollama関連の設定"""
    host: str
    port: int
    llm_model: str
    embed_model: str

    @property
    def base_url(self) -> str:
        """Ollamaのベースurl"""
        return f"http://{self.host}:{self.port}"


@dataclass
class QdrantConfig:
    """Qdrant関連の設定"""
    host: str
    port: int
    collection_name: str

    @property
    def url(self) -> str:
        """QdrantのURL"""
        return f"http://{self.host}:{self.port}"


@dataclass
class RAGConfig:
    """RAG処理関連の設定"""
    chunk_size: int
    chunk_overlap: int
    top_k: int
    temperature: float
    max_tokens: int


@dataclass
class DocumentConfig:
    """ドキュメント関連の設定"""
    documents_path: str


class Config:
    """アプリケーション全体の設定を管理するクラス"""

    def __init__(self):
        self.ollama = self._load_ollama_config()
        self.qdrant = self._load_qdrant_config()
        self.rag = self._load_rag_config()
        self.document = self._load_document_config()

    def _load_ollama_config(self) -> OllamaConfig:
        """Ollama設定の読み込み"""
        return OllamaConfig(
            host=os.getenv("OLLAMA_HOST", "ollama"),
            port=int(os.getenv("OLLAMA_PORT", "11434")),
            llm_model=os.getenv("OLLAMA_LLM_MODEL", "mmnga/llama-3-swallow-8b-instruct-v0.1:q4_k_m"),
            embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        )

    def _load_qdrant_config(self) -> QdrantConfig:
        """Qdrant設定の読み込み"""
        return QdrantConfig(
            host=os.getenv("QDRANT_HOST", "qdrant"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            collection_name=os.getenv("QDRANT_COLLECTION_NAME", "documents")
        )

    def _load_rag_config(self) -> RAGConfig:
        """RAG設定の読み込み"""
        return RAGConfig(
            chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            top_k=int(os.getenv("TOP_K", "4")),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS", "2000"))
        )

    def _load_document_config(self) -> DocumentConfig:
        """ドキュメント設定の読み込み"""
        return DocumentConfig(
            documents_path=os.getenv("DOCUMENTS_PATH", "/documents")
        )

    def validate(self) -> bool:
        """設定値のバリデーション"""
        # 基本的な値の検証
        assert self.ollama.port > 0, "OLLAMA_PORTは正の整数である必要があります"
        assert self.qdrant.port > 0, "QDRANT_PORTは正の整数である必要があります"
        assert self.rag.chunk_size > 0, "CHUNK_SIZEは正の整数である必要があります"
        assert self.rag.chunk_overlap >= 0, "CHUNK_OVERLAPは0以上の整数である必要があります"
        assert self.rag.top_k > 0, "TOP_Kは正の整数である必要があります"
        assert 0.0 <= self.rag.temperature <= 2.0, "TEMPERATUREは0.0～2.0の範囲である必要があります"
        assert self.rag.max_tokens > 0, "MAX_TOKENSは正の整数である必要があります"

        return True

    def __str__(self) -> str:
        """設定内容を文字列として返す"""
        return f"""
Config:
  Ollama:
    - Base URL: {self.ollama.base_url}
    - LLM Model: {self.ollama.llm_model}
    - Embed Model: {self.ollama.embed_model}

  Qdrant:
    - URL: {self.qdrant.url}
    - Collection: {self.qdrant.collection_name}

  RAG:
    - Chunk Size: {self.rag.chunk_size}
    - Chunk Overlap: {self.rag.chunk_overlap}
    - Top K: {self.rag.top_k}
    - Temperature: {self.rag.temperature}
    - Max Tokens: {self.rag.max_tokens}

  Document:
    - Path: {self.document.documents_path}
"""


# グローバル設定インスタンス
config = Config()
