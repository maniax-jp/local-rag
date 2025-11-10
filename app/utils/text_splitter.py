"""
テキストスプリッターモジュール
日本語に最適化されたテキスト分割
"""

from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import config


class JapaneseTextSplitter:
    """日本語テキスト用のスプリッタークラス"""

    # 日本語に最適化されたセパレータ
    JAPANESE_SEPARATORS = [
        "\n\n",  # 段落
        "\n",    # 改行
        "。",    # 句点
        "、",    # 読点
        " ",     # 空白
        ""       # 文字単位
    ]

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None
    ):
        """
        初期化

        Args:
            chunk_size: チャンクサイズ（Noneの場合は設定から取得）
            chunk_overlap: チャンクオーバーラップ（Noneの場合は設定から取得）
            separators: セパレータリスト（Noneの場合はデフォルトを使用）
        """
        self.chunk_size = chunk_size if chunk_size is not None else config.rag.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else config.rag.chunk_overlap
        self.separators = separators or self.JAPANESE_SEPARATORS

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,  # 文字数ベース
            is_separator_regex=False
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        ドキュメントリストを分割

        Args:
            documents: Documentオブジェクトのリスト

        Returns:
            分割されたDocumentオブジェクトのリスト
        """
        if not documents:
            return []

        try:
            split_docs = self._splitter.split_documents(documents)
            print(f"ドキュメント分割完了: {len(documents)}件 → {len(split_docs)}チャンク")
            return split_docs
        except Exception as e:
            raise Exception(f"ドキュメントの分割に失敗しました: {str(e)}")

    def split_text(self, text: str) -> List[str]:
        """
        単一テキストを分割

        Args:
            text: 分割するテキスト

        Returns:
            分割されたテキストのリスト
        """
        if not text or text.strip() == "":
            return []

        try:
            chunks = self._splitter.split_text(text)
            return chunks
        except Exception as e:
            raise Exception(f"テキストの分割に失敗しました: {str(e)}")

    def create_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """
        テキストリストからDocumentオブジェクトを作成

        Args:
            texts: テキストのリスト
            metadatas: メタデータのリスト（オプション）

        Returns:
            Documentオブジェクトのリスト
        """
        try:
            documents = self._splitter.create_documents(
                texts=texts,
                metadatas=metadatas
            )
            return documents
        except Exception as e:
            raise Exception(f"ドキュメントの作成に失敗しました: {str(e)}")

    @property
    def splitter(self) -> RecursiveCharacterTextSplitter:
        """内部のスプリッターインスタンスを取得"""
        return self._splitter


def create_text_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> JapaneseTextSplitter:
    """
    テキストスプリッターを作成するヘルパー関数

    Args:
        chunk_size: チャンクサイズ
        chunk_overlap: チャンクオーバーラップ

    Returns:
        JapaneseTextSplitterインスタンス
    """
    return JapaneseTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
