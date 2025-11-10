"""
Qdrantクライアントモジュール
ベクターデータベースとの連携を担当
"""

from typing import List, Optional
from langchain_qdrant import QdrantVectorStore as LangChainQdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient as QdrantClientBase
from qdrant_client.models import Distance, VectorParams
from config import config


class QdrantVectorStoreManager:
    """Qdrantベクターストアのラッパークラス"""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        embeddings=None
    ):
        """
        初期化

        Args:
            collection_name: コレクション名（Noneの場合は設定から取得）
            embeddings: 埋め込みモデルインスタンス
        """
        self.collection_name = collection_name or config.qdrant.collection_name
        self.url = config.qdrant.url
        self.embeddings = embeddings
        self._client: Optional[QdrantClientBase] = None
        self._vector_store: Optional[LangChainQdrantVectorStore] = None
        self.vector_size = 768  # nomic-embed-textの次元数

    def initialize(self) -> QdrantClientBase:
        """
        Qdrantクライアントを初期化

        Returns:
            QdrantClientインスタンス

        Raises:
            Exception: 初期化に失敗した場合
        """
        try:
            self._client = QdrantClientBase(
                url=self.url,
                timeout=60
            )
            return self._client
        except Exception as e:
            raise Exception(f"Qdrantクライアントの初期化に失敗しました: {str(e)}")

    def create_collection(self, force: bool = False) -> bool:
        """
        コレクションを作成

        Args:
            force: Trueの場合、既存コレクションを削除して再作成

        Returns:
            作成成功の場合True

        Raises:
            ValueError: クライアントが初期化されていない場合
            Exception: コレクション作成に失敗した場合
        """
        if self._client is None:
            raise ValueError("Qdrantクライアントが初期化されていません。initialize()を先に呼び出してください。")

        try:
            # 既存コレクションの確認
            collections = self._client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)

            if collection_exists:
                if force:
                    print(f"既存のコレクション '{self.collection_name}' を削除します...")
                    self._client.delete_collection(self.collection_name)
                else:
                    print(f"コレクション '{self.collection_name}' は既に存在します。")
                    return True

            # コレクションを作成
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"コレクション '{self.collection_name}' を作成しました。")
            return True

        except Exception as e:
            raise Exception(f"コレクションの作成に失敗しました: {str(e)}")

    def get_vector_store(self) -> LangChainQdrantVectorStore:
        """
        LangChain用のQdrantVectorStoreインスタンスを取得

        Returns:
            QdrantVectorStoreインスタンス

        Raises:
            ValueError: クライアントまたは埋め込みモデルが初期化されていない場合
        """
        if self._client is None:
            raise ValueError("Qdrantクライアントが初期化されていません。")

        if self.embeddings is None:
            raise ValueError("埋め込みモデルが設定されていません。")

        if self._vector_store is None:
            self._vector_store = LangChainQdrantVectorStore(
                client=self._client,
                collection_name=self.collection_name,
                embedding=self.embeddings
            )

        return self._vector_store

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        ドキュメントをベクターストアに追加

        Args:
            documents: 追加するドキュメントのリスト

        Returns:
            追加されたドキュメントのIDリスト

        Raises:
            Exception: ドキュメント追加に失敗した場合
        """
        try:
            vector_store = self.get_vector_store()
            ids = vector_store.add_documents(documents)
            print(f"{len(documents)}件のドキュメントを追加しました。")
            return ids
        except Exception as e:
            raise Exception(f"ドキュメントの追加に失敗しました: {str(e)}")

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Document]:
        """
        類似度検索を実行

        Args:
            query: 検索クエリ
            k: 取得する件数（Noneの場合は設定から取得）

        Returns:
            類似ドキュメントのリスト

        Raises:
            ValueError: クエリが空の場合
            Exception: 検索に失敗した場合
        """
        if not query or query.strip() == "":
            raise ValueError("検索クエリが空です")

        k = k or config.rag.top_k

        try:
            vector_store = self.get_vector_store()
            results = vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            raise Exception(f"類似度検索に失敗しました: {str(e)}")

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[tuple[Document, float]]:
        """
        スコア付きで類似度検索を実行

        Args:
            query: 検索クエリ
            k: 取得する件数（Noneの場合は設定から取得）

        Returns:
            (ドキュメント, スコア)のタプルのリスト

        Raises:
            ValueError: クエリが空の場合
            Exception: 検索に失敗した場合
        """
        if not query or query.strip() == "":
            raise ValueError("検索クエリが空です")

        k = k or config.rag.top_k

        try:
            vector_store = self.get_vector_store()
            results = vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            raise Exception(f"類似度検索に失敗しました: {str(e)}")

    def delete_collection(self) -> bool:
        """
        コレクションを削除

        Returns:
            削除成功の場合True

        Raises:
            ValueError: クライアントが初期化されていない場合
            Exception: 削除に失敗した場合
        """
        if self._client is None:
            raise ValueError("Qdrantクライアントが初期化されていません。")

        try:
            self._client.delete_collection(self.collection_name)
            print(f"コレクション '{self.collection_name}' を削除しました。")
            return True
        except Exception as e:
            raise Exception(f"コレクションの削除に失敗しました: {str(e)}")

    def get_collection_info(self) -> dict:
        """
        コレクション情報を取得

        Returns:
            コレクション情報の辞書

        Raises:
            ValueError: クライアントが初期化されていない場合
        """
        if self._client is None:
            raise ValueError("Qdrantクライアントが初期化されていません。")

        try:
            info = self._client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            print(f"コレクション情報の取得に失敗しました: {str(e)}")
            return {}

    @property
    def client(self) -> QdrantClientBase:
        """初期化済みのQdrantClientインスタンスを取得"""
        if self._client is None:
            raise ValueError("Qdrantクライアントが初期化されていません。")
        return self._client
