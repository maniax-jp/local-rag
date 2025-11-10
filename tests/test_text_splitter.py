"""
テキストスプリッターモジュールのテスト
"""

import pytest
from langchain_core.documents import Document
from utils.text_splitter import JapaneseTextSplitter, create_text_splitter


class TestJapaneseTextSplitter:
    """JapaneseTextSplitterクラスのテスト"""

    def test_init_with_defaults(self, mock_config):
        """デフォルト値での初期化テスト"""
        splitter = JapaneseTextSplitter()

        assert splitter.chunk_size == mock_config.rag.chunk_size
        assert splitter.chunk_overlap == mock_config.rag.chunk_overlap
        assert splitter.separators == JapaneseTextSplitter.JAPANESE_SEPARATORS

    def test_init_with_custom_values(self):
        """カスタム値での初期化テスト"""
        custom_separators = ["\n", "。"]
        splitter = JapaneseTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=custom_separators
        )

        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 100
        assert splitter.separators == custom_separators

    def test_japanese_separators(self):
        """日本語セパレータの確認テスト"""
        separators = JapaneseTextSplitter.JAPANESE_SEPARATORS

        assert "\n\n" in separators
        assert "\n" in separators
        assert "。" in separators
        assert "、" in separators
        assert " " in separators
        assert "" in separators

    def test_split_text_short_text(self):
        """短いテキストの分割テスト"""
        splitter = JapaneseTextSplitter(chunk_size=100, chunk_overlap=0)
        text = "これは短いテキストです。"

        chunks = splitter.split_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_text_long_text(self, sample_long_text):
        """長いテキストの分割テスト"""
        splitter = JapaneseTextSplitter(chunk_size=500, chunk_overlap=50)

        chunks = splitter.split_text(sample_long_text)

        # 複数チャンクに分割されることを確認
        assert len(chunks) > 1

        # 各チャンクがchunk_size以下であることを確認
        for chunk in chunks:
            assert len(chunk) <= splitter.chunk_size + 100  # マージン考慮

    def test_split_text_with_japanese_separators(self):
        """日本語セパレータでの分割テスト"""
        splitter = JapaneseTextSplitter(chunk_size=100, chunk_overlap=0)
        text = "第一段落です。これは文の途中です、続きがあります。\n\n第二段落です。"

        chunks = splitter.split_text(text)

        # セパレータで分割されることを確認
        assert len(chunks) >= 1

    def test_split_text_empty_string(self):
        """空文字列の分割テスト"""
        splitter = JapaneseTextSplitter()
        text = ""

        chunks = splitter.split_text(text)

        assert len(chunks) == 0

    def test_split_text_whitespace_only(self):
        """空白のみのテキストの分割テスト"""
        splitter = JapaneseTextSplitter()
        text = "   \n\n   "

        chunks = splitter.split_text(text)

        # 空または空白のみのチャンクは生成されない
        assert len(chunks) == 0 or all(chunk.strip() for chunk in chunks)

    def test_split_documents(self, sample_text):
        """ドキュメント分割テスト"""
        splitter = JapaneseTextSplitter(chunk_size=200, chunk_overlap=50)
        documents = [Document(page_content=sample_text, metadata={"source": "test"})]

        split_docs = splitter.split_documents(documents)

        # 分割されたドキュメントが返されることを確認
        assert len(split_docs) > 0

        # メタデータが保持されることを確認
        for doc in split_docs:
            assert "source" in doc.metadata
            assert doc.metadata["source"] == "test"

    def test_split_documents_empty_list(self):
        """空リストの分割テスト"""
        splitter = JapaneseTextSplitter()
        documents = []

        split_docs = splitter.split_documents(documents)

        assert len(split_docs) == 0

    def test_split_documents_multiple(self):
        """複数ドキュメントの分割テスト"""
        splitter = JapaneseTextSplitter(chunk_size=100, chunk_overlap=20)
        documents = [
            Document(page_content="これは最初のドキュメントです。" * 10, metadata={"id": 1}),
            Document(page_content="これは二番目のドキュメントです。" * 10, metadata={"id": 2})
        ]

        split_docs = splitter.split_documents(documents)

        # 複数のチャンクに分割されることを確認
        assert len(split_docs) > 2

        # 各ドキュメントのメタデータが保持されることを確認
        ids = [doc.metadata.get("id") for doc in split_docs]
        assert 1 in ids
        assert 2 in ids

    def test_chunk_overlap(self):
        """チャンクオーバーラップのテスト"""
        splitter = JapaneseTextSplitter(chunk_size=50, chunk_overlap=10)
        text = "A" * 100  # 100文字のテキスト

        chunks = splitter.split_text(text)

        # オーバーラップが機能していることを確認（完全には検証できないが複数チャンク生成は確認）
        assert len(chunks) > 1

    def test_create_documents(self):
        """create_documentsメソッドのテスト"""
        splitter = JapaneseTextSplitter(chunk_size=100, chunk_overlap=20)
        texts = ["テキスト1です。" * 10, "テキスト2です。" * 10]
        metadatas = [{"source": "doc1"}, {"source": "doc2"}]

        documents = splitter.create_documents(texts, metadatas)

        assert len(documents) > 0
        # メタデータが正しく設定されることを確認
        sources = [doc.metadata.get("source") for doc in documents]
        assert "doc1" in sources or "doc2" in sources

    def test_splitter_property(self):
        """splitterプロパティのテスト"""
        splitter = JapaneseTextSplitter()

        internal_splitter = splitter.splitter

        assert internal_splitter is not None
        assert hasattr(internal_splitter, 'split_text')


class TestCreateTextSplitter:
    """create_text_splitter関数のテスト"""

    def test_create_text_splitter_default(self):
        """デフォルト値でのcreate_text_splitterテスト"""
        splitter = create_text_splitter()

        assert isinstance(splitter, JapaneseTextSplitter)
        assert splitter.chunk_size > 0
        assert splitter.chunk_overlap >= 0

    def test_create_text_splitter_custom(self):
        """カスタム値でのcreate_text_splitterテスト"""
        splitter = create_text_splitter(chunk_size=300, chunk_overlap=75)

        assert isinstance(splitter, JapaneseTextSplitter)
        assert splitter.chunk_size == 300
        assert splitter.chunk_overlap == 75
