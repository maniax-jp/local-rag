"""
ドキュメントローダーモジュールのテスト
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from app.loaders.document_loader import DocumentLoaderManager


class TestDocumentLoaderManager:
    """DocumentLoaderManagerクラスのテスト"""

    def test_supported_extensions(self):
        """サポート対象拡張子のテスト"""
        loader = DocumentLoaderManager()

        expected_extensions = {".pdf", ".txt", ".md", ".csv", ".json"}
        assert set(loader.SUPPORTED_EXTENSIONS.keys()) == expected_extensions

    def test_list_supported_formats(self):
        """サポート形式一覧取得のテスト"""
        formats = DocumentLoaderManager.list_supported_formats()

        assert ".pdf" in formats
        assert ".txt" in formats
        assert ".md" in formats
        assert ".csv" in formats
        assert ".json" in formats

    def test_get_loader_valid_extension(self):
        """有効な拡張子でのローダー取得テスト"""
        loader = DocumentLoaderManager()

        txt_loader = loader._get_loader(".txt")
        pdf_loader = loader._get_loader(".pdf")

        assert txt_loader is not None
        assert pdf_loader is not None

    def test_get_loader_invalid_extension(self):
        """無効な拡張子でのローダー取得テスト"""
        loader = DocumentLoaderManager()

        result = loader._get_loader(".docx")

        assert result is None

    def test_load_document_file_not_found(self):
        """存在しないファイルの読み込みテスト"""
        loader = DocumentLoaderManager()

        with pytest.raises(FileNotFoundError):
            loader.load_document("/nonexistent/file.txt")

    def test_load_document_unsupported_format(self, tmp_path):
        """サポート外形式の読み込みテスト"""
        # 一時ファイル作成
        test_file = tmp_path / "test.docx"
        test_file.write_text("test content")

        loader = DocumentLoaderManager()

        with pytest.raises(ValueError) as exc_info:
            loader.load_document(str(test_file))

        assert "サポートされていないファイル形式" in str(exc_info.value)

    def test_load_document_txt_success(self, sample_txt_path):
        """テキストファイル読み込み成功テスト"""
        loader = DocumentLoaderManager()
        documents = loader.load_document(sample_txt_path)

        assert len(documents) > 0
        assert documents[0].page_content is not None
        assert documents[0].metadata["file_name"] == "sample.txt"
        assert documents[0].metadata["file_extension"] == ".txt"

    def test_load_document_md_success(self, sample_md_path):
        """Markdownファイル読み込み成功テスト"""
        loader = DocumentLoaderManager()
        documents = loader.load_document(sample_md_path)

        assert len(documents) > 0
        assert documents[0].metadata["file_name"] == "sample.md"
        assert documents[0].metadata["file_extension"] == ".md"

    def test_load_document_csv_success(self, sample_csv_path):
        """CSVファイル読み込み成功テスト"""
        loader = DocumentLoaderManager()
        documents = loader.load_document(sample_csv_path)

        assert len(documents) > 0
        assert documents[0].metadata["file_name"] == "sample.csv"
        assert documents[0].metadata["file_extension"] == ".csv"

    def test_load_document_json_success(self, sample_json_path):
        """JSONファイル読み込み成功テスト"""
        loader = DocumentLoaderManager()
        documents = loader.load_document(sample_json_path)

        assert len(documents) > 0
        assert documents[0].metadata["file_name"] == "sample.json"
        assert documents[0].metadata["file_extension"] == ".json"

    def test_load_directory_not_found(self):
        """存在しないディレクトリの読み込みテスト"""
        loader = DocumentLoaderManager()

        with pytest.raises(NotADirectoryError):
            loader.load_directory("/nonexistent/directory")

    def test_load_directory_success(self, fixtures_dir):
        """ディレクトリ一括読み込み成功テスト"""
        loader = DocumentLoaderManager()
        documents = loader.load_directory(str(fixtures_dir), recursive=False)

        # 少なくとも1つのドキュメントが読み込まれることを確認
        assert len(documents) > 0

        # 各ドキュメントにメタデータが付与されていることを確認
        for doc in documents:
            assert "file_name" in doc.metadata
            assert "file_extension" in doc.metadata
            assert "file_path" in doc.metadata

    def test_load_directory_empty(self, tmp_path):
        """空ディレクトリの読み込みテスト"""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        loader = DocumentLoaderManager()
        documents = loader.load_directory(str(empty_dir))

        assert len(documents) == 0

    def test_load_directory_with_unsupported_files(self, tmp_path):
        """サポート外ファイルを含むディレクトリの読み込みテスト"""
        # サポート対象ファイル
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("test content", encoding="utf-8")

        # サポート外ファイル
        docx_file = tmp_path / "test.docx"
        docx_file.write_text("test content")

        loader = DocumentLoaderManager()
        documents = loader.load_directory(str(tmp_path))

        # サポート対象のみ読み込まれる
        assert len(documents) >= 1
        assert all(doc.metadata["file_extension"] in [".txt"] for doc in documents)

    @patch('app.loaders.document_loader.TextLoader')
    def test_load_document_with_loader_error(self, mock_text_loader, sample_txt_path):
        """ローダーエラー時のテスト"""
        mock_text_loader.side_effect = Exception("Loader error")

        loader = DocumentLoaderManager()

        with pytest.raises(Exception) as exc_info:
            loader.load_document(sample_txt_path)

        assert "ファイルの読み込みに失敗しました" in str(exc_info.value)
