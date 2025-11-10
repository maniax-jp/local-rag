"""
ドキュメントローダーモジュール
各種形式のドキュメントを読み込む
"""

import os
from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    JSONLoader
)


class DocumentLoaderManager:
    """ドキュメントローダーを管理するクラス"""

    # サポートする形式
    SUPPORTED_EXTENSIONS = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".csv": CSVLoader,
        ".json": JSONLoader
    }

    def __init__(self):
        """初期化"""
        pass

    def load_document(self, file_path: str) -> List[Document]:
        """
        単一ファイルを読み込む

        Args:
            file_path: ファイルパス

        Returns:
            Documentオブジェクトのリスト

        Raises:
            FileNotFoundError: ファイルが存在しない場合
            ValueError: サポートされていない形式の場合
            Exception: 読み込みに失敗した場合
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

        extension = path.suffix.lower()
        loader_class = self._get_loader(extension)

        if loader_class is None:
            raise ValueError(
                f"サポートされていないファイル形式です: {extension}\n"
                f"サポート形式: {', '.join(self.SUPPORTED_EXTENSIONS.keys())}"
            )

        try:
            # ローダーの初期化と実行
            if extension == ".txt":
                # TextLoaderはUTF-8エンコーディングを明示的に指定
                loader = loader_class(file_path, encoding="utf-8")
            elif extension == ".json":
                # JSONLoaderはjq_schemaが必要
                loader = loader_class(
                    file_path=file_path,
                    jq_schema=".",
                    text_content=False
                )
            else:
                loader = loader_class(file_path)

            documents = loader.load()

            # メタデータにファイル情報を追加
            for doc in documents:
                doc.metadata["file_name"] = path.name
                doc.metadata["file_extension"] = extension
                doc.metadata["file_path"] = str(path.absolute())

            print(f"読み込み完了: {path.name} ({len(documents)}件)")
            return documents

        except Exception as e:
            raise Exception(f"ファイルの読み込みに失敗しました ({path.name}): {str(e)}")

    def load_directory(self, dir_path: str, recursive: bool = True) -> List[Document]:
        """
        ディレクトリ内のファイルを一括読み込み

        Args:
            dir_path: ディレクトリパス
            recursive: サブディレクトリも再帰的に読み込むか

        Returns:
            全Documentオブジェクトのリスト

        Raises:
            NotADirectoryError: ディレクトリが存在しない場合
        """
        dir_path_obj = Path(dir_path)

        if not dir_path_obj.exists() or not dir_path_obj.is_dir():
            raise NotADirectoryError(f"ディレクトリが見つかりません: {dir_path}")

        all_documents = []
        failed_files = []

        # ファイルを検索
        if recursive:
            files = [f for f in dir_path_obj.rglob("*") if f.is_file()]
        else:
            files = [f for f in dir_path_obj.iterdir() if f.is_file()]

        # サポート対象ファイルのみフィルタ
        supported_files = [
            f for f in files
            if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        print(f"\n{len(supported_files)}個のファイルを処理します...\n")

        for file_path in supported_files:
            try:
                documents = self.load_document(str(file_path))
                all_documents.extend(documents)
            except Exception as e:
                print(f"エラー: {file_path.name} - {str(e)}")
                failed_files.append(str(file_path))

        print(f"\n処理完了:")
        print(f"  成功: {len(supported_files) - len(failed_files)}ファイル")
        print(f"  失敗: {len(failed_files)}ファイル")
        print(f"  合計ドキュメント数: {len(all_documents)}")

        if failed_files:
            print(f"\n失敗したファイル:")
            for f in failed_files:
                print(f"  - {f}")

        return all_documents

    def _get_loader(self, extension: str):
        """
        ファイル拡張子に対応するローダークラスを取得

        Args:
            extension: ファイル拡張子（例: ".pdf"）

        Returns:
            ローダークラス、またはNone
        """
        return self.SUPPORTED_EXTENSIONS.get(extension.lower())

    @classmethod
    def list_supported_formats(cls) -> List[str]:
        """
        サポートされているファイル形式のリストを返す

        Returns:
            拡張子のリスト
        """
        return list(cls.SUPPORTED_EXTENSIONS.keys())
