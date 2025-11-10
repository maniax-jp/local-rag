"""
pytest設定とフィクスチャ
"""

import os
import sys
from pathlib import Path
import pytest

# アプリケーションのルートディレクトリをパスに追加
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture
def fixtures_dir():
    """テストフィクスチャディレクトリのパスを返す"""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_txt_path(fixtures_dir):
    """sample.txtのパスを返す"""
    return str(fixtures_dir / "sample.txt")


@pytest.fixture
def sample_md_path(fixtures_dir):
    """sample.mdのパスを返す"""
    return str(fixtures_dir / "sample.md")


@pytest.fixture
def sample_csv_path(fixtures_dir):
    """sample.csvのパスを返す"""
    return str(fixtures_dir / "sample.csv")


@pytest.fixture
def sample_json_path(fixtures_dir):
    """sample.jsonのパスを返す"""
    return str(fixtures_dir / "sample.json")


@pytest.fixture
def sample_text():
    """サンプルテキストを返す"""
    return """これはテスト用の日本語テキストです。

RAGシステムのテストに使用します。
このテキストは複数の段落から構成されています。

第一段落では基本的な情報を提供します。
第二段落では詳細な説明を行います。
第三段落ではまとめを記述します。"""


@pytest.fixture
def sample_long_text():
    """長いサンプルテキストを返す（チャンク分割テスト用）"""
    paragraphs = []
    for i in range(10):
        paragraphs.append(
            f"これは段落{i+1}です。" * 50 +
            f"この段落にはテスト用の日本語テキストが含まれています。" * 10
        )
    return "\n\n".join(paragraphs)


@pytest.fixture
def mock_config(monkeypatch):
    """テスト用の設定をモックする"""
    # 環境変数を設定
    monkeypatch.setenv("OLLAMA_HOST", "localhost")
    monkeypatch.setenv("OLLAMA_PORT", "11434")
    monkeypatch.setenv("QDRANT_HOST", "localhost")
    monkeypatch.setenv("QDRANT_PORT", "6333")
    monkeypatch.setenv("CHUNK_SIZE", "800")
    monkeypatch.setenv("CHUNK_OVERLAP", "150")

    # 設定を再読み込み
    from config import Config
    return Config()


@pytest.fixture(scope="session")
def test_collection_name():
    """テスト用のコレクション名を返す"""
    return "test_documents"


# pytestの設定
def pytest_configure(config):
    """pytest設定"""
    config.addinivalue_line(
        "markers", "integration: 統合テスト（実際のサービスが必要）"
    )
    config.addinivalue_line(
        "markers", "slow: 実行時間が長いテスト"
    )
