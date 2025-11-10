"""
プロンプトテンプレートモジュールのテスト
"""

import pytest
from langchain_core.documents import Document
from app.prompts.templates import (
    create_rag_prompt,
    format_documents,
    create_prompt_with_context,
    RAG_PROMPT_TEMPLATE
)


class TestRAGPromptTemplate:
    """RAG_PROMPT_TEMPLATEのテスト"""

    def test_template_contains_placeholders(self):
        """テンプレートにプレースホルダーが含まれることを確認"""
        assert "{context}" in RAG_PROMPT_TEMPLATE
        assert "{question}" in RAG_PROMPT_TEMPLATE

    def test_template_is_japanese(self):
        """テンプレートが日本語であることを確認"""
        assert "日本語" in RAG_PROMPT_TEMPLATE or "回答" in RAG_PROMPT_TEMPLATE


class TestCreateRAGPrompt:
    """create_rag_prompt関数のテスト"""

    def test_create_rag_prompt(self):
        """プロンプトテンプレート作成のテスト"""
        prompt = create_rag_prompt()

        assert prompt is not None
        assert hasattr(prompt, 'format')
        assert "context" in prompt.input_variables
        assert "question" in prompt.input_variables

    def test_prompt_format(self):
        """プロンプトフォーマットのテスト"""
        prompt = create_rag_prompt()

        formatted = prompt.format(
            context="これはテストコンテキストです。",
            question="テスト質問は何ですか？"
        )

        assert "これはテストコンテキストです。" in formatted
        assert "テスト質問は何ですか？" in formatted


class TestFormatDocuments:
    """format_documents関数のテスト"""

    def test_format_empty_documents(self):
        """空のドキュメントリストのフォーマットテスト"""
        result = format_documents([])

        assert "関連する情報が見つかりませんでした" in result

    def test_format_single_document(self):
        """単一ドキュメントのフォーマットテスト"""
        documents = [
            Document(
                page_content="これはテストドキュメントです。",
                metadata={"file_name": "test.txt"}
            )
        ]

        result = format_documents(documents)

        assert "これはテストドキュメントです。" in result
        assert "test.txt" in result
        assert "[ドキュメント 1]" in result

    def test_format_multiple_documents(self):
        """複数ドキュメントのフォーマットテスト"""
        documents = [
            Document(
                page_content="第一のドキュメントです。",
                metadata={"file_name": "doc1.txt"}
            ),
            Document(
                page_content="第二のドキュメントです。",
                metadata={"file_name": "doc2.txt"}
            ),
            Document(
                page_content="第三のドキュメントです。",
                metadata={"file_name": "doc3.txt"}
            )
        ]

        result = format_documents(documents)

        # 全てのドキュメントが含まれることを確認
        assert "第一のドキュメントです。" in result
        assert "第二のドキュメントです。" in result
        assert "第三のドキュメントです。" in result

        # 全てのファイル名が含まれることを確認
        assert "doc1.txt" in result
        assert "doc2.txt" in result
        assert "doc3.txt" in result

        # ドキュメント番号が含まれることを確認
        assert "[ドキュメント 1]" in result
        assert "[ドキュメント 2]" in result
        assert "[ドキュメント 3]" in result

    def test_format_document_without_file_name(self):
        """ファイル名なしのドキュメントのフォーマットテスト"""
        documents = [
            Document(
                page_content="メタデータなしのドキュメント。",
                metadata={}
            )
        ]

        result = format_documents(documents)

        assert "メタデータなしのドキュメント。" in result
        assert "不明" in result

    def test_format_document_strips_whitespace(self):
        """前後の空白が削除されることを確認"""
        documents = [
            Document(
                page_content="  \n  テキスト  \n  ",
                metadata={"file_name": "test.txt"}
            )
        ]

        result = format_documents(documents)

        # 余分な空白が削除されている
        assert "テキスト" in result
        # 前後の空白は削除されるが、内部の改行は保持される可能性がある
        assert result.count("  \n  ") < 2

    def test_format_documents_separator(self):
        """ドキュメント間のセパレータが正しいことを確認"""
        documents = [
            Document(
                page_content="ドキュメント1",
                metadata={"file_name": "doc1.txt"}
            ),
            Document(
                page_content="ドキュメント2",
                metadata={"file_name": "doc2.txt"}
            )
        ]

        result = format_documents(documents)

        # ドキュメント間に改行が入ることを確認
        assert "\n\n" in result


class TestCreatePromptWithContext:
    """create_prompt_with_context関数のテスト"""

    def test_create_prompt_with_context(self):
        """コンテキスト付きプロンプト生成のテスト"""
        context = "これはテストコンテキストです。"
        question = "テストの質問は何ですか？"

        prompt = create_prompt_with_context(context, question)

        assert context in prompt
        assert question in prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_create_prompt_japanese_content(self):
        """日本語コンテンツでのプロンプト生成テスト"""
        context = "富士山は日本最高峰の山です。標高は3,776メートルです。"
        question = "富士山の高さは？"

        prompt = create_prompt_with_context(context, question)

        assert "富士山は日本最高峰の山です。" in prompt
        assert "富士山の高さは？" in prompt

    def test_create_prompt_empty_context(self):
        """空のコンテキストでのプロンプト生成テスト"""
        context = ""
        question = "質問です"

        prompt = create_prompt_with_context(context, question)

        assert "質問です" in prompt
        assert isinstance(prompt, str)

    def test_create_prompt_empty_question(self):
        """空の質問でのプロンプト生成テスト"""
        context = "コンテキストです"
        question = ""

        prompt = create_prompt_with_context(context, question)

        assert "コンテキストです" in prompt
        assert isinstance(prompt, str)

    def test_create_prompt_long_context(self):
        """長いコンテキストでのプロンプト生成テスト"""
        context = "テストコンテキスト。" * 100
        question = "質問"

        prompt = create_prompt_with_context(context, question)

        assert context in prompt
        assert question in prompt
        assert len(prompt) > len(context) + len(question)
