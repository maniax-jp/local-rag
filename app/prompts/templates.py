"""
プロンプトテンプレートモジュール
RAG用の日本語プロンプトテンプレート
"""

from langchain.prompts import PromptTemplate


# RAG用のプロンプトテンプレート
RAG_PROMPT_TEMPLATE = """あなたは親切で正確な日本語アシスタントです。
以下のコンテキスト情報を使用して、質問に回答してください。

コンテキスト:
{context}

質問: {question}

回答の際は以下の点に注意してください:
- コンテキストに基づいて正確に回答する
- コンテキストに情報がない場合は、その旨を正直に伝える
- 簡潔で分かりやすい日本語で回答する
- 推測や想像で回答せず、事実に基づいて回答する

回答:"""


def create_rag_prompt() -> PromptTemplate:
    """
    RAG用のプロンプトテンプレートを作成

    Returns:
        PromptTemplateインスタンス
    """
    return PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )


def format_documents(documents: list) -> str:
    """
    ドキュメントリストをコンテキスト文字列にフォーマット

    Args:
        documents: Documentオブジェクトのリスト

    Returns:
        フォーマットされたコンテキスト文字列
    """
    if not documents:
        return "関連する情報が見つかりませんでした。"

    context_parts = []
    for i, doc in enumerate(documents, 1):
        # ドキュメント内容
        content = doc.page_content.strip()

        # メタデータからソース情報を取得
        source = doc.metadata.get("file_name", "不明")

        context_parts.append(f"[ドキュメント {i}] (出典: {source})\n{content}")

    return "\n\n".join(context_parts)


def create_prompt_with_context(context: str, question: str) -> str:
    """
    コンテキストと質問からプロンプトを生成

    Args:
        context: コンテキスト文字列
        question: 質問文

    Returns:
        完成したプロンプト文字列
    """
    prompt = create_rag_prompt()
    return prompt.format(context=context, question=question)
