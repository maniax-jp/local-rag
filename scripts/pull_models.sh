#!/bin/bash

# Ollamaモデルのダウンロードスクリプト

set -e

echo "==================================="
echo "Ollamaモデルのダウンロード"
echo "==================================="

# Llama-3.1-Swallow-8B-Instruct-v0.1のダウンロード
echo ""
echo "Llama-3.1-Swallow-8B-Instruct-v0.1をダウンロードします..."
echo "※ このモデルは約5GBあります。時間がかかる場合があります。"
docker exec local-rag-ollama ollama pull hf.co/mmnga/tokyotech-llm-Llama-3.1-Swallow-8B-Instruct-v0.1-gguf:Q4_K_M

# nomic-embed-textのダウンロード
echo ""
echo "nomic-embed-text（埋め込みモデル）をダウンロードします..."
docker exec local-rag-ollama ollama pull nomic-embed-text

# ダウンロード済みモデルの確認
echo ""
echo "ダウンロード済みモデル一覧:"
docker exec local-rag-ollama ollama list

echo ""
echo "==================================="
echo "モデルのダウンロード完了!"
echo "==================================="
echo ""
echo "使用可能なモデル:"
echo "- LLM: hf.co/mmnga/tokyotech-llm-Llama-3.1-Swallow-8B-Instruct-v0.1-gguf:Q4_K_M"
echo "- 埋め込み: nomic-embed-text"
echo ""
