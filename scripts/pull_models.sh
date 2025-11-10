#!/bin/bash

# Ollamaモデルのダウンロードスクリプト

set -e

echo "==================================="
echo "Ollamaモデルのダウンロード"
echo "==================================="

# Llama-3-Swallow-8B-Instruct-v0.1のダウンロード
echo ""
echo "Llama-3-Swallow-8B-Instruct-v0.1をダウンロードします..."
echo "※ このモデルは約5GBあります。時間がかかる場合があります。"
docker exec local-rag-ollama ollama pull mmnga/llama-3-swallow-8b-instruct-v0.1:q4_k_m

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
echo "- LLM: mmnga/llama-3-swallow-8b-instruct-v0.1:q4_k_m"
echo "- 埋め込み: nomic-embed-text"
echo ""
