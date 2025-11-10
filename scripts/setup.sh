#!/bin/bash

# Local RAG環境のセットアップスクリプト

set -e

echo "==================================="
echo "Local RAG環境セットアップ開始"
echo "==================================="

# .envファイルの確認
if [ ! -f .env ]; then
    echo ".envファイルが見つかりません。.env.exampleからコピーします..."
    cp .env.example .env
    echo ".envファイルを作成しました。必要に応じて編集してください。"
fi

# Docker Composeで環境を起動
echo ""
echo "Docker Compose環境を起動します..."
docker-compose up -d

echo ""
echo "サービスの起動を待機しています..."
sleep 10

# サービスの状態確認
echo ""
echo "サービスの状態を確認中..."
docker-compose ps

# Ollamaの接続確認
echo ""
echo "Ollamaサービスの接続確認..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    echo "Ollamaの起動を待機中..."
    sleep 5
done
echo "Ollama: OK"

# Qdrantの接続確認
echo ""
echo "Qdrantサービスの接続確認..."
until curl -s http://localhost:6333/healthz > /dev/null 2>&1; do
    echo "Qdrantの起動を待機中..."
    sleep 5
done
echo "Qdrant: OK"

echo ""
echo "==================================="
echo "セットアップ完了!"
echo "==================================="
echo ""
echo "次のステップ:"
echo "1. モデルをダウンロード: ./scripts/pull_models.sh"
echo "2. ドキュメントを配置: documents/ディレクトリにファイルを配置"
echo "3. ドキュメント取り込み: docker exec local-rag-app python ingest.py --source /documents"
echo "4. 質問実行: docker exec local-rag-app python query.py --question '質問内容'"
echo ""
