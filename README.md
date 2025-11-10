# Local RAG - 日本語対応ローカルLLMによるRAG推論実行環境

OllamaとLlama-3-Swallow-8B-Instruct-v0.1を使用した、完全ローカル環境で動作する日本語対応RAGシステムです。

## 特徴

- **完全ローカル実行**: 外部APIへの依存なし、プライベートデータを安全に処理
- **日本語最適化**: Llama-3-Swallow-8B-Instruct-v0.1による高精度な日本語処理
- **Docker環境**: 簡単なセットアップと環境の再現性
- **高性能ベクターDB**: Qdrantによる高速な類似度検索
- **柔軟なドキュメント対応**: PDF、TXT、MD、CSV、JSON形式をサポート

## 技術スタック

| コンポーネント | 技術 |
|------------|------|
| LLM | Ollama + Llama-3-Swallow-8B-Instruct-v0.1 |
| 埋め込みモデル | nomic-embed-text |
| ベクターDB | Qdrant |
| フレームワーク | LangChain |
| 言語 | Python 3.11 |
| 実行環境 | Docker / Docker Compose |

## 前提条件

- Docker & Docker Compose
- ディスク空き容量: 10GB以上（モデル保存用）
- メモリ: 8GB以上推奨（GPU使用時はVRAM 6GB以上推奨）

## セットアップ

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd local-rag
```

### 2. 環境変数の設定

```bash
# .envファイルは既に用意されていますが、必要に応じて編集
cp .env.example .env
```

### 3. Docker環境の起動

```bash
# セットアップスクリプトを実行
./scripts/setup.sh

# または手動で起動
docker-compose up -d
```

### 4. モデルのダウンロード

```bash
# モデルダウンロードスクリプトを実行（初回のみ、約5GB）
./scripts/pull_models.sh
```

**注意**: モデルのダウンロードには時間がかかります（環境により10〜30分程度）。

## 使い方

### ドキュメント取り込み

```bash
# ドキュメントをdocuments/ディレクトリに配置
cp your_documents.pdf documents/pdf/

# ドキュメントを取り込み
docker exec local-rag-app python ingest.py --source /documents
```

**オプション**:
- `--source`: ドキュメントファイルまたはディレクトリのパス（必須）
- `--collection`: コレクション名（デフォルト: documents）
- `--chunk-size`: チャンクサイズ（デフォルト: 800）
- `--chunk-overlap`: オーバーラップ（デフォルト: 150）
- `--force`: 既存のコレクションを削除して再作成

### 質問実行（単発）

```bash
docker exec local-rag-app python query.py --question "あなたの質問"
```

**オプション**:
- `--question`: 質問文（必須）
- `--collection`: コレクション名
- `--top-k`: 取得するコンテキスト数（デフォルト: 4）
- `--temperature`: LLM温度パラメータ（デフォルト: 0.7）
- `--show-context`: 取得したコンテキストを表示

### 対話モード

```bash
docker exec -it local-rag-app python main.py --interactive
```

対話モードでは以下のコマンドが使用できます:
- 質問を入力: そのまま質問文を入力
- `info`: システム情報を表示
- `exit` / `quit`: 終了

## ディレクトリ構成

```
local-rag/
├── docker-compose.yml          # Docker Compose設定
├── .env                        # 環境変数
├── app/                        # アプリケーションコード
│   ├── config.py              # 設定管理
│   ├── ingest.py              # ドキュメント取り込み
│   ├── query.py               # RAG推論実行
│   ├── main.py                # 対話モード
│   ├── models/                # LLMと埋め込みモデル
│   ├── vector_store/          # Qdrantクライアント
│   ├── loaders/               # ドキュメントローダー
│   ├── prompts/               # プロンプトテンプレート
│   └── utils/                 # ユーティリティ
├── documents/                  # ドキュメント格納
│   ├── pdf/
│   ├── txt/
│   └── md/
└── scripts/                    # セットアップスクリプト
    ├── setup.sh
    └── pull_models.sh
```

## サポートドキュメント形式

- **PDF** (.pdf): pypdf使用
- **テキスト** (.txt): UTF-8エンコーディング
- **Markdown** (.md): UnstructuredMarkdownLoader
- **CSV** (.csv): CSVLoader
- **JSON** (.json): JSONLoader

## 環境変数

主要な環境変数（`.env`ファイル）:

```bash
# Ollama設定
OLLAMA_HOST=ollama
OLLAMA_PORT=11434
OLLAMA_LLM_MODEL=mmnga/llama-3-swallow-8b-instruct-v0.1:q4_k_m
OLLAMA_EMBED_MODEL=nomic-embed-text

# Qdrant設定
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=documents

# RAG設定
CHUNK_SIZE=800
CHUNK_OVERLAP=150
TOP_K=4
TEMPERATURE=0.7
MAX_TOKENS=2000
```

## パフォーマンスチューニング

### チャンキングパラメータ

- `CHUNK_SIZE`: 大きくすると文脈が保たれるが、検索精度が下がる可能性
- `CHUNK_OVERLAP`: 大きくすると文脈の連続性が向上するが、重複が増加

### 検索パラメータ

- `TOP_K`: 大きくするとより多くのコンテキストを参照するが、ノイズも増加
- `TEMPERATURE`: 低い（0.0〜0.3）と決定的、高い（0.7〜1.0）と創造的

## トラブルシューティング

### モデルのダウンロードが失敗する

```bash
# Ollamaコンテナのログを確認
docker logs local-rag-ollama

# 手動でモデルをpull
docker exec local-rag-ollama ollama pull mmnga/llama-3-swallow-8b-instruct-v0.1:q4_k_m
docker exec local-rag-ollama ollama pull nomic-embed-text
```

### Qdrantに接続できない

```bash
# Qdrantコンテナの状態確認
docker ps
docker logs local-rag-qdrant

# ヘルスチェック
curl http://localhost:6333/healthz
```

### メモリ不足

```bash
# Dockerのメモリ制限を確認・増加
# Docker Desktopの設定でメモリを8GB以上に設定
```

### 日本語が文字化けする

- すべてのドキュメントがUTF-8エンコーディングであることを確認
- テキストファイルの場合、BOMなしのUTF-8を使用

## 開発

### ローカルでの開発

```bash
# Pythonパッケージのインストール
cd app
pip install -r requirements.txt

# コード修正後、コンテナを再起動
docker-compose restart rag-app
```

### テスト

```bash
# テストの実行（実装予定）
docker exec local-rag-app pytest tests/
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

Issue、Pull Requestを歓迎します。

## 参考

- [Ollama](https://ollama.ai/)
- [Llama-3-Swallow](https://huggingface.co/tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1)
- [Qdrant](https://qdrant.tech/)
- [LangChain](https://www.langchain.com/)
