# Local RAG - 日本語対応ローカルLLMによるRAG推論実行環境

OllamaとLlama-3.1-Swallow-8B-Instruct-v0.1を使用した、完全ローカル環境で動作する日本語対応RAGシステムです。

## 特徴

- **完全ローカル実行**: 外部APIへの依存なし、プライベートデータを安全に処理
- **日本語最適化**: Llama-3.1-Swallow-8B-Instruct-v0.1による高精度な日本語処理
- **GPU高速化対応**: NVIDIA GPU使用で推論速度が最大18倍向上
- **Docker環境**: 簡単なセットアップと環境の再現性
- **高性能ベクターDB**: Qdrantによる高速な類似度検索
- **柔軟なドキュメント対応**: PDF、TXT、MD、CSV、JSON形式をサポート
- **包括的なテスト**: 84個の単体テスト + E2Eテストで品質保証

## 技術スタック

| コンポーネント | 技術 |
|------------|------|
| LLM | Ollama + Llama-3.1-Swallow-8B-Instruct-v0.1 (Q4_K_M量子化) |
| 埋め込みモデル | nomic-embed-text |
| ベクターDB | Qdrant |
| フレームワーク | LangChain |
| 言語 | Python 3.11 |
| 実行環境 | Docker / Docker Compose |

## 前提条件

- Docker & Docker Compose
- ディスク空き容量: 10GB以上（モデル保存用）
- メモリ: 8GB以上推奨
- **GPU使用時（オプション）**:
  - NVIDIA GPU（VRAM 6GB以上推奨）
  - NVIDIA Container Toolkit
  - CUDA 12.x対応ドライバー

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

### 実際の動作例

#### 例1: 高市早苗Q&Aデータセットの質問応答

```bash
# JSONLデータをテキスト形式に変換
python3 scripts/convert_jsonl_to_txt.py input.jsonl documents/qa_data.txt

# データを取り込み（チャンクサイズを小さめに設定）
docker exec local-rag-app bash -c "cd /app && python ingest.py --source /documents/qa_data.txt --collection qa_data --chunk-size 300 --chunk-overlap 50 --force"

# テストで動作確認
docker exec local-rag-app bash -c "cd /app && PYTHONPATH=/app pytest /tests/test_takaichi_qa.py -v -s"
```

**出力例:**

```
質問: 高市早苗さんの生年月日は？
回答: 高市早苗さんの生年月日は1961年3月7日です。

質問: 高市早苗さんの出身地はどこですか？
回答: 高市早苗さんの出身地は奈良県です。

質問: 高市早苗さんは何年生まれですか？
回答: 高市早苗さんは1961年3月7日生まれです。したがって、2025年現在は64歳です。
```

#### 例2: 類似度検索による関連情報の取得

RAGシステムは質問に対して、ベクターDBから関連性の高いドキュメントを類似度スコア付きで自動検索します。

**検索の仕組み:**
1. 質問文を埋め込みモデルでベクトル化
2. Qdrantで類似度検索を実行（コサイン類似度を使用）
3. スコアの高い順にドキュメントを取得（デフォルトTOP-5）
4. 取得したドキュメントをコンテキストとしてLLMに渡す

**出力例（質問: 高市早苗さんの政治経歴について教えてください）:**

```
[1] スコア: 0.8011
    内容: Q: 高市早苗さんの政治スタイルは？
          A: 論理的で実務的なアプローチを重視する政治スタイルです。

[2] スコア: 0.7937
    内容: Q: 高市早苗さんは討論番組によく出演しますか？
          A: はい、政治討論番組に頻繁に出演しています。

[3] スコア: 0.7914
    内容: Q: 高市早苗さんが尊敬する政治家は誰ですか？
          A: 安倍晋三元首相を深く尊敬していました。
```

**スコアの解釈:**
- **0.8以上**: 質問に直接関連する内容
- **0.7〜0.8**: 間接的に関連する情報
- **0.7未満**: 関連性が低い可能性

#### 例3: GPU使用による高速推論

GPU（NVIDIA）を使用することで推論速度が大幅に向上します。

```bash
# GPU使用状況の確認
nvidia-smi

# 推論速度の比較
docker exec local-rag-app bash -c "cd /app && PYTHONPATH=/app pytest /tests/test_takaichi_qa.py::TestTakaichiQA::test_query_birth_date -v"
```

**パフォーマンス:**
- **CPU使用時**: 約34秒/質問
- **GPU使用時**: 約1.9秒/質問（**18倍高速化**）
- **GPUメモリ**: 約8.6GB使用（RTX 5090の場合）

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
OLLAMA_LLM_MODEL=hf.co/mmnga/tokyotech-llm-Llama-3.1-Swallow-8B-Instruct-v0.1-gguf:Q4_K_M
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
docker exec local-rag-ollama ollama pull hf.co/mmnga/tokyotech-llm-Llama-3.1-Swallow-8B-Instruct-v0.1-gguf:Q4_K_M
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
# 全テストの実行
docker exec local-rag-app bash -c "cd /app && PYTHONPATH=/app pytest /tests/ -v"

# 特定のテストのみ実行
docker exec local-rag-app bash -c "cd /app && PYTHONPATH=/app pytest /tests/test_takaichi_qa.py -v -s"

# 単体テストのみ実行
docker exec local-rag-app bash -c "cd /app && PYTHONPATH=/app pytest /tests/test_llm.py /tests/test_embeddings.py -v"

# E2Eテストのみ実行
docker exec local-rag-app bash -c "cd /app && PYTHONPATH=/app pytest /tests/test_takaichi_qa.py -v"
```

**テストカバレッジ:**
- 単体テスト: 84個（モデル、ローダー、分割、テンプレートなど）
- E2Eテスト: 8個（高市早苗Q&Aデータセット使用）
- 総合カバレッジ: 92個のテストケース

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

Issue、Pull Requestを歓迎します。

## 参考

- [Ollama](https://ollama.ai/)
- [Llama-3.1-Swallow](https://huggingface.co/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1)
- [Qdrant](https://qdrant.tech/)
- [LangChain](https://www.langchain.com/)
