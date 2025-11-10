FROM python:3.11-slim

WORKDIR /app

# システム依存パッケージのインストール
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードのコピー
COPY . /app

# 環境変数
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
