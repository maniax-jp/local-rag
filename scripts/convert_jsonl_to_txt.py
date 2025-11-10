#!/usr/bin/env python3
"""
JSONL形式のQ&Aデータをテキストファイルに変換するスクリプト
"""

import json
import sys
from pathlib import Path


def convert_jsonl_to_txt(jsonl_file: str, output_file: str):
    """
    JSONL形式のQ&Aファイルをテキストファイルに変換

    Args:
        jsonl_file: 入力JSONLファイルのパス
        output_file: 出力テキストファイルのパス
    """
    qa_pairs = []

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                messages = data.get('messages', [])

                # ユーザーの質問とアシスタントの回答を抽出
                question = None
                answer = None

                for msg in messages:
                    if msg.get('role') == 'user':
                        question = msg.get('content')
                    elif msg.get('role') == 'assistant':
                        answer = msg.get('content')

                if question and answer:
                    # Q&A形式のテキストを作成
                    qa_text = f"Q: {question}\nA: {answer}\n"
                    qa_pairs.append(qa_text)

            except json.JSONDecodeError as e:
                print(f"警告: {line_no}行目のJSONパースに失敗しました: {e}", file=sys.stderr)
                continue

    # ファイルに書き込み
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(qa_pairs))

    print(f"✅ 変換完了: {len(qa_pairs)}件のQ&Aを出力しました")
    print(f"   出力ファイル: {output_file}")


def main():
    """メイン処理"""
    if len(sys.argv) < 3:
        print("使用法: python convert_jsonl_to_txt.py <input_jsonl> <output_txt>")
        sys.exit(1)

    jsonl_file = sys.argv[1]
    output_file = sys.argv[2]

    # ファイル存在チェック
    if not Path(jsonl_file).exists():
        print(f"エラー: ファイルが見つかりません: {jsonl_file}", file=sys.stderr)
        sys.exit(1)

    convert_jsonl_to_txt(jsonl_file, output_file)


if __name__ == "__main__":
    main()
