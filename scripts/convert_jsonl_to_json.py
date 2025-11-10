#!/usr/bin/env python3
"""
JSONL形式のQ&Aデータを構造化JSON形式に変換するスクリプト
"""

import json
import sys
from pathlib import Path


def convert_jsonl_to_json(jsonl_file: str, output_file: str):
    """
    JSONL形式のQ&Aファイルを構造化JSON形式に変換

    Args:
        jsonl_file: 入力JSONLファイルのパス
        output_file: 出力JSONファイルのパス
    """
    qa_data = {
        "title": "高市早苗さんに関するQ&A",
        "description": "高市早苗さんについての質問と回答のデータセット",
        "qa_pairs": []
    }

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
                    qa_data["qa_pairs"].append({
                        "id": line_no,
                        "question": question,
                        "answer": answer
                    })

            except json.JSONDecodeError as e:
                print(f"警告: {line_no}行目のJSONパースに失敗しました: {e}", file=sys.stderr)
                continue

    # ファイルに書き込み
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 変換完了: {len(qa_data['qa_pairs'])}件のQ&Aを出力しました")
    print(f"   出力ファイル: {output_file}")


def main():
    """メイン処理"""
    if len(sys.argv) < 3:
        print("使用法: python convert_jsonl_to_json.py <input_jsonl> <output_json>")
        sys.exit(1)

    jsonl_file = sys.argv[1]
    output_file = sys.argv[2]

    # ファイル存在チェック
    if not Path(jsonl_file).exists():
        print(f"エラー: ファイルが見つかりません: {jsonl_file}", file=sys.stderr)
        sys.exit(1)

    convert_jsonl_to_json(jsonl_file, output_file)


if __name__ == "__main__":
    main()
