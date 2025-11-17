"""Extract texts from a SQLite DB into a raw JSONL for embedding.

Usage:
    python -m aparecium.scripts.extract_from_db \
        --db_path tweets.db \
        --table tweets \
        --text_col text \
        --limit 10000 \
        --out raw_10k.jsonl

Each output line: { "text": str }
"""

import argparse
import json
import sqlite3


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_path", type=str, required=True)
    ap.add_argument("--table", type=str, default="tweets")
    ap.add_argument("--text_col", type=str, default="text")
    ap.add_argument(
        "--where",
        type=str,
        default=None,
        help="Optional WHERE SQL (without the word WHERE)",
    )
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--out", type=str, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    conn = sqlite3.connect(args.db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    where_sql = f"WHERE {args.where}" if args.where else ""
    sql = f"SELECT {args.text_col} AS text FROM {args.table} {where_sql} LIMIT ?"

    with open(args.out, "w", encoding="utf-8") as fout:
        for row in cur.execute(sql, (int(args.limit),)):
            text = (row["text"] or "").strip()
            if not text:
                continue
            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    conn.close()
    print(f"Wrote raw texts to {args.out}")


if __name__ == "__main__":
    main()
