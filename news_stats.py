"""
news_stats.py -- быстрая проверка качества парсера на реальной базе.

Запуск:  python news_stats.py --db news.db
Показывает: сколько новостей на каждый тег, сколько новостей с 1/2/3+ тегами,
случайные 15 новостей с тегами (для глазной проверки) и 15 отброшенных
заголовков (какие новости парсер счёл нерелевантными).
"""
import argparse
import json
import random
import sqlite3
from collections import Counter

import news_parser as npar

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="news.db")
ap.add_argument("--sample", type=int, default=15)
args = ap.parse_args()

conn = sqlite3.connect(args.db)
rows = conn.execute("SELECT title, topic_tags, source FROM news_items").fetchall()
print(f"В базе новостей: {len(rows)}\n")

tag_counts = Counter()
tags_per_item = Counter()
for _, tags, _ in rows:
    t = json.loads(tags)
    tags_per_item[min(len(t), 3)] += 1
    tag_counts.update(t)

print("Новостей на тег:")
for tag, n in tag_counts.most_common():
    print(f"  {tag:<26}{n}")
print("\nТегов на новость: 1 -> %d, 2 -> %d, 3+ -> %d" % (
    tags_per_item[1], tags_per_item[2], tags_per_item[3]))

print(f"\n--- {args.sample} случайных сохранённых (проверь, правильные ли теги) ---")
for title, tags, src in random.sample(rows, min(args.sample, len(rows))):
    print(f"[{src}] {title[:90]}\n      -> {json.loads(tags)}")

print(f"\n--- {args.sample} случайных ОТБРОШЕННЫХ (нет тегов; нет ли среди них экономики?) ---")
dropped = []
for src in npar.SOURCES:
    dropped += [r for r in npar.fetch_source(src) if not npar.tag_text(r["title"] + " " + r["body"])]
for r in random.sample(dropped, min(args.sample, len(dropped))):
    print(f"[{r['source']}] {r['title'][:100]}")
