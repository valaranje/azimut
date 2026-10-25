"""
news_parser.py  (Чат 2: парсер внешних новостей)

Автономный модуль: НЕ импортирует client_profiling.py и НЕ использует LLM.
Конвейер: fetch (RSS, с таймаутами) -> нормализация -> теги по ключевым
словам -> дедуп (URL-хэш + rapidfuzz по заголовку/началу текста) -> SQLite.

Отдаёт news_item строго по блоку "event" из contract_schema.json:
    {event_id, title, published_at (YYYY-MM-DD), source_type: "external_news",
     topic_tags[], summary_facts[]}

Установка:
    pip install feedparser rapidfuzz requests

Запуск:
    python news_parser.py --db news.db --days 7
    python news_parser.py --db news.db --export news_items.json --limit 5
"""

import argparse
import hashlib
import html
import json
import logging
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

import feedparser
import requests
from rapidfuzz import fuzz

log = logging.getLogger("news_parser")

# ---------------------------------------------------------------------------
# Источники. URL могут меняться/блокироваться -- это нормально: недоступный
# источник просто пропускается. Добавляй/убирай строки здесь.
# ---------------------------------------------------------------------------
SOURCES = [
    {"name": "rbc", "url": "https://rssexport.rbc.ru/rbcnews/news/30/full.rss"},
    {"name": "kommersant", "url": "https://www.kommersant.ru/RSS/news.xml"},
    {"name": "interfax", "url": "https://www.interfax.ru/rss.asp"},
    {"name": "vedomosti", "url": "https://www.vedomosti.ru/rss/news"},
    {"name": "tass", "url": "https://tass.ru/rss/v2.xml"},
]

REQUEST_TIMEOUT = 10  # секунд на один источник
USER_AGENT = "Mozilla/5.0 (compatible; alfa-hack-news-parser/1.0)"

# Теги отраслей -- те же 12 значений industry_sector, что в client_profiles.json.
# Если real OKVED_SECTOR_MAP в client_profiling.py отличается -- правь только тут.
#
# Ключи -- регулярные выражения. По умолчанию ищем С НАЧАЛА СЛОВА (чтобы "порт"
# не находился в "портфель", а "ставк" в "поставки"). Префикс "*" -- искать
# внутри слова (нужно для "Сбербанк", "Т-банк").
# Тег ставится, если ключ найден в ЗАГОЛОВКЕ, либо если в тексте совпало >= 2
# разных ключей (одно случайное слово в тексте тег не даёт).
SECTOR_KEYWORDS = {
    "metals_mining": ["металл", "сталь", "стальн", "металлург", r"руд(?:а|ы|у|е|ой)?\b",
                      "железорудн", "никел", "алюмин", r"мед(?:ь|и)\b", "золот",
                      "горнодобыв", "уголь", "угольн", "норникел"],
    "construction": ["строител", "застройщик", "стройматериал", "цемент", "подрядчик",
                     "подрядн", "инфраструктурн", "госконтракт", "тендер"],
    "construction_realestate": ["недвижим", r"жиль(?:е|я|ё)\b", "жилой", "жилых", "новостройк",
                                "девелопер", "ипотек", r"аренд(?:а|ы|у|ой|ных)\b"],
    "consumption": ["потребител", "розничн", "ритейл", "потребительск", "спрос населения",
                    "fmcg", "торговые сети"],
    "wholesale_trade": ["оптов", "дистрибь", "поставщик", r"импорт\w* товар", "товарооборот",
                        "закупк"],
    "trade": ["торговл", "маркетплейс", "e-commerce", "интернет-торговл", "ozon", "wildberries"],
    "oil_gas": ["нефт", r"газ(?:а|у|ом|е)?\b", "газпром", "роснефть", "лукойл", "опек", "opec",
                "нпз", "бензин", "дизельн", "urals", "brent", "спг", "энергоресурс", "энергоноситель"],
    "logistics": ["логистик", "перевозк", "грузоперевоз", r"порт(?:а|у|ы|ов|ом|е|ах)?\b",
                  "ржд", "железнодорожн", "контейнер", "фрахт", "авиаперевоз", "грузопоток",
                  "транспортн"],
    "it": ["it-компан", "программн", "software", r"искусственн\w* интеллект", "нейросет",
           r"цод\b", "кибербезопасн", "облачн", "цифров", "разработчик"],
    "finance": [r"*банк(?!ро)", "кредит", "ключев", r"цб\b", "банк россии", "ставк", "инфляц",
                "облигац", "офз", "страхов", "финансов", "мосбирж", "биржев", "минфин",
                "фондов", "инвестиц", "дивиденд", r"прибыль|прибыли"],
    "agro": ["агро", "урожа", "зерн", "пшениц", "сельхоз", "удобрен", "аграри", "животновод",
             "картофел", "минсельхоз"],
}

# Макро-теги, ложатся на sensitivities клиентов (key_rate, валюта и т.д.).
MACRO_KEYWORDS = {
    "key_rate": [r"ключев\w* ставк", r"ставк\w* цб", "банк россии", r"цб рф",
                 "денежно-кредитн", r"заседани\w* цб"],
    "inflation": ["инфляц", r"потребительск\w* цен", "рост цен", "дефляц"],
    "fx": [r"курс\w* (?:доллар|евро|юан|рубл)", "валютн", "девальвац",
           r"укреплени\w* рубл", r"ослаблени\w* рубл"],
    "sanctions": ["санкци", "эмбарго", "потолок цен", r"ограничени\w* на экспорт"],
    "gov_contracts": ["госзаказ", "госконтракт", "госзакуп", "тендер", "44-фз", "223-фз"],
    "commodities": [r"цен\w* на (?:нефть|золото|металл|газ)", "сырьев", "brent", "wti", "urals"],
    "macro": ["ввп", "экономик", r"денежн\w* баз", "бюджет", "налог", "профицит", "росстат",
              "минэкономразвит"],
}

SCHEMA_VERSION_NOTE = "published_at -- только дата (format: date в contract_schema.json)"

FUZZY_THRESHOLD = 85  # token_sort_ratio, % сходства
FUZZY_TEXT_CHARS = 200


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------
def make_event_id(url: str) -> str:
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:16]


def clean_text(raw: Optional[str]) -> str:
    if not raw:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def parse_entry_datetime(entry) -> Optional[datetime]:
    for key in ("published_parsed", "updated_parsed"):
        t = entry.get(key)
        if t:
            return datetime(*t[:6], tzinfo=timezone.utc)
    return None


def extract_facts(title: str, body: str, max_facts: int = 3) -> list:
    """Правило, без LLM: берём предложения с цифрами/датами в первую очередь,
    иначе -- первые предложения. Только факты из текста источника."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if len(s.strip()) > 25]
    with_numbers = [s for s in sentences if re.search(r"\d", s)]
    picked = []
    for s in with_numbers + sentences:
        if s not in picked:
            picked.append(s)
        if len(picked) >= max_facts:
            break
    if not picked and title:
        picked = [title]
    return picked[:max_facts]


def _compile(keywords: list) -> list:
    pats = []
    for kw in keywords:
        pat = kw[1:] if kw.startswith("*") else r"(?<!\w)" + kw
        pats.append(re.compile(pat, re.IGNORECASE))
    return pats


_TAG_PATTERNS = {tag: _compile(kws) for tag, kws in {**SECTOR_KEYWORDS, **MACRO_KEYWORDS}.items()}


def tag_text(title: str, body: str = "") -> list:
    """Теги по ключевым словам (без LLM). Тег ставится, если ключ найден в
    заголовке ИЛИ в тексте совпало >= 2 разных ключей."""
    tags = []
    for tag, pats in _TAG_PATTERNS.items():
        if any(p.search(title) for p in pats):
            tags.append(tag)
        elif sum(1 for p in pats if p.search(body)) >= 2:
            tags.append(tag)
    return tags


def fuzzy_key(title: str, body: str) -> str:
    return (title + " " + body[:FUZZY_TEXT_CHARS]).lower()


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def fetch_source(source: dict, session: Optional[requests.Session] = None) -> list:
    """Возвращает список сырых записей. Любая ошибка -> [] + warning (прогон не падает)."""
    sess = session or requests
    try:
        resp = sess.get(source["url"], timeout=REQUEST_TIMEOUT,
                        headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
    except Exception as e:  # сеть, таймаут, 4xx/5xx, битый XML
        log.warning("источник %s недоступен: %s", source["name"], e)
        return []

    items = []
    for e in feed.entries:
        url = e.get("link", "").strip()
        title = clean_text(e.get("title"))
        if not url or not title:
            continue
        body = clean_text(e.get("summary") or e.get("description"))
        items.append({
            "source": source["name"],
            "url": url,
            "title": title,
            "body": body,
            "published_dt": parse_entry_datetime(e),
        })
    log.info("источник %s: получено %d записей", source["name"], len(items))
    return items


# ---------------------------------------------------------------------------
# Нормализация + дедуп
# ---------------------------------------------------------------------------
def normalize(raw: dict, now: Optional[datetime] = None) -> dict:
    now = now or datetime.now(timezone.utc)
    dt = raw.get("published_dt") or now  # нет даты у источника -> считаем "сегодня"
    tags = tag_text(raw["title"], raw["body"])
    return {
        "event_id": make_event_id(raw["url"]),
        "title": raw["title"],
        "published_at": dt.date().isoformat(),
        "source_type": "external_news",
        "topic_tags": tags,
        "summary_facts": extract_facts(raw["title"], raw["body"]),
        # служебные поля -- наружу в news_item не отдаются:
        "_url": raw["url"],
        "_source": raw["source"],
        "_published_dt": dt.isoformat(),
        "_fuzzy": fuzzy_key(raw["title"], raw["body"]),
    }


def _is_title_dup(t1: str, t2: str) -> bool:
    """Заголовки одного события из разных источников звучат по-разному, поэтому
    сравниваем и сами заголовки: token_set_ratio >= 85 при >= 4 словах в каждом
    (короткие заголовки не сравниваем так мягко -- иначе ложные дубли)."""
    a, b = t1.lower(), t2.lower()
    if len(a.split()) < 4 or len(b.split()) < 4:
        return False
    return fuzz.token_set_ratio(a, b) >= FUZZY_THRESHOLD


def deduplicate(items: list, known_fuzzy: Optional[list] = None,
                known_ids: Optional[set] = None) -> tuple:
    """Двухуровневый дедуп: (1) одинаковый event_id (=URL), (2) fuzzy по
    заголовку+началу текста. Возвращает (уникальные, число_дублей)."""
    seen_ids = set(known_ids or [])
    seen_fuzzy = list(known_fuzzy or [])
    seen_titles = []  # заголовки текущего прогона (из прошлых прогонов -- только fuzzy_key)
    unique, dups = [], 0
    for it in items:
        if it["event_id"] in seen_ids:
            dups += 1
            continue
        if any(fuzz.token_sort_ratio(it["_fuzzy"], f) >= FUZZY_THRESHOLD for f in seen_fuzzy) \
                or any(_is_title_dup(it["title"], t) for t in seen_titles):
            dups += 1
            continue
        seen_ids.add(it["event_id"])
        seen_fuzzy.append(it["_fuzzy"])
        seen_titles.append(it["title"])
        unique.append(it)
    return unique, dups


def public_item(row: dict) -> dict:
    """news_item строго по схеме: без служебных полей."""
    return {k: row[k] for k in ("event_id", "title", "published_at", "source_type",
                                "topic_tags", "summary_facts")}


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------
DDL = """
CREATE TABLE IF NOT EXISTS news_items (
    event_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    published_at TEXT NOT NULL,       -- YYYY-MM-DD (то, что уходит в контракт)
    published_dt TEXT,                -- полный timestamp UTC, для сортировки
    source_type TEXT NOT NULL DEFAULT 'external_news',
    topic_tags TEXT NOT NULL,         -- JSON-массив
    summary_facts TEXT NOT NULL,      -- JSON-массив
    url TEXT NOT NULL,
    source TEXT,
    fuzzy_key TEXT,
    fetched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_news_published ON news_items(published_at);
"""


def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(DDL)
    return conn


def load_known(conn: sqlite3.Connection, days: int) -> tuple:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    rows = conn.execute(
        "SELECT event_id, fuzzy_key FROM news_items WHERE published_at >= ?", (cutoff,)
    ).fetchall()
    return {r[0] for r in rows}, [r[1] for r in rows if r[1]]


def save_items(conn: sqlite3.Connection, items: list) -> int:
    now = datetime.now(timezone.utc).isoformat()
    n = 0
    for it in items:
        cur = conn.execute(
            "INSERT OR IGNORE INTO news_items VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (it["event_id"], it["title"], it["published_at"], it["_published_dt"],
             it["source_type"], json.dumps(it["topic_tags"], ensure_ascii=False),
             json.dumps(it["summary_facts"], ensure_ascii=False),
             it["_url"], it["_source"], it["_fuzzy"], now),
        )
        n += cur.rowcount
    conn.commit()
    return n


def get_news(conn: sqlite3.Connection, tags: Optional[list] = None,
             since: Optional[str] = None, limit: int = 100) -> list:
    """API для Чата 5 (ранжирование): news_item'ы из БД, опционально по тегам/дате."""
    q = "SELECT event_id,title,published_at,source_type,topic_tags,summary_facts FROM news_items"
    where, args = [], []
    if since:
        where.append("published_at >= ?")
        args.append(since)
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY published_dt DESC"
    out = []
    for r in conn.execute(q, args):
        item = {"event_id": r[0], "title": r[1], "published_at": r[2], "source_type": r[3],
                "topic_tags": json.loads(r[4]), "summary_facts": json.loads(r[5])}
        if tags and not set(tags) & set(item["topic_tags"]):
            continue
        out.append(item)
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Основной прогон
# ---------------------------------------------------------------------------
def run(db_path: str, days: int = 7, sources: Optional[list] = None,
        keep_untagged: bool = False, fetcher=fetch_source) -> dict:
    sources = sources if sources is not None else SOURCES
    conn = open_db(db_path)
    known_ids, known_fuzzy = load_known(conn, days)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    stats = {"sources_ok": 0, "sources_failed": 0, "collected": 0, "too_old": 0,
             "untagged_dropped": 0, "duplicates": 0, "saved": 0}

    normalized = []
    for src in sources:
        raw_items = fetcher(src)
        if not raw_items:
            stats["sources_failed"] += 1
            continue
        stats["sources_ok"] += 1
        for raw in raw_items:
            stats["collected"] += 1
            if raw["published_dt"] and raw["published_dt"] < cutoff:
                stats["too_old"] += 1
                continue
            item = normalize(raw, now)
            if not item["topic_tags"] and not keep_untagged:
                stats["untagged_dropped"] += 1
                continue
            if not item["topic_tags"]:
                item["topic_tags"] = ["other"]
            normalized.append(item)

    unique, dups = deduplicate(normalized, known_fuzzy, known_ids)
    stats["duplicates"] = dups
    stats["saved"] = save_items(conn, unique)
    conn.close()

    log.info("ИТОГ: собрано=%(collected)d, старше %(days)dд=%(too_old)d, без тем=%(untagged_dropped)d, "
             "дубли=%(duplicates)d, сохранено=%(saved)d (источников ок=%(sources_ok)d, "
             "недоступно=%(sources_failed)d)", {**stats, "days": days})
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="news.db")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--keep-untagged", action="store_true",
                    help="сохранять новости без отраслевых тегов с тегом 'other'")
    ap.add_argument("--export", help="сохранить news_item'ы в JSON-файл")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--no-fetch", action="store_true", help="только экспорт из БД")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.no_fetch:
        run(args.db, args.days, keep_untagged=args.keep_untagged)
    if args.export:
        conn = open_db(args.db)
        items = get_news(conn, limit=args.limit)
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        log.info("экспортировано %d news_item -> %s", len(items), args.export)


if __name__ == "__main__":
    sys.exit(main())
