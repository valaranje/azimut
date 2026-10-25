"""
Тесты news_parser.py -- полностью офлайн (сеть не нужна).

Запуск:  pytest test_news_parser.py -v
"""
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import news_parser as np_  # noqa: E402


def _dt(days_ago=0):
    return datetime.now(timezone.utc) - timedelta(days=days_ago)


def _raw(url, title, body="", days_ago=0, source="test"):
    return {"source": source, "url": url, "title": title, "body": body,
            "published_dt": _dt(days_ago)}


FIXTURE = [
    _raw("https://a.ru/1", "ЦБ сохранил ключевую ставку на уровне 14,0%",
         "Банк России сохранил ключевую ставку на уровне 14,0% годовых. Решение принято 11 сентября."),
    # тот же URL -> дубль уровня 1
    _raw("https://a.ru/1", "ЦБ сохранил ключевую ставку на уровне 14,0%", "дубль по URL"),
    # другой URL, почти тот же заголовок -> дубль уровня 2 (fuzzy)
    _raw("https://b.ru/zzz", "Банк России сохранил ключевую ставку на уровне 14,0%",
         "Банк России сохранил ключевую ставку на уровне 14,0% годовых. Решение принято 11 сентября.",
         source="other"),
    _raw("https://a.ru/2", "Цены на сталь выросли на 5% из-за спроса со стороны строителей",
         "Российские металлурги повысили цены на сталь на 5% в сентябре. Спрос строителей вырос."),
    _raw("https://a.ru/3", "Урожай пшеницы в 2026 году составит 85 млн тонн",
         "Минсельхоз прогнозирует урожай зерна на уровне 130 млн тонн. Экспорт пшеницы сохранится."),
    _raw("https://a.ru/4", "Футбольный клуб выиграл кубок страны", "Матч завершился со счётом 2:1."),
    _raw("https://a.ru/5", "Старая новость про нефть и газ", "Добыча нефти снизилась.", days_ago=30),
]


def test_event_id_is_deterministic():
    assert np_.make_event_id("https://a.ru/1") == np_.make_event_id("https://a.ru/1 ")
    assert len(np_.make_event_id("https://a.ru/1")) == 16


def test_dedup_url_and_fuzzy():
    items = [np_.normalize(r) for r in FIXTURE[:3]]
    unique, dups = np_.deduplicate(items)
    assert len(unique) == 1
    assert dups == 2


def test_normalize_matches_contract_event_shape():
    item = np_.public_item(np_.normalize(FIXTURE[0]))
    assert set(item) == {"event_id", "title", "published_at", "source_type",
                         "topic_tags", "summary_facts"}
    assert item["source_type"] == "external_news"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", item["published_at"])  # только дата
    assert isinstance(item["topic_tags"], list) and isinstance(item["summary_facts"], list)
    assert "key_rate" in item["topic_tags"] and "finance" in item["topic_tags"]


def test_contract_schema_validation_if_available():
    try:
        import jsonschema
    except ImportError:
        return
    schema_path = os.path.join(os.path.dirname(__file__), "contract_schema.json")
    if not os.path.exists(schema_path):
        return
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    event_schema = schema["properties"]["event"]
    for r in FIXTURE[:4]:
        jsonschema.validate(np_.public_item(np_.normalize(r)), event_schema,
                            format_checker=jsonschema.FormatChecker())


def test_full_run_filters_and_saves(tmp_path):
    db = str(tmp_path / "n.db")
    stats = np_.run(db, days=7, sources=[{"name": "fx", "url": "x"}],
                    fetcher=lambda s: FIXTURE)
    assert stats["collected"] == len(FIXTURE)
    assert stats["too_old"] == 1            # 30-дневная
    assert stats["untagged_dropped"] == 1   # футбол
    assert stats["duplicates"] == 2
    assert stats["saved"] == 3              # ставка, сталь, пшеница

    # повторный прогон ничего не добавляет (идемпотентность)
    stats2 = np_.run(db, days=7, sources=[{"name": "fx", "url": "x"}],
                     fetcher=lambda s: FIXTURE)
    assert stats2["saved"] == 0

    conn = np_.open_db(db)
    news = np_.get_news(conn)
    assert len(news) == 3
    assert [n for n in news if "agro" in n["topic_tags"]]
    only_oil = np_.get_news(conn, tags=["oil_gas"])
    assert only_oil == []


def test_failed_source_does_not_crash_run(tmp_path):
    def fetcher(src):
        if src["name"] == "dead":
            return []  # fetch_source возвращает [] при любой сетевой ошибке
        return FIXTURE[:4]

    stats = np_.run(str(tmp_path / "n.db"), sources=[{"name": "dead", "url": "x"},
                                                      {"name": "alive", "url": "y"}],
                    fetcher=fetcher)
    assert stats["sources_failed"] == 1 and stats["sources_ok"] == 1
    assert stats["saved"] >= 1


def test_fetch_source_swallows_network_errors():
    import requests

    class Boom:
        def get(self, *a, **k):
            raise requests.exceptions.ConnectTimeout("timeout")

    assert np_.fetch_source({"name": "t", "url": "http://x"}, session=Boom()) == []


def test_fetch_source_parses_rss():
    rss = """<?xml version="1.0" encoding="utf-8"?>
    <rss version="2.0"><channel><title>t</title>
    <item><title>Нефть &amp; газ: добыча выросла</title><link>https://x.ru/1</link>
    <description>&lt;p&gt;Добыча нефти выросла на 3%.&lt;/p&gt;</description>
    <pubDate>Mon, 21 Sep 2026 10:00:00 +0300</pubDate></item>
    <item><title></title><link>https://x.ru/empty</link></item>
    </channel></rss>""".encode("utf-8")

    class Resp:
        content = rss
        def raise_for_status(self): pass

    class Sess:
        def get(self, *a, **k): return Resp()

    items = np_.fetch_source({"name": "t", "url": "http://x"}, session=Sess())
    assert len(items) == 1
    assert items[0]["title"] == "Нефть & газ: добыча выросла"
    assert items[0]["body"] == "Добыча нефти выросла на 3%."
    assert items[0]["published_dt"].year == 2026


def test_tags_do_not_match_inside_other_words():
    # "экспорт" не должен давать logistics, "рынок труда" -- metals_mining,
    # "добыча нефти" -- metals_mining
    assert "logistics" not in np_.tag_text("Экспорт пшеницы сохранится")
    assert "metals_mining" not in np_.tag_text("Данные по рынку труда не подтверждают охлаждения")
    assert "metals_mining" not in np_.tag_text("Добыча нефти снизилась")
    assert "oil_gas" in np_.tag_text("Добыча нефти снизилась")
    assert "logistics" in np_.tag_text("Грузы в порту выросли")


def test_regression_false_tags_from_real_run():
    # реальные заголовки из первого прогона, где теги были неверными
    assert np_.tag_text("«Локомотив» проиграл «Ростову»", "Матч подряд, стадион.") == []
    assert np_.tag_text("Россия выиграла второй матч подряд на турнире ФИФА") == []
    assert "logistics" not in np_.tag_text("Кредитный портфель Т-банка растет")
    assert "finance" in np_.tag_text("Кредитный портфель Т-банка растет")
    assert "construction_realestate" not in np_.tag_text(
        "В Татарстане при пожаре в многоквартирном доме пострадал человек")
    # "поставок" не должно давать finance/key_rate
    tags = np_.tag_text("Путин: Россия не отказывалась от поставок энергоресурсов в Европу")
    assert "finance" not in tags and "key_rate" not in tags


def test_bank_names_and_macro_news_are_tagged():
    assert "finance" in np_.tag_text("Греф: Сбербанк получит около 2 трлн рублей прибыли")
    assert "macro" in np_.tag_text("Денежная база в России выросла до 23,07 трлн рублей")
    assert "oil_gas" in np_.tag_text("«Газпром» в шесть раз сократил долю заимствований")


def test_body_needs_two_hits():
    # одно слово в тексте тег не даёт, два -- даёт
    assert np_.tag_text("Новость дня", "Упомянули нефть.") == []
    assert "oil_gas" in np_.tag_text("Новость дня", "Нефть подорожала, Brent выросла.")


def test_title_dedup_across_sources():
    a = np_.normalize(_raw("https://a.ru/x", "ЦБ сохранил ключевую ставку на уровне 14%",
                           "Первый текст про ставку."))
    b = np_.normalize(_raw("https://b.ru/y", "Банк России сохранил ключевую ставку на уровне 14%",
                           "Совсем другой текст, другие слова про решение регулятора.", source="o"))
    c = np_.normalize(_raw("https://c.ru/z", "Урожай картофеля в России снизился на 8,7%",
                           "Данные Росстата."))
    unique, dups = np_.deduplicate([a, b, c])
    assert dups == 1 and len(unique) == 2
