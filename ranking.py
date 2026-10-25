"""
ranking.py  (Чат 5: персонализированный отбор и ранжирование новостей)

Два уровня, чтобы не тратить бюджет 500р/день:
  1) правила (бесплатно): сопоставляем topic_tags новости с профилем клиента
     (отрасль, чувствительности, продукты банка) -> балл 0..1 и список
     matched_client_features -- по ним потом объясняем, ПОЧЕМУ новость выбрана;
  2) LLM (дорого): переранжируем только top-N кандидатов одной пачкой.
Если LLM недоступна или вернула мусор -- работаем на правилах, прогон не падает.
"""

import logging
from datetime import date, datetime
from typing import Optional

log = logging.getLogger("ranking")

# Какие поля профиля можно отправлять во внешний вызов LLM (allowlist).
# Платёжные суммы, идентификаторы контрагентов и соседей по группе НЕ отправляются.
LLM_PROFILE_ALLOWLIST = [
    "industry_sector", "industry_name", "okved_name", "revenue_bucket", "products",
    "sensitivities", "region", "city", "company_age", "stage", "fx_exposed",
    "counterparty_count",
]

# Смежные отрасли: совпадение даёт половинный вес
SECTOR_RELATED = {
    "construction": ["construction_realestate"],
    "construction_realestate": ["construction"],
    "wholesale_trade": ["trade"],
    "trade": ["wholesale_trade", "consumption"],
    "consumption": ["trade"],
    "metals_mining": ["commodities"],
    "oil_gas": ["commodities"],
}

# чувствительность клиента (подстрока в названии) -> теги новостей
SENSITIVITY_TAGS = [
    (("key_rate", "ставк"), ["key_rate", "inflation"]),
    (("валют", "fx", "внешнеторг"), ["fx", "sanctions"]),
    (("госконтракт", "тендер"), ["gov_contracts"]),
    (("сырь", "commod"), ["commodities"]),
    (("инфляц",), ["inflation"]),
]

# продукт банка -> теги новостей, которые на него влияют
PRODUCT_TAGS = {
    "loan": ["key_rate"], "leasing": ["key_rate"], "factoring": ["key_rate"],
    "deposit": ["key_rate"], "guarantee": ["gov_contracts"],
}


def sanitize_profile(profile: dict) -> dict:
    """Профиль для промпта LLM: только разрешённые поля (безопасность, п.6 ТЗ)."""
    out = {k: profile[k] for k in LLM_PROFILE_ALLOWLIST if k in profile and profile[k] not in (None, "")}
    peers = profile.get("group_peers")
    if isinstance(peers, list):
        out["group_peers_count"] = len(peers)  # только число, без идентификаторов
    return out


def _parse_date(s: str) -> Optional[date]:
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def rule_score(profile: dict, item: dict, today: Optional[date] = None) -> tuple:
    """Возвращает (score 0..1, matched_client_features). Кандидат = есть хотя бы
    одно реальное совпадение с профилем (не только свежесть/макро)."""
    today = today or date.today()
    tags = set(item.get("topic_tags") or [])
    matched, score = [], 0.0

    sector = profile.get("industry_sector")
    if sector and sector in tags:
        score += 0.45
        matched.append(f"industry_sector={sector}")
    else:
        rel = set(SECTOR_RELATED.get(sector, [])) & tags
        if rel:
            score += 0.22
            matched.append(f"industry_sector~{sector} (смежный тег {','.join(sorted(rel))})")

    sens_hit_total = 0
    for sens in profile.get("sensitivities") or []:
        s_low = str(sens).lower()
        want = set()
        for needles, tgs in SENSITIVITY_TAGS:
            if any(n in s_low for n in needles):
                want.update(tgs)
        if want & tags:
            sens_hit_total += 1
            matched.append(f"sensitivities={sens}")
    if sens_hit_total:
        score += min(0.25 + 0.05 * (sens_hit_total - 1), 0.35)

    prod_hits = [p for p in profile.get("products") or [] if set(PRODUCT_TAGS.get(p, [])) & tags]
    if prod_hits:
        score += 0.15
        matched.append("products=" + ",".join(prod_hits))

    if profile.get("fx_exposed") and ({"fx", "sanctions"} & tags):
        score += 0.1
        matched.append("fx_exposed=true")

    if not matched:
        return 0.0, []

    if "macro" in tags:
        score += 0.05
    d = _parse_date(item.get("published_at", ""))
    if d:
        age = (today - d).days
        score += 0.1 if age <= 1 else 0.05 if age <= 3 else 0.0
    return round(min(score, 1.0), 3), matched


def prefilter(profile: dict, news: list, top_n: int = 12, today: Optional[date] = None) -> list:
    scored = []
    for item in news:
        sc, matched = rule_score(profile, item, today)
        if matched:
            scored.append({"item": item, "rule_score": sc, "matched": matched})
    scored.sort(key=lambda x: x["rule_score"], reverse=True)
    return scored[:top_n]


def _slim(item: dict) -> dict:
    return {"event_id": item["event_id"], "title": item["title"],
            "topic_tags": item.get("topic_tags", []), "summary_facts": item.get("summary_facts", [])}


def rank(profile: dict, news: list, top_k: int = 3, prefilter_n: int = 12,
         min_score: float = 0.3, llm=None, llm_weight: float = 0.6,
         today: Optional[date] = None, model_label: str = "") -> list:
    """Итог: список {item, score, matched_client_features, ranking_model}, лучшие первыми."""
    cands = prefilter(profile, news, prefilter_n, today)
    if not cands:
        log.info("ранжирование: кандидатов нет")
        return []

    llm_scores, ranking_model = {}, "rule-based"
    if llm is not None:
        try:
            raw = llm.rank_news_for_client(sanitize_profile(profile),
                                           [_slim(c["item"]) for c in cands])
            ids = {c["item"]["event_id"] for c in cands}
            for r in raw if isinstance(raw, list) else []:
                try:
                    eid, sc = r["event_id"], float(r["score"])
                except (KeyError, TypeError, ValueError):
                    continue
                if eid in ids and 0.0 <= sc <= 1.0:
                    llm_scores[eid] = (sc, r.get("matched_client_features") or [])
            if llm_scores:
                ranking_model = model_label or "LLM bulk_rank + rule-based"
            else:
                log.warning("LLM вернула пустой/некорректный ранжир -- используем правила")
                ranking_model = "rule-based (LLM ответ не разобран)"
        except Exception as e:
            log.warning("LLM-ранжирование недоступно (%s) -- используем правила", e)
            ranking_model = "rule-based (LLM недоступна)"

    out = []
    for c in cands:
        eid = c["item"]["event_id"]
        final, feats = c["rule_score"], list(c["matched"])
        if eid in llm_scores:
            sc, llm_feats = llm_scores[eid]
            final = round(llm_weight * sc + (1 - llm_weight) * c["rule_score"], 3)
            # признаки, которые назвала LLM и которых нет у правил, добавляем как есть
            feats += [f for f in llm_feats if isinstance(f, str) and f not in feats]
        if final >= min_score:
            out.append({"item": c["item"], "score": final,
                        "matched_client_features": feats, "ranking_model": ranking_model})
    out.sort(key=lambda x: x["score"], reverse=True)
    log.info("ранжирование: кандидатов=%d, выбрано=%d (%s)", len(cands), len(out[:top_k]), ranking_model)
    return out[:top_k]
