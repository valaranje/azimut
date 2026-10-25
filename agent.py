"""
agent.py  (Чат 4: агентный цикл)

client_id -> PersonalizedInsight[] (формат contract_schema.json).

Шаги (требования ТЗ, слайд 7):
  1. planning   -- какие теги новостей и темы аналитики банка нужны этому клиенту
                   (правила по профилю, без LLM -- бесплатно и объяснимо);
  2. retrieval  -- новости из SQLite (news_parser) + выдержки из аналитики банка (RAG);
  3. ranking    -- ranking.rank (правила + LLM-переранжирование top-N);
  4. generation -- интерпретация по 6 вопросам (call_llm.generate_interpretation);
  5. self_check -- проверка; при ok=false повторяется ТОЛЬКО нужный шаг
                   (re-retrieve / re-rank / re-generate), максимум max_retries раз.
Каждый шаг пишется в provenance.steps (трассируемость, п.12 критериев).

Запуск:
    python agent.py --client U_9250 --out insights.json
"""

import argparse
import json
import logging
import os
import sqlite3
import uuid
from datetime import date, timedelta
from typing import Optional

import news_parser
import ranking

log = logging.getLogger("agent")

SECTOR_RU = {
    "metals_mining": "металлургия и горнодобыча", "construction": "строительство",
    "construction_realestate": "недвижимость и девелопмент", "consumption": "потребление",
    "wholesale_trade": "оптовая торговля", "trade": "торговля", "oil_gas": "нефть и газ",
    "logistics": "логистика и транспорт", "it": "IT", "finance": "финансы", "agro": "агросектор",
}
RAG_MIN_SCORE = 0.0  # порог схожести; подними (напр. до 0.78) после просмотра реальных score


# ---------------------------------------------------------------------------
# RAG: модель и индекс грузятся один раз
# ---------------------------------------------------------------------------
_RAG = {}


def default_rag_search(query: str, top_k: int = 3, index_path: str = "bank_rag_index.json") -> list:
    try:
        import numpy as np
        if "data" not in _RAG:
            from sentence_transformers import SentenceTransformer
            with open(index_path, encoding="utf-8") as f:
                data = json.load(f)
            if not data.get("embeddings"):
                raise RuntimeError("индекс RAG пуст")
            _RAG["data"] = data
            _RAG["emb"] = np.array(data["embeddings"])
            _RAG["model"] = SentenceTransformer(data["model"])
        q = _RAG["model"].encode(["query: " + query], normalize_embeddings=True)[0]
        scores = _RAG["emb"] @ q
        out = []
        for i in np.argsort(-scores)[:top_k]:
            r = _RAG["data"]["records"][i]
            out.append({"doc": r["doc"], "page": r["page"], "chunk_id": r["chunk_id"],
                        "score": float(scores[i]), "text": r["text"]})
        return out
    except Exception as e:
        log.warning("RAG недоступен (%s) -- продолжаем без аналитики банка", e)
        return []


# ---------------------------------------------------------------------------
class Agent:
    def __init__(self, profiles_path: str = "client_profiles.json", news_db: str = "news.db",
                 llm=None, rag_search=None, max_retries: int = 2, max_llm_calls: int = 12,
                 schema_path: str = "contract_schema.json", rank_model_label: str = ""):
        self.profiles = self._load_profiles(profiles_path)
        self.news_db = news_db
        self.llm = llm
        self.rag_search = rag_search or default_rag_search
        self.max_retries = max_retries
        self.max_llm_calls = max_llm_calls
        self.schema_path = schema_path
        self.rank_model_label = rank_model_label
        self._calls = 0

    # -- загрузка ----------------------------------------------------------
    @staticmethod
    def _load_profiles(path: str) -> dict:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        items = data.values() if isinstance(data, dict) else data
        return {p["client_id"]: p for p in items if "client_id" in p}

    def _budget_ok(self) -> bool:
        return self._calls < self.max_llm_calls

    def _llm_call(self, fn, *a, **kw):
        self._calls += 1
        return fn(*a, **kw)

    # -- шаг 1 -------------------------------------------------------------
    def plan(self, profile: dict) -> dict:
        tags = set()
        sector = profile.get("industry_sector")
        if sector:
            tags.add(sector)
            tags.update(ranking.SECTOR_RELATED.get(sector, []))
        for sens in profile.get("sensitivities") or []:
            s_low = str(sens).lower()
            for needles, tgs in ranking.SENSITIVITY_TAGS:
                if any(n in s_low for n in needles):
                    tags.update(tgs)
        for p in profile.get("products") or []:
            tags.update(ranking.PRODUCT_TAGS.get(p, []))
        tags.add("macro")

        queries = []
        if "key_rate" in tags:
            queries.append("ключевая ставка прогноз")
        if "fx" in tags:
            queries.append("курс рубля прогноз")
        if "inflation" in tags:
            queries.append("инфляция прогноз")
        if sector in SECTOR_RU:
            queries.append(SECTOR_RU[sector])
        return {"news_tags": sorted(tags), "bank_queries": queries or ["макроэкономика прогноз"],
                "need_graph": bool(profile.get("group_peers"))}

    # -- шаг 2 -------------------------------------------------------------
    def _news(self, tags: list, days: int, today: date) -> list:
        conn = news_parser.open_db(self.news_db)
        try:
            since = (today - timedelta(days=days)).isoformat()
            return news_parser.get_news(conn, tags=tags, since=since, limit=300)
        finally:
            conn.close()

    def _news_url(self, event_id: str) -> str:
        conn = sqlite3.connect(self.news_db)
        try:
            row = conn.execute("SELECT url FROM news_items WHERE event_id=?", (event_id,)).fetchone()
            return row[0] if row else ""
        finally:
            conn.close()

    def _snippets(self, queries: list, per_query: int = 2) -> list:
        seen, out = set(), []
        for q in queries:
            for r in self.rag_search(q, top_k=per_query):
                key = (r["doc"], r["chunk_id"])
                if key not in seen and r["score"] >= RAG_MIN_SCORE:
                    seen.add(key)
                    out.append(r)
        return out[:5]

    # -- основной прогон ---------------------------------------------------
    def run(self, client_id: str, today: Optional[date] = None, top_k: int = 1,
            days: int = 7) -> list:
        today = today or date.today()
        profile = self.profiles.get(client_id)
        if profile is None:
            raise KeyError(f"клиент {client_id} не найден в профилях")
        safe_profile = ranking.sanitize_profile(profile)  # ТОЛЬКО этот клиент, только allowlist
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        steps = []

        plan = self.plan(profile)
        steps.append({"step": "planning", "model_used": "rule-based",
                      "input_ref": f"client_profile:{client_id}",
                      "output_ref": f"plan:{client_id}:{today.isoformat()}",
                      "notes": f"tags={plan['news_tags']}; queries={plan['bank_queries']}"})

        news = self._news(plan["news_tags"], days, today)
        steps.append({"step": "retrieval", "model_used": "-",
                      "input_ref": f"plan:{client_id}", "output_ref": f"news candidates: {len(news)}",
                      "notes": f"SQLite {self.news_db}, окно {days} дн."})
        if not news:
            log.warning("для клиента %s нет подходящих новостей за %d дн.", client_id, days)
            return []

        ranked = ranking.rank(profile, news, top_k=max(top_k, 3), llm=self.llm if self._budget_ok() else None,
                              today=today, model_label=self.rank_model_label)
        if self.llm is not None:
            self._calls += 1  # пачка ранжирования = 1 вызов
        steps.append({"step": "ranking",
                      "model_used": ranked[0]["ranking_model"] if ranked else "rule-based",
                      "input_ref": f"candidates:{len(news)}",
                      "output_ref": "scores=" + ",".join(f"{r['score']:.2f}" for r in ranked),
                      "notes": "matched: " + "; ".join(ranked[0]["matched_client_features"]) if ranked else "нет"})
        if not ranked:
            return []

        results = []
        for cand in ranked[:top_k]:
            insight = self._process_event(client_id, profile, safe_profile, plan, cand, steps, run_id)
            if insight:
                results.append(insight)
        return results

    # -- шаги 4-5 для одного события --------------------------------------
    def _process_event(self, client_id, profile, safe_profile, plan, cand, steps, run_id):
        item = cand["item"]
        event_facts = {k: item[k] for k in ("title", "published_at", "summary_facts", "topic_tags")}
        queries = [item["title"]] + plan["bank_queries"]
        snippets = self._snippets(queries)
        snippets_payload = [{"source": f"{s['doc']}, стр. {s['page']}", "text": s["text"]} for s in snippets]

        if self.llm is None:
            log.warning("LLM не подключена -- интерпретация не генерируется")
            return None

        feedback, interp, check = None, None, {"ok": False, "issues": ["не проверено"], "action": "none"}
        action_taken = "none"
        for attempt in range(self.max_retries + 1):
            if not self._budget_ok():
                log.warning("лимит LLM-вызовов на прогон исчерпан (%d)", self.max_llm_calls)
                action_taken = "escalate"
                break
            try:
                interp = self._llm_call(self.llm.generate_interpretation, safe_profile, event_facts,
                                        snippets_payload, feedback)
            except Exception as e:
                log.error("генерация не удалась: %s", e)
                action_taken = "escalate"
                break
            steps.append({"step": "generation", "model_used": "generation (GLM)",
                          "input_ref": f"event:{item['event_id']}; snippets={len(snippets)}",
                          "output_ref": f"interpretation attempt {attempt + 1}",
                          "notes": "с учётом замечаний self-check" if feedback else ""})
            if not self._budget_ok():
                action_taken = "escalate"
                break
            try:
                check = self._llm_call(self.llm.self_check, interp, event_facts, snippets_payload)
            except Exception as e:
                log.error("self-check не удался: %s", e)
                check = {"ok": False, "issues": [f"self-check недоступен: {e}"], "action": "none"}
                action_taken = "escalate"
                steps.append({"step": "self_check", "model_used": "self_check (Kimi)",
                              "input_ref": "interpretation", "output_ref": "error", "notes": str(e)})
                break
            steps.append({"step": "self_check", "model_used": "self_check (Kimi)",
                          "input_ref": "interpretation", "output_ref": f"ok={check.get('ok')}",
                          "notes": "; ".join(check.get("issues") or [])})
            if check.get("ok"):
                break
            if attempt == self.max_retries:
                action_taken = "escalate"
                break
            action = check.get("action", "re-generate")
            feedback = check.get("issues") or []
            if action == "re-retrieve":
                extra = [item["title"] + " " + " ".join(feedback)[:150]]
                more = self._snippets(extra, per_query=3)
                known = {(s["doc"], s["chunk_id"]) for s in snippets}
                snippets += [s for s in more if (s["doc"], s["chunk_id"]) not in known]
                snippets = snippets[:6]
                snippets_payload = [{"source": f"{s['doc']}, стр. {s['page']}", "text": s["text"]}
                                    for s in snippets]
                steps.append({"step": "retrieval", "model_used": "-", "input_ref": "self_check issues",
                              "output_ref": f"snippets: {len(snippets)}", "notes": "повторный поиск (re-retrieve)"})
                action_taken = "re-retrieve"
            else:
                action_taken = "re-generate" if action != "re-rank" else "re-rank->re-generate"
            log.info("self-check не пройден, действие: %s", action_taken)

        if interp is None:
            return None
        return self._assemble(client_id, safe_profile, cand, interp, check, snippets, steps, run_id,
                              action_taken)

    # -- сборка по контракту ----------------------------------------------
    @staticmethod
    def _norm_consequences(raw) -> list:
        out = []
        for c in raw or []:
            if isinstance(c, dict) and c.get("text"):
                t = c.get("type") if c.get("type") in ("risk", "opportunity") else "risk"
                out.append({"type": t, "text": str(c["text"])})
            elif isinstance(c, str):
                low = c.lower()
                out.append({"type": "opportunity" if "возможност" in low else "risk", "text": c})
        return out

    def _assemble(self, client_id, safe_profile, cand, interp, check, snippets, steps, run_id,
                  action_taken) -> dict:
        item = cand["item"]
        basis = [str(b) for b in interp.get("basis_for_conclusion") or []]
        attribution = {"from_bank_analytics": [], "from_client_data": [], "from_model_inference": []}
        for b in basis:
            low = b.lower()
            if low.startswith("alfa_analytics") or low.startswith("bank"):
                attribution["from_bank_analytics"].append(b)
            elif low.startswith("client_data") or low.startswith("client"):
                attribution["from_client_data"].append(b)
            elif low.startswith("news"):
                continue  # источник новости виден в sources
            else:
                attribution["from_model_inference"].append(b)

        sources = [{"source_id": item["event_id"], "source_type": "external_news",
                    "title": item["title"], "url_or_doc": self._news_url(item["event_id"])}]
        for s in snippets:
            sources.append({"source_id": f"{s['doc']}#p{s['page']}", "source_type": "alfa_analytics",
                            "title": s["doc"], "url_or_doc": f"{s['doc']}, стр. {s['page']}"})
        sources.append({"source_id": f"client_profile_{client_id}", "source_type": "client_data",
                        "title": f"Профиль клиента {client_id}", "url_or_doc": "client_profiles.json"})

        ok = bool(check.get("ok"))
        insight = {
            "client_id": client_id,
            "_client_context_used": safe_profile,
            "event": {k: item[k] for k in ("event_id", "title", "published_at", "source_type",
                                           "topic_tags", "summary_facts")},
            "relevance": {"score": cand["score"],
                          "matched_client_features": cand["matched_client_features"],
                          "ranking_model": cand["ranking_model"]},
            "interpretation": {
                "what_happened": str(interp.get("what_happened", "")),
                "why_relevant_to_client": str(interp.get("why_relevant_to_client", "")),
                "possible_consequences": self._norm_consequences(interp.get("possible_consequences")),
                "possible_actions": [str(a) for a in interp.get("possible_actions") or []],
                "bank_view": str(interp.get("bank_view", "")),
                "basis_for_conclusion": basis,
            },
            "attribution": attribution,
            "sources": sources,
            "provenance": {
                "pipeline_run_id": run_id,
                "steps": steps,
                "self_check": {"completeness_ok": ok, "consistency_ok": ok,
                               "issues_found": list(check.get("issues") or []),
                               "action_taken": action_taken},
            },
        }
        self._validate(insight)
        return insight

    def _validate(self, insight: dict) -> None:
        try:
            import jsonschema
            with open(self.schema_path, encoding="utf-8") as f:
                schema = json.load(f)
            jsonschema.validate(insight, schema, format_checker=jsonschema.FormatChecker())
        except ImportError:
            pass
        except FileNotFoundError:
            log.warning("contract_schema.json не найден -- валидация пропущена")
        except Exception as e:
            log.warning("результат НЕ соответствует contract_schema.json: %s", str(e)[:300])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", required=True)
    ap.add_argument("--profiles", default="client_profiles.json")
    ap.add_argument("--db", default="news.db")
    ap.add_argument("--out", default="insights.json")
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--no-llm", action="store_true", help="только ранжирование по правилам")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    llm = None
    if not args.no_llm:
        import call_llm
        llm = call_llm
    agent = Agent(args.profiles, args.db, llm=llm)
    if args.no_llm:
        profile = agent.profiles[args.client]
        plan = agent.plan(profile)
        news = agent._news(plan["news_tags"], 7, date.today())
        for r in ranking.rank(profile, news, top_k=args.top_k + 4):
            print(f"{r['score']:.2f}  {r['item']['title'][:80]}\n      {r['matched_client_features']}")
        return
    results = agent.run(args.client, top_k=args.top_k)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info("готово: %d инсайт(ов) -> %s", len(results), args.out)
    if llm:
        log.info("LLM-вызовов за прогон: %s", llm.USAGE)


if __name__ == "__main__":
    main()
