"""
call_llm.py
Вызов РАЗРЕШЁННЫХ генеративных LLM (Приложение 3 ТЗ: Gemma 4 26B A4B,
GLM-5.3-Flash, Qwen3.6 35B A3B, DeepSeek-V4.1-Flash, Qwen3-VL-30B-A3B-Instruct,
Kimi-K2.6) через OpenAI-совместимый API.

Бюджет 500 руб/день ОБЩИЙ на команду -- не гоняй вызовы в цикле без нужды.

НАСТРОЙКА (файл .env рядом с проектом, НЕ коммитить):
    ALFA_LLM_API_KEY=...
    ALFA_LLM_BASE_URL=https://.../v1
    # необязательно: точные id моделей из письма организаторов
    ALFA_MODEL_CHEAP_CLASSIFY=...
    ALFA_MODEL_BULK_RANK=...
    ALFA_MODEL_GENERATION=...
    ALFA_MODEL_SELF_CHECK=...

ПРОВЕРКА КЛЮЧА (один крошечный запрос к дешёвой модели):
    python call_llm.py --ping
Если ответ "model not found" -- поправь id модели в .env (ALFA_MODEL_...).
"""

import json
import logging
import os
import re
import sys
import time
from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

log = logging.getLogger("call_llm")

_DEFAULT_MODELS = {
    "cheap_classify": "gemma-4-26b-a4b",      # классификация, извлечение фактов
    "bulk_rank": "qwen3.6-35b-a3b",           # ранжирование релевантности пачками
    "deep_links": "deepseek-v4.1-flash",      # неочевидные связи
    "generation": "glm-5.3-flash",            # финальная интерпретация (дорогой шаг)
    "self_check": "kimi-k2.6",                # проверка полноты/непротиворечивости
    "vision": "qwen3-vl-30b-a3b-instruct",    # графики/таблицы из PDF
}
# id можно переопределить в .env: ALFA_MODEL_<ЗАДАЧА В ВЕРХНЕМ РЕГИСТРЕ>
MODELS = {k: os.environ.get("ALFA_MODEL_" + k.upper(), v) for k, v in _DEFAULT_MODELS.items()}

# Счётчик для контроля бюджета в рамках одного запуска
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


def _client():
    if OpenAI is None:
        raise RuntimeError("Установи пакет: pip install openai")
    api_key = os.environ.get("ALFA_LLM_API_KEY")
    base_url = os.environ.get("ALFA_LLM_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("Не заданы ALFA_LLM_API_KEY / ALFA_LLM_BASE_URL (см. .env)")
    return OpenAI(api_key=api_key, base_url=base_url, timeout=60)


def extract_json(text: str):
    """Достаёт JSON из ответа модели: убирает ```json-ограждения и текст вокруг."""
    if text is None:
        raise ValueError("пустой ответ модели")
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # ищем первый {...} или [...] с балансом скобок
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = t.find(open_c)
        end = t.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(t[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("в ответе модели нет валидного JSON: %r" % t[:200])


def call_model(task: str, system_prompt: str, user_prompt: str,
               temperature: float = 0.2, response_json: bool = True,
               retries: int = 2) -> str:
    """Сырой текст ответа. Повторяет при сетевых ошибках; если провайдер не
    поддерживает response_format -- повторяет без него."""
    if task not in MODELS:
        raise ValueError(f"Неизвестная задача '{task}'. Доступно: {list(MODELS)}")
    client = _client()
    use_json_mode = response_json
    last_err = None
    for attempt in range(retries + 1):
        kwargs = dict(
            model=MODELS[task], temperature=temperature,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
        )
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = client.chat.completions.create(**kwargs)
            USAGE["calls"] += 1
            if getattr(resp, "usage", None):
                USAGE["prompt_tokens"] += resp.usage.prompt_tokens or 0
                USAGE["completion_tokens"] += resp.usage.completion_tokens or 0
            log.info("LLM %s (%s): всего вызовов=%d, токенов вход/выход=%d/%d", task,
                     MODELS[task], USAGE["calls"], USAGE["prompt_tokens"],
                     USAGE["completion_tokens"])
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if use_json_mode and ("response_format" in msg or "json_object" in msg):
                use_json_mode = False  # провайдер не умеет json-режим
                continue
            if "model" in msg and ("not found" in msg or "does not exist" in msg):
                raise RuntimeError(f"Модель '{MODELS[task]}' не найдена у провайдера -- "
                                   f"поправь id в .env (ALFA_MODEL_{task.upper()})") from e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM недоступна после {retries + 1} попыток: {last_err}")


# ---------------------------------------------------------------------------
# Обёртки под шаги пайплайна (промпты -- из agent_pipeline.md)
# ---------------------------------------------------------------------------
def rank_news_for_client(client_profile: dict, news_batch: list) -> list:
    """Шаг 3: оценки релевантности пачки новостей. Возвращает список
    {event_id, score, matched_client_features}."""
    system = (
        "Для каждой новости оцени релевантность клиенту от 0 до 1 и укажи, какие "
        "признаки клиента сработали (industry_sector, sensitivities, fx_exposed, "
        "revenue_bucket, region, products). Не пересказывай новость -- только "
        "оценка и причина. Верни строго JSON-объект вида "
        '{"ranking": [{"event_id": "...", "score": 0.0, "matched_client_features": ["..."]}]}.'
    )
    user = json.dumps({"client": client_profile, "news": news_batch}, ensure_ascii=False)
    data = extract_json(call_model("bulk_rank", system, user))
    if isinstance(data, dict):
        data = data.get("ranking") or data.get("results") or data.get("items") or []
    return data


def generate_interpretation(client_profile: dict, event_facts: dict,
                            bank_analytics_snippets: list, feedback: list = None) -> dict:
    """Шаг 4: полная интерпретация одного события (дорогая модель).
    feedback -- замечания self-check при повторной генерации."""
    system = (
        "Ты формируешь персонализированный бизнес-инсайт для представителя крупного "
        "бизнеса в интерфейсе Альфа-Бизнеса. НЕ пересказывай новость -- объясни её "
        "значение именно для этой компании. Опирайся ТОЛЬКО на переданные факты "
        "события, профиль клиента и выдержки аналитики банка; не выдумывай цифры. "
        "Ответь на 6 вопросов и верни строго JSON с ключами: "
        "what_happened (строка), why_relevant_to_client (строка), "
        "possible_consequences (массив {type: 'risk'|'opportunity', text}), "
        "possible_actions (массив строк), bank_view (строка; если аналитики банка "
        "по теме нет -- так и напиши), basis_for_conclusion (массив строк вида "
        "'client_data: ...', 'news: ...', 'alfa_analytics: <документ, стр.>', "
        "'model_inference: ...')."
    )
    payload = {"client": client_profile, "event_facts": event_facts,
               "bank_analytics": bank_analytics_snippets}
    if feedback:
        payload["fix_these_issues_from_previous_attempt"] = feedback
    return extract_json(call_model("generation", system,
                                   json.dumps(payload, ensure_ascii=False)))


def self_check(interpretation: dict, event_facts: dict = None,
               bank_analytics_snippets: list = None) -> dict:
    """Шаг 5: проверка полноты/непротиворечивости/обоснованности."""
    system = (
        "Проверь интерпретацию на: (а) полноту -- есть ли ответы на все 6 вопросов, "
        "(б) непротиворечивость -- не противоречит ли исходным фактам события и "
        "выдержкам аналитики, нет ли выдуманных цифр, (в) обоснованность -- у каждого "
        'вывода есть источник в basis_for_conclusion. Верни строго JSON '
        '{"ok": true|false, "issues": ["..."], "action": '
        '"re-retrieve|re-rank|re-generate|none"}.'
    )
    user = json.dumps({"interpretation": interpretation, "event_facts": event_facts,
                       "bank_analytics": bank_analytics_snippets}, ensure_ascii=False)
    return extract_json(call_model("self_check", system, user))


def ping(all_models: bool = False) -> None:
    """Крошечный запрос для проверки ключа/адреса/id моделей."""
    tasks = list(MODELS) if all_models else ["cheap_classify"]
    for task in tasks:
        if task == "vision":
            continue
        try:
            out = call_model(task, 'Верни JSON {"pong": true}', "ping", response_json=True,
                             retries=0)
            print(f"[OK ] {task:<15} {MODELS[task]:<28} -> {out.strip()[:60]!r}")
        except Exception as e:
            print(f"[ERR] {task:<15} {MODELS[task]:<28} -> {e}")
    print("Использовано вызовов:", USAGE["calls"], "токенов:",
          USAGE["prompt_tokens"] + USAGE["completion_tokens"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if "--ping" in sys.argv:
        ping(all_models="--all" in sys.argv)
    elif "--demo" in sys.argv:
        with open("client_profiles.json", encoding="utf-8") as f:
            client = json.load(f)[0]
        news = [{"event_id": "test_1", "title": "ЦБ РФ сохранил ключевую ставку на уровне 14,0%",
                 "published_at": "2026-09-11",
                 "summary_facts": ["Ставка сохранена на 14,0% второе заседание подряд"]}]
        print(rank_news_for_client(client, news))
    else:
        print("Использование: python call_llm.py --ping   (или --ping --all, --demo)")
