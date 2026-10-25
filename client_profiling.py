"""
client_profiling.py  (Чат 1: профиль компании -- K4 архитектуры)

!!! ВАЖНО (найдено 25.10.2026): в репозитории этот файл был случайно
перезаписан содержимым test_client_profiling.py (тот же текст, импорт
"from client_profiling import build_profile..." прямо из себя же).
Поэтому ничего не импортировалось и ничего не запускалось: build_profile,
build_group_peers_index, load_clients физически отсутствовали. Этот файл --
восстановленная реализация, подогнанная так, чтобы:
  1) test_client_profiling.py проходил без изменений;
  2) для уже закоммиченных 115 клиентов (client_profiles.json) все поля,
     кроме НОВОГО group_peers, совпадали 1-в-1 с тем, что там уже лежит
     (проверено сравнением построчно -- см. Git-коммит с этим файлом).

ЧТО ДЕЛАЕТ:
    SYNTHETIC_U -> профиль компании (ClientProfile): отрасль, выручка и её
    категория, продукты банка, "чувствительности" к макрофакторам,
    валютная экспозиция, агрегаты по платежам (входящие/исходящие,
    контрагенты), связанные физлица, "соседи" по группе (GROUP_ID).

ОТКУДА ПРАВИЛА (реконструированы по данным, не придуманы с нуля):
  - industry_sector: по первым 2 цифрам MAINOKVED_CODE (секция ОКВЭД2).
    Коды, которых нет среди уже обработанных 115 клиентов (агро, нефть
    и газ, металлургия/добыча) -- это ДОПУЩЕНИЕ команды по стандартной
    структуре ОКВЭД2, со вскрытия реальных данных не проверялись. Пометь
    при ревью.
  - revenue_bucket: граница ровно по условию кейса (выручка >= 15 млрд ₽
    = "крупный бизнес"); >= 50 млрд -- "mega", иначе "ниже порога".
  - products: по бинарным флагам CL_PRODUCT_*_MARK и FACTORING_MARK.
  - sensitivities: products (loan/leasing/deposit/factoring) -> key_rate;
    guarantee -> госконтракты/крупные тендеры; FTS_FLAG==1 -> валютный
    курс; сектор wholesale_trade/trade/consumption -> инфляция и спрос;
    сектор construction_realestate -> ипотека и ставка.
  - fx_exposed: FTS_FLAG == 1 (компания отмечена во внешнеторговых данных).
  - inbound/outbound по платежам: сумма PAY_AMT и PAY_CNT, где клиент --
    BENEFICIARY (входящие) или PAYER (исходящие) соответственно.
  - counterparty_count: число уникальных контрагентов по обоим
    направлениям платежей (self-платежи клиента самому себе в данных есть
    и в оригинальном выводе учтены как "контрагент = сам клиент" --
    сохранено для совместимости с уже закоммиченным client_profiles.json).
  - related_persons_count: число строк в df_Fid_Uid_link_hack.csv с этим
    SYNTHETIC_U.
  - group_peers: другие SYNTHETIC_U с тем же непустым GROUP_ID (во ВСЕЙ
    базе юрлиц, не только среди крупного бизнеса). РАНЬШЕ не попадало в
    client_profiles.json -- теперь попадает (нужно agent.py: plan() решает
    need_graph по этому полю). Наружу в LLM/UI уходит только КОЛИЧЕСТВО
    (ranking.sanitize_profile), сами ID -- нет (приватность, п.6 ТЗ).

ЗАПУСК (пересобрать client_profiles.json с нуля):
    python client_profiling.py --out client_profiles.json

    Необязательные флаги (значения по умолчанию -- как в кейсе):
    --data data/df_Uid_data_hack.csv
    --payments data/df_Uid_payment_hack.csv
    --fid data/df_Fid_Uid_link_hack.csv
    --min-revenue 15000000000

ТЕСТЫ:
    pytest test_client_profiling.py -v
"""

import argparse
import csv
import dataclasses
import json
import logging
import os
from typing import Optional

log = logging.getLogger("client_profiling")

REVENUE_THRESHOLD = 15_000_000_000.0   # "крупный бизнес" по условию кейса
REVENUE_MEGA = 50_000_000_000.0

# Секция ОКВЭД2 (первые 2 знака кода) -> наш internal industry_sector.
# Значения, подтверждённые по 115 уже обработанным клиентам, помечены "data";
# остальные -- "assumption" (стандартная структура ОКВЭД2, не проверено по
# реальным строкам этого набора).
OKVED_SECTION_TO_SECTOR = {
    # data: construction
    "41": "construction", "42": "construction", "43": "construction",
    # data: construction_realestate
    "68": "construction_realestate",
    # data: wholesale_trade / trade / consumption
    "46": "wholesale_trade",
    "45": "trade",
    "47": "consumption", "56": "consumption",
    # data: logistics
    "49": "logistics", "50": "logistics", "51": "logistics", "52": "logistics", "53": "logistics",
    # data: finance
    "64": "finance", "65": "finance", "66": "finance",
    # data: it
    "58": "it", "59": "it", "60": "it", "61": "it", "62": "it", "63": "it",
    # assumption: agro (растениеводство/животноводство/переработка)
    "01": "agro", "02": "agro", "03": "agro", "10": "agro", "11": "agro",
    # assumption: нефть и газ (добыча, кокс/нефтепродукты)
    "06": "oil_gas", "19": "oil_gas",
    # assumption: металлургия и добыча
    "05": "metals_mining", "07": "metals_mining", "08": "metals_mining",
    "09": "metals_mining", "24": "metals_mining", "25": "metals_mining",
}

# Флаг в CSV -> название продукта банка (как в client_profiles.json)
PRODUCT_FLAG_MAP = {
    "CL_PRODUCT_RKO_MARK": "rko",
    "CL_PRODUCT_GAR_MARK": "guarantee",
    "CL_PRODUCT_LOA_MARK": "loan",
    "CL_PRODUCT_LEA_MARK": "leasing",
    "CL_PRODUCT_DEP_MARK": "deposit",
    "CL_PRODUCT_ZPP_MARK": "payroll",
    "FACTORING_MARK": "factoring",
}

# Продукты, из-за которых компания чувствительна к ключевой ставке
KEY_RATE_PRODUCTS = {"loan", "leasing", "deposit", "factoring"}

# Базовая (не продуктовая, не валютная) чувствительность по отрасли
SECTOR_BASE_SENSITIVITY = {
    "wholesale_trade": "инфляция_и_потребительский_спрос",
    "trade": "инфляция_и_потребительский_спрос",
    "consumption": "инфляция_и_потребительский_спрос",
    "construction_realestate": "ипотека_и_ставка",
}


# ---------------------------------------------------------------------------
# Разбор отдельных полей (каждая функция -- одна "честная" ошибка из
# Приложения D: пустые/нечисловые значения не должны ронять весь прогон)
# ---------------------------------------------------------------------------
def parse_revenue(raw, client_id: str = "") -> float:
    try:
        val = float(str(raw).strip().replace(",", "."))
        if val < 0:
            raise ValueError("отрицательная выручка")
        return val
    except (TypeError, ValueError):
        log.warning("клиент %s: пустое/некорректное REVENUE_COMPANY=%r -> 0.0", client_id, raw)
        return 0.0


def revenue_bucket(revenue: float) -> str:
    if revenue >= REVENUE_MEGA:
        return "mega (>=50 млрд)"
    if revenue >= REVENUE_THRESHOLD:
        return "крупный (15-50 млрд)"
    return "ниже порога"


def industry_sector_from_okved(code: Optional[str], client_id: str = "") -> str:
    code = (code or "").strip()
    if not code:
        log.warning("клиент %s: пустой MAINOKVED_CODE -> industry_sector='other'", client_id)
        return "other"
    section = code.split(".")[0]
    return OKVED_SECTION_TO_SECTOR.get(section, "other")


def parse_flag(raw) -> bool:
    try:
        return float(str(raw).strip().replace(",", ".")) == 1.0
    except (TypeError, ValueError):
        return False


def extract_products(row: dict) -> list:
    return sorted(name for col, name in PRODUCT_FLAG_MAP.items() if parse_flag(row.get(col)))


def compute_sensitivities(industry_sector: str, products: list, fx_exposed: bool) -> list:
    sens = []
    if KEY_RATE_PRODUCTS & set(products):
        sens.append("key_rate")
    if "guarantee" in products:
        sens.append("госконтракты/крупные тендеры")
    if fx_exposed:
        sens.append("валютный_курс_и_внешнеторговые_ограничения")
    base = SECTOR_BASE_SENSITIVITY.get(industry_sector)
    if base:
        sens.append(base)
    return sens


# ---------------------------------------------------------------------------
# Загрузка CSV (все три набора данных Приложения 1, разделитель ';')
# ---------------------------------------------------------------------------
def load_clients(path: str) -> dict:
    """SYNTHETIC_U -> строка CSV (dict). Дубликаты -- последняя строка
    побеждает (с warning). Строки без SYNTHETIC_U пропускаются (с warning)."""
    clients: dict = {}
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            cid = (row.get("SYNTHETIC_U") or "").strip()
            if not cid:
                log.warning("строка без SYNTHETIC_U пропущена: %r", row)
                continue
            if cid in clients:
                log.warning("дублирующийся SYNTHETIC_U=%s -- используется последняя строка", cid)
            clients[cid] = row
    return clients


def load_fid_links(path: str) -> dict:
    """SYNTHETIC_U -> список SYNTHETIC_F (связанные физлица)."""
    out: dict = {}
    if not os.path.exists(path):
        log.warning("файл физлиц не найден: %s -- related_persons_count будет 0", path)
        return out
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            cid = (row.get("SYNTHETIC_U") or "").strip()
            fid = (row.get("SYNTHETIC_F") or "").strip()
            if not cid or not fid:
                continue
            out.setdefault(cid, []).append(fid)
    return out


def load_payment_aggregates(path: str) -> tuple:
    """Возвращает (pay_in, pay_out): SYNTHETIC_U -> {volume, count, counterparties}.
    pay_in -- клиент как BENEFICIARY (получатель), pay_out -- как PAYER (плательщик).
    Файл большой (~1.4 млн строк) -- читаем потоково, без pandas."""
    pay_in: dict = {}
    pay_out: dict = {}
    if not os.path.exists(path):
        log.warning("файл платежей не найден: %s -- платёжные агрегаты будут нулевыми", path)
        return pay_in, pay_out

    def _bump(d: dict, cid: str, other: str, amt: float, cnt: int) -> None:
        agg = d.setdefault(cid, {"volume": 0.0, "count": 0, "counterparties": set()})
        agg["volume"] += amt
        agg["count"] += cnt
        agg["counterparties"].add(other)

    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            beneficiary = (row.get("BENEFICIARY") or "").strip()
            payer = (row.get("PAYER") or "").strip()
            try:
                amt = float((row.get("PAY_AMT") or "0").replace(",", "."))
            except ValueError:
                amt = 0.0
            try:
                cnt = int(float(row.get("PAY_CNT") or 0))
            except ValueError:
                cnt = 0
            if beneficiary:
                _bump(pay_in, beneficiary, payer, amt, cnt)
            if payer:
                _bump(pay_out, payer, beneficiary, amt, cnt)
    return pay_in, pay_out


def build_group_peers_index(all_clients: dict) -> dict:
    """SYNTHETIC_U -> список ДРУГИХ SYNTHETIC_U с тем же непустым GROUP_ID.
    Компании без GROUP_ID (пусто/None) изолированы -- пустой список, и не
    попадают в чужие peers."""
    groups: dict = {}
    for cid, row in all_clients.items():
        gid = (row.get("GROUP_ID") or "").strip() if isinstance(row, dict) else ""
        if not gid or str(gid).lower() == "none":
            continue
        groups.setdefault(gid, []).append(cid)

    peers_index: dict = {cid: [] for cid in all_clients}
    for members in groups.values():
        if len(members) < 2:
            continue
        for cid in members:
            peers_index[cid] = sorted(m for m in members if m != cid)
    return peers_index


# ---------------------------------------------------------------------------
# Профиль
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class ClientProfile:
    client_id: str
    cl_type: str
    org_type: str
    ownership_form: str
    company_age_bucket: str
    lifecycle_stage: str
    okved_code: str
    okved_name: str
    industry_sector: str
    revenue: float
    revenue_bucket: str
    employees_bucket: str
    region: str
    city: str
    group_id: str
    has_negative_mark: bool
    products: list
    sensitivities: list
    fx_exposed: bool
    fts_country: str
    inbound_payment_volume: float
    inbound_payment_count: int
    outbound_payment_volume: float
    outbound_payment_count: int
    counterparty_count: int
    related_persons_count: int
    group_peers: list = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def build_profile(client_id: str, row: dict, fid_info: dict, pay_in: dict, pay_out: dict,
                  peers_index: Optional[dict] = None) -> ClientProfile:
    """Строка CSV юрлица -> ClientProfile. fid_info/pay_in/pay_out -- индексы,
    посчитанные заранее (load_fid_links / load_payment_aggregates), а не
    пересчитываются здесь на каждого клиента."""
    revenue = parse_revenue(row.get("REVENUE_COMPANY"), client_id)
    industry_sector = industry_sector_from_okved(row.get("MAINOKVED_CODE"), client_id)
    products = extract_products(row)
    fx_exposed = parse_flag(row.get("FTS_FLAG"))
    sensitivities = compute_sensitivities(industry_sector, products, fx_exposed)

    cp_in = pay_in.get(client_id, {})
    cp_out = pay_out.get(client_id, {})
    counterparties = set(cp_in.get("counterparties", ())) | set(cp_out.get("counterparties", ()))

    group_peers = (peers_index or {}).get(client_id, [])

    return ClientProfile(
        client_id=client_id,
        cl_type=row.get("CL_TYPE") or "",
        org_type=row.get("ORG_TYPE") or "",
        ownership_form=row.get("OKFS_NAME") or "",
        company_age_bucket=row.get("COMPANY_AGE_YEARS_GRADATION") or "",
        lifecycle_stage=row.get("ECO_MARK_LABEL") or "",
        okved_code=row.get("MAINOKVED_CODE") or "",
        okved_name=row.get("MAINOKVED_NAME") or "",
        industry_sector=industry_sector,
        revenue=revenue,
        revenue_bucket=revenue_bucket(revenue),
        employees_bucket=row.get("AMOUNT_WORKERS_FNS_GRADATION") or "",
        region=row.get("REGION") or "",
        city=row.get("CITY") or "",
        group_id=row.get("GROUP_ID") or "",
        has_negative_mark=parse_flag(row.get("NEGATIVEMARK")),
        products=products,
        sensitivities=sensitivities,
        fx_exposed=fx_exposed,
        fts_country=row.get("FTS_COUNTRY_NAME") or "",
        inbound_payment_volume=round(float(cp_in.get("volume", 0.0)), 2),
        inbound_payment_count=int(cp_in.get("count", 0)),
        outbound_payment_volume=round(float(cp_out.get("volume", 0.0)), 2),
        outbound_payment_count=int(cp_out.get("count", 0)),
        counterparty_count=len(counterparties),
        related_persons_count=len(fid_info.get(client_id, [])),
        group_peers=group_peers,
    )


def load_clients_full(data_path: str) -> dict:
    """Публичная обёртка для скриптов, которым нужны все строки (например,
    агентный цикл при будущем расширении на весь портфель, не только
    client_profiles.json)."""
    return load_clients(data_path)


# ---------------------------------------------------------------------------
# CLI: пересборка client_profiles.json
# ---------------------------------------------------------------------------
def build_all_profiles(data_path: str, payments_path: str, fid_path: str,
                       min_revenue: float = REVENUE_THRESHOLD) -> list:
    all_clients = load_clients(data_path)
    log.info("юрлиц в базе: %d", len(all_clients))
    fid_info = load_fid_links(fid_path)
    pay_in, pay_out = load_payment_aggregates(payments_path)
    peers_index = build_group_peers_index(all_clients)

    profiles = []
    for cid, row in all_clients.items():
        revenue = parse_revenue(row.get("REVENUE_COMPANY"), cid)
        if revenue < min_revenue:
            continue
        profiles.append(build_profile(cid, row, fid_info, pay_in, pay_out, peers_index))
    profiles.sort(key=lambda p: p.revenue, reverse=True)
    log.info("профилей с выручкой >= %.0f: %d", min_revenue, len(profiles))
    return profiles


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=os.path.join("data", "df_Uid_data_hack.csv"))
    ap.add_argument("--payments", default=os.path.join("data", "df_Uid_payment_hack.csv"))
    ap.add_argument("--fid", default=os.path.join("data", "df_Fid_Uid_link_hack.csv"))
    ap.add_argument("--out", default="client_profiles.json")
    ap.add_argument("--min-revenue", type=float, default=REVENUE_THRESHOLD)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    profiles = build_all_profiles(args.data, args.payments, args.fid, args.min_revenue)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump([p.to_dict() for p in profiles], f, ensure_ascii=False, indent=2)
    log.info("готово: %d профилей -> %s", len(profiles), args.out)


if __name__ == "__main__":
    main()
