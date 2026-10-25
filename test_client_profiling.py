"""
pytest-тесты на build_profile() (Чат 1: доработка client_profiling.py).

Запуск:
    pytest test_client_profiling.py -v
"""

import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client_profiling import (  # noqa: E402
    build_profile,
    build_group_peers_index,
    load_clients,
)


def _empty_aux():
    return {}, {}, {}


def test_build_profile_happy_path():
    row = {
        "SYNTHETIC_U": "U_1",
        "REVENUE_COMPANY": "20000000000",
        "MAINOKVED_CODE": "41.20",
        "MAINOKVED_NAME": "Строительство жилых и нежилых зданий",
        "CL_TYPE": "CLIENT",
        "ORG_TYPE": "КОММ_ОРГ",
        "OKFS_NAME": "ЧАСТНАЯ СОБСТВЕННОСТЬ",
        "COMPANY_AGE_YEARS_GRADATION": "8-15",
        "ECO_MARK_LABEL": "Оптимизация",
        "AMOUNT_WORKERS_FNS_GRADATION": "16-100",
        "REGION": "Москва",
        "CITY": "г. Москва",
        "GROUP_ID": "group_1",
        "NEGATIVEMARK": "0",
        "CL_PRODUCT_LOA_MARK": "1",
        "CL_PRODUCT_LEA_MARK": "1",
        "FACTORING_MARK": "0",
        "FTS_FLAG": "0",
    }
    fid_info, pay_in, pay_out = _empty_aux()

    profile = build_profile("U_1", row, fid_info, pay_in, pay_out)

    assert profile.client_id == "U_1"
    assert profile.revenue == 20_000_000_000.0
    assert profile.revenue_bucket == "крупный (15-50 млрд)"
    assert profile.industry_sector == "construction"
    assert "loan" in profile.products and "leasing" in profile.products
    assert "key_rate" in profile.sensitivities
    assert profile.group_id == "group_1"
    assert profile.group_peers == []


def test_build_profile_empty_revenue_does_not_crash(caplog):
    row = {
        "SYNTHETIC_U": "U_2",
        "REVENUE_COMPANY": "",
        "MAINOKVED_CODE": "46.90",
        "MAINOKVED_NAME": "Торговля оптовая неспециализированная",
    }
    fid_info, pay_in, pay_out = _empty_aux()

    with caplog.at_level(logging.WARNING):
        profile = build_profile("U_2", row, fid_info, pay_in, pay_out)

    assert profile.revenue == 0.0
    assert profile.revenue_bucket == "ниже порога"
    assert any("REVENUE_COMPANY" in rec.message for rec in caplog.records)


def test_build_profile_empty_okved_does_not_crash(caplog):
    row = {
        "SYNTHETIC_U": "U_3",
        "REVENUE_COMPANY": "16000000000",
        "MAINOKVED_CODE": "",
        "MAINOKVED_NAME": "",
    }
    fid_info, pay_in, pay_out = _empty_aux()

    with caplog.at_level(logging.WARNING):
        profile = build_profile("U_3", row, fid_info, pay_in, pay_out)

    assert profile.industry_sector == "other"
    assert any("MAINOKVED_CODE" in rec.message for rec in caplog.records)


def test_load_clients_duplicate_and_missing_uid(caplog):
    import csv
    import tempfile

    csv_text = (
        "SYNTHETIC_U;REVENUE_COMPANY;MAINOKVED_CODE\n"
        "U_10;10000000000;46.90\n"
        "U_10;20000000000;41.20\n"
        ";5000000000;41.20\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write(csv_text)
        path = f.name

    try:
        with caplog.at_level(logging.WARNING):
            clients = load_clients(path)
    finally:
        os.unlink(path)

    assert len(clients) == 1
    assert clients["U_10"]["REVENUE_COMPANY"] == "20000000000"
    assert any("дублирующийся" in rec.message for rec in caplog.records)
    assert any("без SYNTHETIC_U" in rec.message for rec in caplog.records)


def test_group_peers_two_siblings_same_group():
    all_clients = {
        "U_A": {"SYNTHETIC_U": "U_A", "GROUP_ID": "group_X"},
        "U_B": {"SYNTHETIC_U": "U_B", "GROUP_ID": "group_X"},
        "U_C": {"SYNTHETIC_U": "U_C", "GROUP_ID": "group_Y"},
    }
    peers_index = build_group_peers_index(all_clients)

    assert sorted(peers_index["U_A"]) == ["U_B"]
    assert sorted(peers_index["U_B"]) == ["U_A"]
    assert peers_index["U_C"] == []

    row_a = {"SYNTHETIC_U": "U_A", "REVENUE_COMPANY": "16000000000", "MAINOKVED_CODE": "41.20"}
    fid_info, pay_in, pay_out = _empty_aux()
    profile_a = build_profile("U_A", row_a, fid_info, pay_in, pay_out, peers_index)
    assert profile_a.group_peers == ["U_B"]


def test_group_peers_no_group_id_is_isolated():
    all_clients = {
        "U_D": {"SYNTHETIC_U": "U_D", "GROUP_ID": ""},
        "U_E": {"SYNTHETIC_U": "U_E", "GROUP_ID": None},
        "U_F": {"SYNTHETIC_U": "U_F", "GROUP_ID": "group_Z"},
    }
    peers_index = build_group_peers_index(all_clients)

    assert peers_index.get("U_D", []) == []
    assert peers_index.get("U_E", []) == []
    assert peers_index.get("U_F", []) == []
