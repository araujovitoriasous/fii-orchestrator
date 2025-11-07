import pathlib
import sys

import pandas as pd
import pytest

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.fundamentalist_agent import (
    FundamentalScoresConfig,
    generate_fundamental_scores,
)


def _build_sample_data():
    prices = pd.DataFrame(
        {
            "ticker": ["AAA11", "AAA11", "BBB11", "BBB11"],
            "data": [
                "2024-12-30",
                "2025-01-10",
                "2024-12-30",
                "2025-01-10",
            ],
            "pvp": [0.95, 0.90, 1.10, 1.20],
            "vacancia": [0.08, 0.10, 0.07, 0.05],
            "dy_12m": [0.075, 0.080, 0.090, 0.120],
            "vp_cota": [110.0, 110.0, 95.0, 95.0],
            "tipo_gestao": ["Ativa", "Ativa", "Passiva", "Passiva"],
        }
    )

    fundamentals = pd.DataFrame(
        {
            "ticker": ["AAA11", "AAA11", "BBB11", "BBB11"],
            "data_referencia": [
                "2024-09-30",
                "2024-12-31",
                "2024-09-30",
                "2024-12-31",
            ],
            "Lucro_Caixa_Trimestral": [1_000_000, 1_100_000, 800_000, 900_000],
            "Receita_Caixa": [1_800_000, 1_900_000, 1_600_000, 1_700_000],
            "Liquidez_Caixa": [500_000, 520_000, 650_000, 670_000],
            "Taxa_Administracao": [0.012, 0.012, 0.009, 0.009],
        }
    )

    dividends = pd.DataFrame(
        {
            "ticker": [
                "AAA11",
                "AAA11",
                "AAA11",
                "AAA11",
                "BBB11",
                "BBB11",
                "BBB11",
                "BBB11",
            ],
            "data": [
                "2024-04-01",
                "2024-07-01",
                "2024-10-01",
                "2025-01-01",
                "2024-04-01",
                "2024-07-01",
                "2024-10-01",
                "2025-01-01",
            ],
            "dividendo": [0.55, 0.56, 0.57, 0.58, 0.60, 0.62, 0.64, 0.70],
        }
    )

    return prices, fundamentals, dividends


def test_generate_fundamental_scores_produces_expected_ranking():
    prices, fundamentals, dividends = _build_sample_data()

    scores = generate_fundamental_scores(prices, fundamentals, dividends)

    print(
        "\n[Agente Fundamentalista] Avaliação realizada com os seguintes pilares: "
        "P/VP, Vacância, DY, Crescimento do Lucro, Receita Caixa, Liquidez Caixa, "
        "Taxa de Administração e estabilidade/crescimento dos dividendos."
    )
    print(scores[[
        "ticker",
        "pvp_score",
        "vacancia_score",
        "dy_score",
        "lucro_growth_score",
        "receita_caixa_score",
        "liquidez_caixa_score",
        "taxa_administracao_score",
        "dividend_forward_score",
        "score_fundamentalista",
    ]])

    assert list(scores["ticker"]) == ["BBB11", "AAA11"]

    expected_columns = {
        "ticker",
        "data_preco",
        "data_referencia_fundo",
        "pvp_raw",
        "vacancia_raw",
        "dy_12m_raw",
        "vp_cota",
        "tipo_gestao",
        "lucro_caixa_trimestral",
        "receita_caixa",
        "liquidez_caixa",
        "taxa_administracao",
        "pvp_score",
        "vacancia_score",
        "dy_score",
        "lucro_growth_score",
        "receita_caixa_score",
        "liquidez_caixa_score",
        "taxa_administracao_score",
        "dividend_forward_score",
        "score_fundamentalista",
    }
    assert expected_columns.issubset(scores.columns)

    bbb11 = scores.loc[scores["ticker"] == "BBB11"].iloc[0]
    aaa11 = scores.loc[scores["ticker"] == "AAA11"].iloc[0]

    assert 0 <= bbb11["score_fundamentalista"] <= 1
    assert 0 <= aaa11["score_fundamentalista"] <= 1
    assert bbb11["score_fundamentalista"] > aaa11["score_fundamentalista"]
    assert not pd.isna(bbb11["dividend_forward_score"]) or not pd.isna(
        aaa11["dividend_forward_score"]
    )