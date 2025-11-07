"""
Agente Fundamentalista responsável por calcular o score de fundamentos dos FIIs.

O módulo expõe a função `generate_fundamental_scores` que recebe DataFrames já
carregados (por exemplo, a partir de `prices.parquet` e
`fundamentals_trimestral.parquet`) e devolve um DataFrame com as métricas
ajustadas, normalizadas e o score final no intervalo [0, 1].

Exemplo de uso:

```python
import pandas as pd
from src.models.fundamentalist_agent import generate_fundamental_scores

prices = pd.read_parquet("data/01_raw/prices.parquet")
fundamentals = pd.read_parquet("data/02_processed/fundamentals/fundamentals_trimestral.parquet")

scores = generate_fundamental_scores(prices, fundamentals)
print(scores.head())
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


REQUIRED_PRICE_COLUMNS: frozenset[str] = frozenset(
    {"ticker", "data", "pvp", "vacancia", "dy_12m", "vp_cota", "tipo_gestao"}
)
REQUIRED_FUND_COLUMNS: frozenset[str] = frozenset(
    {
        "ticker",
        "data_referencia",
        "Lucro_Caixa_Trimestral",
        "Receita_Caixa",
        "Liquidez_Caixa",
        "Taxa_Administracao",
    }
)
REQUIRED_DIVIDENDS_COLUMNS: frozenset[str] = frozenset(
    {"ticker", "data", "dividendo"}
)

@dataclass(frozen=True)
class FundamentalScoresConfig:
    """
    Parâmetros de configuração do agente fundamentalista.

    Attributes:
        weight_pvp: Peso atribuído ao P/VP ajustado e normalizado.
        weight_vacancia: Peso atribuído à Vacância ajustada e normalizada.
        weight_dy: Peso atribuído ao Dividend Yield (12m) normalizado.
        weight_lucro_growth: Peso atribuído ao crescimento do Lucro Caixa.
        weight_receita_caixa: Peso atribuído à Receita Caixa normalizada.
        weight_liquidez_caixa: Peso atribuído à Liquidez Caixa normalizada.
        weight_taxa_administracao: Peso atribuído à Taxa de Administração ajustada.
        weight_dividend_forward: Peso atribuído ao indicador de estabilidade/crescimento de dividendos.

    A configuração padrão assume pesos iguais para manter transparência na fase inicial.
    """
    weight_pvp: float = 0.25
    weight_vacancia: float = 0.25
    weight_dy: float = 0.25
    weight_lucro_growth: float = 0.25
    weight_receita_caixa: float = 0.25
    weight_liquidez_caixa: float = 0.25
    weight_taxa_administracao: float = 0.25
    weight_dividend_forward: float = 0.25

    @property
    def weights(self) -> Mapping[str, float]:
        return {
            "pvp_score": self.weight_pvp,
            "vacancia_score": self.weight_vacancia,
            "dy_score": self.weight_dy,
            "lucro_growth_score": self.weight_lucro_growth,
            "receita_caixa_score": self.weight_receita_caixa,
            "liquidez_caixa_score": self.weight_liquidez_caixa,
            "taxa_administracao_score": self.weight_taxa_administracao,
            "dividend_forward_score": self.weight_dividend_forward,
        }


def _ensure_datetime(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Garante que a coluna especificada esteja em formato datetime.

    Mantemos esse ajuste centralizado para evitar erros silenciosos em merges e
    ordenações cronológicas.
    """
    series = df[column]
    if not pd.api.types.is_datetime64_any_dtype(series):
        series = pd.to_datetime(series, errors="coerce")
    return series


def _latest_per_ticker(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Seleciona o registro mais recente de cada ticker com base na coluna de data.

    Fazemos isso para trabalhar sempre com o snapshot corrente disponível dos dados.
    """
    if df.empty:
        return df.copy()
    series = _ensure_datetime(df, date_col)
    idx = series.groupby(df["ticker"]).idxmax()
    return df.loc[idx].copy()


def _min_max_normalize(values: pd.Series) -> pd.Series:
    """
    Normaliza valores para o intervalo [0, 1] usando min-max.

    Essa padronização coloca todas as métricas na mesma escala antes do cálculo do score.
    """
    valid = values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=values.index)
    min_val = valid.min()
    max_val = valid.max()
    if np.isclose(max_val, min_val, equal_nan=True):
        return pd.Series(1.0, index=values.index)
    normalized = (values - min_val) / (max_val - min_val)
    return normalized.clip(0.0, 1.0)


def _adjust_direction(latest_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Ajusta as métricas de preço/vacância para que valores altos signifiquem bons sinais.

    Fazemos transformações direcionais aqui para que a normalização posterior seja coerente.
    """
    adjusted = latest_prices.copy()
    # P/VP baixo é melhor → transformação linear suave (valores próximos mantidos)
    adjusted["pvp_adjusted"] = 1.0 + (1.0 - adjusted["pvp"])

    # Vacância baixa é melhor → complemento em 1
    adjusted["vacancia_adjusted"] = 1 - adjusted["vacancia"]

    # Garantir limites plausíveis
    adjusted["vacancia_adjusted"] = adjusted["vacancia_adjusted"].clip(0.0, 1.0)

    # DY já tem direção correta
    adjusted["dy_value"] = adjusted["dy_12m"]
    return adjusted


def _compute_latest_lucro_growth(
    fundamentals: pd.DataFrame, 
    latest_funds: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Calcula o crescimento percentual mais recente do lucro caixa por ticker.

    Usamos essa função para captar a tendência de lucro recente preservando robustez
    via clipping e validações.
    """
    if fundamentals.empty:
        raise ValueError("DataFrame de fundamentos está vazio.")
    if latest_funds.empty:
        raise ValueError("Não há registros mais recentes de fundamentos para calcular crescimento.")

    fundamentals = fundamentals.copy()
    fundamentals["data_referencia"] = _ensure_datetime(fundamentals, "data_referencia")
    fundamentals.sort_values(["ticker", "data_referencia"], inplace=True)

    fundamentals["lucro_growth"] = fundamentals.groupby("ticker")["Lucro_Caixa_Trimestral"].pct_change()
    fundamentals["lucro_growth"].replace([np.inf, -np.inf], np.nan, inplace=True)
    fundamentals["lucro_growth"] = fundamentals["lucro_growth"].clip(-0.5, 1.0)

    fundamentals.set_index(["ticker", "data_referencia"], inplace=True)

    latest_funds = latest_funds.copy()
    latest_funds["data_referencia"] = _ensure_datetime(latest_funds, "data_referencia")
    latest_funds.set_index(["ticker", "data_referencia"], inplace=True)

    joined = latest_funds.join(
        fundamentals[["Lucro_Caixa_Trimestral", "lucro_growth"]],
        how="left",
        rsuffix="_historico",
    )
    joined.reset_index(inplace=True)
    return joined[
        [
            "ticker",
            "data_referencia",
            "Lucro_Caixa_Trimestral",
            "Receita_Caixa",
            "Liquidez_Caixa",
            "Taxa_Administracao",
            "lucro_growth",
        ]
    ]

def _compute_dividend_forward_metrics(
    dividends: pd.DataFrame | None,
    reference_dates: Mapping[str, pd.Timestamp],
) -> pd.DataFrame:
    """
    Calcula métricas de dividendos futuros (forward):
      - Dividend Yield Forward
      - Payout Ratio Forward
      - Crescimento do Dividendo Esperado
    """

    if dividends is None or dividends.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "dividend_mean_12m",
                "dividend_cv_12m",
                "dividend_growth_12m",
            ]
        )

    missing_div_cols = REQUIRED_DIVIDENDS_COLUMNS.difference(dividends.columns)
    if missing_div_cols:
        raise ValueError(
            "DataFrame de dividendos não possui as colunas obrigatórias: "
            f"{sorted(missing_div_cols)}"
        )

    dividends = dividends.copy()
    dividends["data"] = _ensure_datetime(dividends, "data")

    records: list[dict[str, float]] = []
    for ticker, group in dividends.groupby("ticker"):
        ref_date = reference_dates.get(ticker)
        if pd.isna(ref_date):
            continue
        window_start = ref_date - pd.Timedelta(days=365)
        recent = group[(group["data"] <= ref_date) & (group["data"] > window_start)]
        if recent.empty:
            continue
        recent = recent.sort_values("data")
        mean_value = float(recent["dividendo"].mean())
        std_value = float(recent["dividendo"].std(ddof=0)) if len(recent) > 1 else 0.0
        cv_value = std_value / mean_value if mean_value > 0 else np.nan

        growth_value = np.nan
        if len(recent) >= 2:
            first = float(recent.iloc[0]["dividendo"])
            last = float(recent.iloc[-1]["dividendo"])
            if first > 0:
                growth_value = (last / first) - 1.0

        records.append(
            {
                "ticker": ticker,
                "dividend_mean_12m": mean_value,
                "dividend_cv_12m": cv_value,
                "dividend_growth_12m": growth_value,
            }
        )

    return pd.DataFrame(records)


def _fill_weights(weights: Mapping[str, float]) -> Mapping[str, float]:
    """
    Normaliza os pesos para que somem 1, assegurando média ponderada válida.

    Centralizamos essa lógica para evitar configuração inconsistente de pesos.
    """
    if not weights:
        raise ValueError("Nenhum peso válido foi fornecido.")
    total = sum(weights.values())
    if np.isclose(total, 0.0):
        raise ValueError("Os pesos fornecidos são inválidos (soma zero).")
    return {column: weight / total for column, weight in weights.items()}


def generate_fundamental_scores(
    data_prices: pd.DataFrame,
    data_fundamentals: pd.DataFrame,
    data_dividends: pd.DataFrame | None = None,
    *,
    config: FundamentalScoresConfig | None = None,
) -> pd.DataFrame:
    """
    Calcula o score fundamentalista a partir dos dados de preços e fundamentos.

    Objetivo: disponibilizar um resumo numérico padronizado (0 a 1) que sintetiza a
    saúde fundamentalista de cada FII para ser consumido pelo orquestrador.

    Args:
        data_prices: DataFrame com colunas `ticker`, `data`, `pvp`, `vacancia`, `dy_12m`.
        data_fundamentals: DataFrame com colunas `ticker`, `data_referencia`,
            `Lucro_Caixa_Trimestral`, `Receita_Caixa`, `Liquidez_Caixa`, `Taxa_Administracao`.
        data_dividends: DataFrame opcional com histórico de proventos para cada ticker.
        config: Configuração opcional com os pesos das métricas.

    Returns:
        DataFrame com `ticker`, colunas normalizadas das métricas e `score_fundamentalista`.

    Levanta:
        ValueError: se colunas obrigatórias estiverem ausentes ou se os pesos forem inválidos.
    """

    config = config or FundamentalScoresConfig()

    missing_prices = REQUIRED_PRICE_COLUMNS.difference(data_prices.columns)
    if missing_prices:
        raise ValueError(
            f"DataFrame de preços não possui as colunas obrigatórias: {sorted(missing_prices)}"
        )

    missing_funds = REQUIRED_FUND_COLUMNS.difference(data_fundamentals.columns)
    if missing_funds:
        raise ValueError(
            "DataFrame de fundamentos não possui as colunas obrigatórias: "
            f"{sorted(missing_funds)}"
        )

    latest_prices = _latest_per_ticker(data_prices, "data")
    latest_funds = _latest_per_ticker(data_fundamentals, "data_referencia")

    if latest_prices.empty:
        raise ValueError("DataFrame de preços está vazio após filtrar últimos registros.")
    if latest_funds.empty:
        raise ValueError(
            "DataFrame de fundamentos está vazio após filtrar últimos registros."
        )

    latest_prices = _adjust_direction(latest_prices)
    latest_growth = _compute_latest_lucro_growth(data_fundamentals, latest_funds)

    merged = latest_prices.merge(
        latest_growth,
        on="ticker",
        how="left",
        suffixes=("_preco", "_fundo"),
    )

    metrics = pd.DataFrame(index=merged.index)
    metrics["ticker"] = merged["ticker"]
    metrics["data_preco"] = _ensure_datetime(merged, "data")
    metrics["data_referencia_fundo"] = _ensure_datetime(merged, "data_referencia")
    metrics["pvp_raw"] = merged["pvp"]
    metrics["vacancia_raw"] = merged["vacancia"]
    metrics["dy_12m_raw"] = merged["dy_12m"]
    metrics["vp_cota"] = merged["vp_cota"]
    metrics["tipo_gestao"] = merged["tipo_gestao"]
    metrics["lucro_caixa_trimestral"] = merged["Lucro_Caixa_Trimestral"]
    metrics["receita_caixa"] = merged["Receita_Caixa"]
    metrics["liquidez_caixa"] = merged["Liquidez_Caixa"]
    metrics["taxa_administracao"] = merged["Taxa_Administracao"]
    metrics["pvp_score"] = _min_max_normalize(merged["pvp_adjusted"])
    metrics["vacancia_score"] = _min_max_normalize(merged["vacancia_adjusted"])
    metrics["dy_score"] = _min_max_normalize(merged["dy_value"])
    metrics["lucro_growth_score"] = _min_max_normalize(merged["lucro_growth"])
    metrics["receita_caixa_score"] = _min_max_normalize(metrics["receita_caixa"])
    metrics["liquidez_caixa_score"] = _min_max_normalize(metrics["liquidez_caixa"])
    taxa_norm = _min_max_normalize(metrics["taxa_administracao"])
    metrics["taxa_administracao_score"] = 1 - taxa_norm

    reference_dates = metrics.set_index("ticker")["data_preco"].to_dict()
    dividend_metrics = _compute_dividend_forward_metrics(data_dividends, reference_dates)
    if not dividend_metrics.empty:
        metrics = metrics.merge(dividend_metrics, on="ticker", how="left")
    else:
        metrics["dividend_mean_12m"] = np.nan
        metrics["dividend_cv_12m"] = np.nan
        metrics["dividend_growth_12m"] = np.nan

    cv_norm = _min_max_normalize(metrics["dividend_cv_12m"])
    metrics["dividend_stability_score"] = 1 - cv_norm
    growth_clipped = metrics["dividend_growth_12m"].clip(-0.5, 1.0)
    metrics["dividend_growth_score"] = _min_max_normalize(growth_clipped)
    metrics["dividend_forward_score"] = (
        metrics[["dividend_stability_score", "dividend_growth_score"]]
        .mean(axis=1, skipna=True)
        .clip(0.0, 1.0)
    )

    score_columns = [column for column in config.weights if column in metrics.columns]
    if not score_columns:
        raise ValueError(
            "Nenhuma métrica de score disponível para calcular o score fundamentalista."
        )
    weight_map = _fill_weights({column: config.weights[column] for column in score_columns})
    scores = (
        metrics[score_columns]
        .multiply([weight_map[col] for col in score_columns], axis=1)
        .sum(axis=1, min_count=1)
    )

    metrics["score_fundamentalista"] = scores
    metrics = metrics.sort_values("score_fundamentalista", ascending=False)
    return metrics.reset_index(drop=True)

__all__ = [
    "FundamentalScoresConfig",
    "generate_fundamental_scores"
]

