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

prices = pd.read_parquet(\"data/01_raw/prices.parquet\")
fundamentals = pd.read_parquet(\"data/02_processed/fundamentals/fundamentals_trimestral.parquet\")

scores = generate_fundamental_scores(prices, fundamentals)
print(scores.head())
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


REQUIRED_PRICE_COLUMNS: frozenset[str] = frozenset(
    {"ticker", "data", "pvp", "vacancia", "dy_12m"}
)
REQUIRED_FUND_COLUMNS: frozenset[str] = frozenset(
    {"ticker", "data_referencia", "Lucro_Caixa_Trimestral"}
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

    A configuração padrão assume pesos iguais para manter transparência na fase inicial.
    """
    weight_pvp: float = 0.25
    weight_vacancia: float = 0.25
    weight_dy: float = 0.25
    weight_lucro_growth: float = 0.25

    @property
    def weights(self) -> Mapping[str, float]:
        return {
            "pvp_score": self.weight_pvp,
            "vacancia_score": self.weight_vacancia,
            "dy_score": self.weight_dy,
            "lucro_growth_score": self.weight_lucro_growth,
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
    fundamentals: pd.DataFrame, latest_funds: pd.DataFrame
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

    fundamentals["lucro_growth"] = fundamentals.groupby("ticker")[
        "Lucro_Caixa_Trimestral"
    ].pct_change()
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
        ["ticker", "data_referencia", "Lucro_Caixa_Trimestral", "lucro_growth"]
    ]


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
            `Lucro_Caixa_Trimestral`.
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
    metrics["lucro_caixa_trimestral"] = merged["Lucro_Caixa_Trimestral"]
    metrics["pvp_score"] = _min_max_normalize(merged["pvp_adjusted"])
    metrics["vacancia_score"] = _min_max_normalize(merged["vacancia_adjusted"])
    metrics["dy_score"] = _min_max_normalize(merged["dy_value"])
    metrics["lucro_growth_score"] = _min_max_normalize(merged["lucro_growth"])

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


def export_fundamental_scores_to_csv(
    data_prices: pd.DataFrame,
    data_fundamentals: pd.DataFrame,
    output_path: str,
    *,
    config: FundamentalScoresConfig | None = None,
    index: bool = False,
    **csv_kwargs,
) -> pd.DataFrame:
    """
    Calcula o score fundamentalista e salva o resultado em um arquivo CSV.

    Args:
        data_prices: DataFrame de preços com colunas mínimas exigidas pelo agente.
        data_fundamentals: DataFrame de fundamentos trimestrais requerido pelo agente.
        output_path: Caminho do arquivo CSV a ser gerado.
        config: Configuração opcional para customizar os pesos das métricas.
        index: Define se o índice do DataFrame deve ser persistido no CSV (default False).
        **csv_kwargs: Argumentos adicionais enviados para `DataFrame.to_csv`.

    Returns:
        O DataFrame resultante utilizado na exportação.
    """

    scores = generate_fundamental_scores(
        data_prices,
        data_fundamentals,
        config=config,
    )
    scores.to_csv(output_path, index=index, **csv_kwargs)
    return scores


__all__ = [
    "FundamentalScoresConfig",
    "generate_fundamental_scores",
    "export_fundamental_scores_to_csv",
]

