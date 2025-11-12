"""
Construção do dataset supervisionado para o agente fundamentalista preditivo.

Este módulo consolida preços ajustados, dividendos e (opcionalmente) o benchmark
IFIX para gerar a variável-alvo binária `outperform_ifix_6m`. A abordagem segue
o horizonte de previsão de 6 meses (t+6), calculando o retorno total (preço +
dividendos) de cada FII e comparando com o retorno equivalente do benchmark.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

DEFAULT_HORIZON_MONTHS = 6
REQUIRED_PRICE_COLUMNS = {"ticker", "data", "preco_ajustado"}
REQUIRED_DIVIDEND_COLUMNS = {"ticker", "data", "dividendo"}


def build_predictive_dataset(
    prices_path: Path | str | None = None,
    dividends_path: Path | str | None = None,
    *,
    ifix_path: Path | str | None = None,
    horizon_months: int = DEFAULT_HORIZON_MONTHS,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Constrói o dataset supervisionado com rótulo relativo ao IFIX (ou proxy).

    Args:
        prices_path: Caminho para ``prices.parquet`` (dados brutos).
        dividends_path: Caminho para ``dividends.parquet`` (dados brutos).
        ifix_path: Caminho opcional com série histórica do IFIX.
        horizon_months: Horizonte de previsão em meses (padrão: 6).
        output_path: Caminho opcional para salvar o parquet resultante.

    Returns:
        DataFrame contendo features básicas, retornos totais e a coluna-alvo
        binária ``outperform_ifix_6m``.
    """

    project_root = Path(__file__).parent.parent.parent
    prices_path = Path(prices_path or project_root / "data" / "01_raw" / "prices.parquet")
    dividends_path = Path(dividends_path or project_root / "data" / "01_raw" / "dividends.parquet")

    if not prices_path.exists():
        raise FileNotFoundError(f"Arquivo de preços não encontrado: {prices_path}")
    if not dividends_path.exists():
        raise FileNotFoundError(f"Arquivo de dividendos não encontrado: {dividends_path}")

    LOGGER.info("Carregando dados de preços: %s", prices_path)
    prices_df = pd.read_parquet(prices_path)
    LOGGER.info("Carregando dados de dividendos: %s", dividends_path)
    dividends_df = pd.read_parquet(dividends_path)

    _validate_dataframe(prices_df, REQUIRED_PRICE_COLUMNS, "preços")
    _validate_dataframe(dividends_df, REQUIRED_DIVIDEND_COLUMNS, "dividendos")

    prices_df = _prepare_prices(prices_df)
    dividends_df = _prepare_dividends(dividends_df)

    dataset = _compute_fii_targets(prices_df, dividends_df, horizon_months)

    if dataset.empty:
        raise ValueError("Dataset supervisionado ficou vazio. Verifique disponibilidade de dados t+6.")

    if ifix_path is not None:
        dataset = _attach_ifix_benchmark(dataset, Path(ifix_path), horizon_months)
    else:
        LOGGER.info("Nenhum caminho do IFIX fornecido. Utilizando proxy interno (média cruzada).")
        dataset = _attach_internal_benchmark(dataset)

    if output_path is None:
        output_path = project_root / "data" / "03_model" / "predictive_dataset.parquet"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Salvando dataset supervisionado em: %s", output_path)
    dataset.to_parquet(output_path, index=False)

    return dataset


def _validate_dataframe(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame de {name} sem colunas obrigatórias: {sorted(missing)}")


def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    prepared = prices.copy()
    prepared["data"] = pd.to_datetime(prepared["data"], utc=False, errors="coerce")
    prepared = prepared.dropna(subset=["data", "preco_ajustado", "ticker"])
    prepared = prepared.sort_values(["ticker", "data"])
    return prepared


def _prepare_dividends(dividends: pd.DataFrame) -> pd.DataFrame:
    prepared = dividends.copy()
    prepared["data"] = pd.to_datetime(prepared["data"], utc=False, errors="coerce")
    prepared = prepared.dropna(subset=["data", "dividendo", "ticker"])
    prepared = prepared.sort_values(["ticker", "data"])
    return prepared


def _compute_fii_targets(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    horizon_months: int,
) -> pd.DataFrame:
    ticker_datasets: list[pd.DataFrame] = []

    for ticker, price_slice in prices.groupby("ticker", group_keys=False):
        div_slice = dividends[dividends["ticker"] == ticker]
        ticker_data = _compute_ticker_total_returns(price_slice, div_slice, horizon_months)
        if not ticker_data.empty:
            ticker_datasets.append(ticker_data)

    if not ticker_datasets:
        return pd.DataFrame()

    dataset = pd.concat(ticker_datasets, ignore_index=True)
    dataset = dataset.sort_values(["reference_date", "ticker"]).reset_index(drop=True)
    return dataset


def _compute_ticker_total_returns(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    horizon_months: int,
) -> pd.DataFrame:
    monthly = (
        prices.set_index("data")
        .groupby("ticker")
        .resample("M")
        .last()
        .reset_index()
    )

    monthly = monthly.dropna(subset=["preco_ajustado"])
    if monthly.empty:
        return pd.DataFrame()

    monthly["future_target_date"] = monthly["data"] + pd.DateOffset(months=horizon_months)

    future_lookup = monthly[["data", "preco_ajustado"]].rename(
        columns={"data": "future_lookup_date", "preco_ajustado": "future_price"}
    )
    future_lookup = future_lookup.sort_values("future_lookup_date")

    merged = pd.merge_asof(
        monthly.sort_values("future_target_date"),
        future_lookup,
        left_on="future_target_date",
        right_on="future_lookup_date",
        direction="forward",
    )

    merged = merged.rename(columns={"data": "reference_date"})
    merged = merged.drop(columns=["future_lookup_date"])

    merged = merged.dropna(subset=["future_price"])
    if merged.empty:
        return pd.DataFrame()

    if dividends.empty:
        merged["dividend_sum_6m"] = 0.0
    else:
        merged["dividend_sum_6m"] = _compute_dividend_window(
            dividends,
            merged["reference_date"],
            merged["future_target_date"],
        )

    merged["total_return_6m"] = (
        (merged["future_price"] + merged["dividend_sum_6m"]) / merged["preco_ajustado"]
    ) - 1.0

    merged = merged.rename(columns={"future_target_date": "future_date"})
    merged["total_return_6m"] = merged["total_return_6m"].astype(float)

    valid = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["total_return_6m"])
    selected_columns = [
        "ticker",
        "reference_date",
        "future_date",
        "preco_ajustado",
        "future_price",
        "dividend_sum_6m",
        "total_return_6m",
    ]
    return valid[selected_columns]


def _compute_dividend_window(
    dividends: pd.DataFrame,
    start_dates: pd.Series,
    end_dates: pd.Series,
) -> pd.Series:
    divs = dividends[["data", "dividendo"]].copy()
    divs = divs.sort_values("data")
    divs["cum_div"] = divs["dividendo"].cumsum()

    start_df = pd.DataFrame({"data": start_dates})
    end_df = pd.DataFrame({"data": end_dates})

    cum_start = pd.merge_asof(
        start_df.sort_values("data"),
        divs[["data", "cum_div"]],
        on="data",
        direction="backward",
    )["cum_div"].fillna(0.0)

    cum_end = pd.merge_asof(
        end_df.sort_values("data"),
        divs[["data", "cum_div"]],
        on="data",
        direction="backward",
    )["cum_div"].fillna(0.0)

    aligned_start = cum_start.reindex(start_df.index)
    aligned_end = cum_end.reindex(end_df.index)

    return (aligned_end.values - aligned_start.values).astype(float)


def _attach_ifix_benchmark(
    dataset: pd.DataFrame,
    ifix_path: Path,
    horizon_months: int,
) -> pd.DataFrame:
    if not ifix_path.exists():
        LOGGER.warning(
            "Arquivo do IFIX %s não encontrado. Recuando para proxy interno.", ifix_path
        )
        return _attach_internal_benchmark(dataset)

    LOGGER.info("Carregando série do IFIX: %s", ifix_path)
    ifix_df = pd.read_parquet(ifix_path)
    if ifix_df.empty:
        LOGGER.warning("Arquivo do IFIX vazio. Usando proxy interno.")
        return _attach_internal_benchmark(dataset)

    if "data" not in ifix_df.columns:
        raise ValueError("DataFrame do IFIX deve possuir a coluna 'data'.")

    price_col = None
    for candidate in ("preco_ajustado", "close", "ifix", "valor", "price"):
        if candidate in ifix_df.columns:
            price_col = candidate
            break

    if price_col is None:
        raise ValueError(
            "DataFrame do IFIX precisa de coluna de preço. Exemplos aceitos: "
            "'preco_ajustado', 'close', 'ifix', 'valor', 'price'."
        )

    ifix_prepared = ifix_df[["data", price_col]].rename(columns={price_col: "ifix_price"}).copy()
    ifix_prepared["data"] = pd.to_datetime(ifix_prepared["data"], errors="coerce", utc=False)
    ifix_prepared = ifix_prepared.dropna(subset=["data", "ifix_price"])
    ifix_prepared = ifix_prepared.sort_values("data")

    monthly_ifix = (
        ifix_prepared.set_index("data")
        .resample("M")
        .last()
        .dropna(subset=["ifix_price"])
        .reset_index()
    )
    monthly_ifix["future_target_date"] = monthly_ifix["data"] + pd.DateOffset(months=horizon_months)

    future_lookup = monthly_ifix[["data", "ifix_price"]].rename(
        columns={"data": "future_lookup_date", "ifix_price": "ifix_future_price"}
    )
    future_lookup = future_lookup.sort_values("future_lookup_date")

    merged_ifix = pd.merge_asof(
        monthly_ifix.sort_values("future_target_date"),
        future_lookup,
        left_on="future_target_date",
        right_on="future_lookup_date",
        direction="forward",
    )

    merged_ifix = merged_ifix.dropna(subset=["ifix_future_price"])
    merged_ifix = merged_ifix.rename(columns={"data": "reference_date"})
    merged_ifix = merged_ifix.drop(columns=["future_lookup_date"])

    merged_ifix["ifix_return_6m"] = (merged_ifix["ifix_future_price"] / merged_ifix["ifix_price"]) - 1.0
    merged_ifix["ifix_return_6m"] = merged_ifix["ifix_return_6m"].astype(float)

    benchmark = merged_ifix[["reference_date", "ifix_return_6m"]]
    dataset = dataset.merge(benchmark, on="reference_date", how="left")

    dataset["benchmark_return_6m"] = dataset["ifix_return_6m"]
    dataset["benchmark_name"] = "ifix"
    missing_mask = dataset["benchmark_return_6m"].isna()

    if missing_mask.any():
        LOGGER.warning(
            "Retornos do IFIX ausentes para %d registros. Substituindo por proxy interno.",
            int(missing_mask.sum()),
        )
        internal_benchmark = (
            dataset.loc[missing_mask]
            .groupby("reference_date")["total_return_6m"]
            .mean()
            .rename("benchmark_return_internal")
        )
        dataset = dataset.merge(internal_benchmark, on="reference_date", how="left")
        dataset.loc[missing_mask, "benchmark_return_6m"] = dataset.loc[
            missing_mask, "benchmark_return_internal"
        ]
        dataset.loc[missing_mask, "benchmark_name"] = "equal_weight_internal_proxy"
        dataset = dataset.drop(columns=["benchmark_return_internal"])

    dataset = dataset.drop(columns=["ifix_return_6m"])
    dataset["outperform_ifix_6m"] = (dataset["total_return_6m"] > dataset["benchmark_return_6m"]).astype(int)
    return dataset


def _attach_internal_benchmark(dataset: pd.DataFrame) -> pd.DataFrame:
    benchmark = (
        dataset.groupby("reference_date")["total_return_6m"]
        .mean()
        .rename("benchmark_return_6m")
    )
    augmented = dataset.merge(benchmark, on="reference_date", how="left")
    augmented["benchmark_name"] = "equal_weight_internal"
    augmented["outperform_ifix_6m"] = (
        augmented["total_return_6m"] > augmented["benchmark_return_6m"]
    ).astype(int)
    return augmented


__all__ = ["build_predictive_dataset"]


