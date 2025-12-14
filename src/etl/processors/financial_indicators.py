"""
Processador de Indicadores Financeiros e Fundamentos (Seção 2.1.4 e 2.1.6)
==========================================================================
Responsável por:
1. Receber dados brutos de preços e fundamentos CVM.
2. Aplicar normalização (Min-Max) e ajustes de direção.
3. Calcular métricas derivadas (Crescimento, Alavancagem).
4. Preparar o dataset final para consumo pelos Agentes.

O módulo expõe a função `process_financial_indicators`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUIRED_PRICE_COLUMNS: frozenset[str] = frozenset(
    {"ticker", "data", "pvp", "vacancia", "dy_12m", "vp_cota", "tipo_gestao"}
)

# Atualizado para refletir o output real do collect_fundamentals.py
REQUIRED_FUND_COLUMNS: frozenset[str] = frozenset(
    {
        "ticker",
        "data_referencia",
        "Lucro_Caixa_Trimestral",
        "Receita_Caixa",
        "Liquidez_Caixa",
        "Taxa_Administracao",
        "Valor_Ativo",          # Necessário para Alavancagem
        "Patrimonio_Liquido",   # Necessário para Alavancagem
    }
)

REQUIRED_DIVIDENDS_COLUMNS: frozenset[str] = frozenset(
    {"ticker", "data", "dividendo"}
)

@dataclass(frozen=True)
class IndicatorConfig:
    """
    Configuração para normalização e ponderação dos indicadores (Seção 2.3.4).
    Agora inclui peso específico para Alavancagem.
    """
    weight_pvp: float = 0.125
    weight_vacancia: float = 0.125
    weight_dy: float = 0.125
    weight_lucro_growth: float = 0.125
    weight_receita_caixa: float = 0.100
    weight_liquidez_caixa: float = 0.100
    weight_taxa_administracao: float = 0.100
    weight_alavancagem: float = 0.100        # Novo Peso
    weight_dividend_forward: float = 0.100   # Ajustado para fechar 1.0 (aprox)
    
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
            "alavancagem_score": self.weight_alavancagem,  # Novo Score
            "indicador_fundamentalista": 0.0 
        }

def _ensure_datetime(df: pd.DataFrame, column: str) -> pd.Series:
    """Garante que a coluna especificada esteja em formato datetime."""
    series = df[column]
    if not pd.api.types.is_datetime64_any_dtype(series):
        series = pd.to_datetime(series, errors="coerce")
    if hasattr(series.dt, "tz"):
        series = series.dt.tz_localize(None)
    return series

def _min_max_normalize(values: pd.Series) -> pd.Series:
    """Normaliza valores para o intervalo [0, 1] usando min-max."""
    valid = values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=values.index)
    min_val = valid.min()
    max_val = valid.max()
    if np.isclose(max_val, min_val, equal_nan=True):
        return pd.Series(0.5, index=values.index)
    normalized = (values - min_val) / (max_val - min_val)
    return normalized.clip(0.0, 1.0)

def _adjust_direction(latest_prices: pd.DataFrame) -> pd.DataFrame:
    """Ajusta as métricas para que valores altos signifiquem bons sinais."""
    adjusted = latest_prices.copy()
    
    # P/VP: Ajuste linear onde próximo de 1.0 é melhor, ou desconto é bom.
    adjusted["pvp"] = pd.to_numeric(adjusted["pvp"], errors='coerce').fillna(1.0)
    adjusted["pvp_adjusted"] = 1.0 + (1.0 - adjusted["pvp"])
    
    # Vacância: Baixa é melhor. (1.0 - vacancia)
    adjusted["vacancia"] = pd.to_numeric(adjusted["vacancia"], errors='coerce').fillna(0.0)
    adjusted["vacancia_adjusted"] = 1.0 - adjusted["vacancia"]
    adjusted["vacancia_adjusted"] = adjusted["vacancia_adjusted"].clip(0.0, 1.0)

    # DY: Alto é melhor.
    adjusted["dy_value"] = pd.to_numeric(adjusted["dy_12m"], errors='coerce').fillna(0.0)
    
    return adjusted

def _compute_latest_lucro_growth(fundamentals: pd.DataFrame, latest_funds: pd.DataFrame) -> pd.DataFrame:
    """Calcula o crescimento percentual mais recente do lucro caixa por ticker."""
    if fundamentals.empty:
        return latest_funds.copy()

    df = fundamentals.copy()
    df["data_referencia"] = _ensure_datetime(df, "data_referencia")
    df.sort_values(["ticker", "data_referencia"], inplace=True)

    df["lucro_growth"] = df.groupby("ticker")["Lucro_Caixa_Trimestral"].pct_change()
    df["lucro_growth"] = df["lucro_growth"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["lucro_growth"] = df["lucro_growth"].clip(-0.5, 1.0)

    # Pega o último growth disponível na base histórica filtrada
    latest_growth = df.groupby("ticker")["lucro_growth"].last()
    
    latest_funds = latest_funds.copy()
    latest_funds = latest_funds.merge(latest_growth, on="ticker", how="left")
    latest_funds["lucro_growth"] = latest_funds["lucro_growth"].fillna(0.0)
    
    return latest_funds

def _filter_point_in_time(df: pd.DataFrame, date_col: str, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """Garante que nenhum dado futuro seja utilizado (prevenção de Look-ahead Bias)."""
    df = df.copy()
    series_date = _ensure_datetime(df, date_col)
    mask_past = series_date <= cutoff_date
    df_past = df[mask_past]
    
    if df_past.empty:
        return pd.DataFrame(columns=df.columns)

    idx = df_past.groupby("ticker")[date_col].idxmax()
    return df_past.loc[idx].copy()

def process_financial_indicators(
    data_prices: pd.DataFrame,
    data_fundamentals: pd.DataFrame,
    data_dividends: pd.DataFrame | None = None,
    cutoff_date: str | pd.Timestamp | None = None,
    config: IndicatorConfig | None = None,
) -> pd.DataFrame:
    """
    Processa e normaliza os indicadores financeiros para uma data específica.
    Implementa a lógica da Seção 2.1.6 (Normalização) do Anteprojeto.
    """
    config = config or IndicatorConfig()
    
    if cutoff_date is None:
        cutoff_date = pd.to_datetime("today")
    else:
        cutoff_date = pd.to_datetime(cutoff_date)

    # 1. Filtragem Point-in-Time
    latest_prices = _filter_point_in_time(data_prices, "data", cutoff_date)
    latest_funds = _filter_point_in_time(data_fundamentals, "data_referencia", cutoff_date)

    if latest_prices.empty or latest_funds.empty:
        return pd.DataFrame(columns=["ticker", "indicador_fundamentalista"])

    # 2. Tratamento e Normalização de Mercado
    latest_prices = _adjust_direction(latest_prices)
    
    # 3. Cálculo de Crescimento (Histórico)
    funds_history = data_fundamentals[
        _ensure_datetime(data_fundamentals, "data_referencia") <= cutoff_date
    ].copy()
    
    latest_funds_growth = _compute_latest_lucro_growth(funds_history, latest_funds)

    # 4. Integração
    merged = latest_prices.merge(
        latest_funds_growth,
        on="ticker",
        how="inner", 
        suffixes=("_preco", "_fundo")
    )

    if merged.empty:
        return pd.DataFrame(columns=["ticker", "indicador_fundamentalista"])

    # 5. Cálculo de Alavancagem (Seção 2.1.4)
    # Lógica: Alavancagem = (Ativo - PL) / Ativo
    # Tratamento para evitar divisão por zero
    merged["Valor_Ativo"] = pd.to_numeric(merged["Valor_Ativo"], errors='coerce').fillna(0.0)
    merged["Patrimonio_Liquido"] = pd.to_numeric(merged["Patrimonio_Liquido"], errors='coerce').fillna(0.0)
    
    # Evita divisão por zero
    mask_valid_assets = merged["Valor_Ativo"] > 0
    merged.loc[mask_valid_assets, "alavancagem_raw"] = (
        merged.loc[mask_valid_assets, "Valor_Ativo"] - merged.loc[mask_valid_assets, "Patrimonio_Liquido"]
    ) / merged.loc[mask_valid_assets, "Valor_Ativo"]
    
    # Se ativo for 0 ou inválido, assume alavancagem 0 (neutro) ou trata como erro
    merged["alavancagem_raw"] = merged["alavancagem_raw"].fillna(0.0).clip(0.0, 1.0)

    # 6. Geração dos Scores Normalizados
    metrics = pd.DataFrame(index=merged.index)
    metrics["ticker"] = merged["ticker"]
    
    # Mercado
    metrics["pvp_score"] = _min_max_normalize(merged["pvp_adjusted"])
    metrics["vacancia_score"] = _min_max_normalize(merged["vacancia_adjusted"])
    metrics["dy_score"] = _min_max_normalize(merged["dy_value"])
    
    # Fundamentos
    metrics["lucro_growth_score"] = _min_max_normalize(merged["lucro_growth"])
    
    for col in ["Receita_Caixa", "Liquidez_Caixa", "Taxa_Administracao"]:
        merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(0.0)

    metrics["receita_caixa_score"] = _min_max_normalize(merged["Receita_Caixa"])
    metrics["liquidez_caixa_score"] = _min_max_normalize(merged["Liquidez_Caixa"])
    
    # Taxa ADM e Alavancagem: Quanto menor, melhor -> Inverte (1 - score)
    metrics["taxa_administracao_score"] = 1.0 - _min_max_normalize(merged["Taxa_Administracao"])
    metrics["alavancagem_score"] = 1.0 - _min_max_normalize(merged["alavancagem_raw"])

    # 7. Pré-cálculo do Indicador Fundamentalista
    final_score = 0.0
    total_weight = 0.0
    weights = config.weights
    
    for metric_col, weight in weights.items():
        if metric_col == "indicador_fundamentalista": continue
        
        if metric_col in metrics.columns:
            final_score += metrics[metric_col].fillna(0) * weight
            total_weight += weight
            
    metrics["indicador_fundamentalista"] = final_score / (total_weight if total_weight > 0 else 1.0)
    
    # Colunas úteis para debug/visualização
    metrics["alavancagem_real"] = merged["alavancagem_raw"] # Dado real para o analista ver
    
    return metrics.sort_values("indicador_fundamentalista", ascending=False).reset_index(drop=True)

__all__ = [
    "IndicatorConfig",
    "process_financial_indicators"
]