from typing import List, Dict, Any, Optional, Tuple
import polars as pl
from loguru import logger
import numpy as np
from datetime import datetime, timedelta

class DataValidator:
    """Classe para validação de dados e detecção de outliers."""
    
    def __init__(self):
        self.validation_results = []
    
    def validate_price_data(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, List[Dict]]:
        """
        Valida dados de preços e detecta outliers.
        
        Args:
            df: DataFrame com dados de preços
            
        Returns:
            Tuple com DataFrame limpo e lista de problemas encontrados
        """
        issues = []
        
        # Validações básicas
        if df.is_empty():
            issues.append({"type": "empty_data", "message": "DataFrame vazio"})
            return df, issues
        
        # Verificar colunas obrigatórias
        required_cols = ["date", "ticker", "close"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append({
                "type": "missing_columns", 
                "message": f"Colunas obrigatórias ausentes: {missing_cols}"
            })
            return df, issues
        
        # Validações de tipos de dados
        df_clean = df.clone()
        
        # Converter date para datetime se necessário
        if "date" in df_clean.columns and df_clean["date"].dtype != pl.Datetime:
            try:
                df_clean = df_clean.with_columns(pl.col("date").str.to_datetime())
            except Exception as e:
                issues.append({
                    "type": "date_conversion_error",
                    "message": f"Erro ao converter coluna date: {e}"
                })
        
        # Validar preços
        if "close" in df_clean.columns:
            # Preços negativos ou zero
            negative_prices = df_clean.filter(pl.col("close") <= 0)
            if not negative_prices.is_empty():
                issues.append({
                    "type": "invalid_prices",
                    "message": f"Encontrados {negative_prices.height} preços inválidos (<=0)"
                })
                df_clean = df_clean.filter(pl.col("close") > 0)
            
            # Detectar outliers usando IQR
            outliers = self._detect_price_outliers(df_clean, "close")
            if outliers:
                issues.append({
                    "type": "price_outliers",
                    "message": f"Detectados {len(outliers)} outliers de preço"
                })
                # Marcar outliers mas não remover
                df_clean = df_clean.with_columns(
                    pl.when(pl.col("close").is_in(outliers))
                    .then(pl.lit(True))
                    .otherwise(pl.lit(False))
                    .alias("is_outlier")
                )
        
        # Validar volumes
        if "volume" in df_clean.columns:
            negative_volumes = df_clean.filter(pl.col("volume") < 0)
            if not negative_volumes.is_empty():
                issues.append({
                    "type": "invalid_volumes",
                    "message": f"Encontrados {negative_volumes.height} volumes negativos"
                })
                df_clean = df_clean.with_columns(
                    pl.when(pl.col("volume") < 0)
                    .then(pl.lit(None))
                    .otherwise(pl.col("volume"))
                    .alias("volume")
                )
        
        # Validar datas
        if "date" in df_clean.columns:
            future_dates = df_clean.filter(pl.col("date") > datetime.now())
            if not future_dates.is_empty():
                issues.append({
                    "type": "future_dates",
                    "message": f"Encontradas {future_dates.height} datas futuras"
                })
                df_clean = df_clean.filter(pl.col("date") <= datetime.now())
        
        # Validar tickers
        if "ticker" in df_clean.columns:
            invalid_tickers = df_clean.filter(
                ~pl.col("ticker").str.matches(r"^[A-Z]{4}\d{2}$")
            )
            if not invalid_tickers.is_empty():
                issues.append({
                    "type": "invalid_tickers",
                    "message": f"Encontrados {invalid_tickers.height} tickers com formato inválido"
                })
        
        self.validation_results.extend(issues)
        return df_clean, issues
    
    def validate_dividend_data(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, List[Dict]]:
        """
        Valida dados de dividendos.
        
        Args:
            df: DataFrame com dados de dividendos
            
        Returns:
            Tuple com DataFrame limpo e lista de problemas encontrados
        """
        issues = []
        
        if df.is_empty():
            issues.append({"type": "empty_data", "message": "DataFrame vazio"})
            return df, issues
        
        df_clean = df.clone()
        
        # Validar valores de dividendos
        if "value" in df_clean.columns:
            negative_values = df_clean.filter(pl.col("value") <= 0)
            if not negative_values.is_empty():
                issues.append({
                    "type": "invalid_dividend_values",
                    "message": f"Encontrados {negative_values.height} valores de dividendo inválidos (<=0)"
                })
                df_clean = df_clean.filter(pl.col("value") > 0)
        
        # Validar datas
        if "ex_date" in df_clean.columns:
            future_dates = df_clean.filter(pl.col("ex_date") > datetime.now())
            if not future_dates.is_empty():
                issues.append({
                    "type": "future_ex_dates",
                    "message": f"Encontradas {future_dates.height} datas ex-dividendo futuras"
                })
                df_clean = df_clean.filter(pl.col("ex_date") <= datetime.now())
        
        self.validation_results.extend(issues)
        return df_clean, issues
    
    def _detect_price_outliers(self, df: pl.DataFrame, price_col: str, threshold: float = 1.5) -> List[float]:
        """
        Detecta outliers usando método IQR.
        
        Args:
            df: DataFrame com dados
            price_col: Nome da coluna de preços
            threshold: Multiplicador do IQR para detecção de outliers
            
        Returns:
            Lista de valores considerados outliers
        """
        try:
            prices = df[price_col].drop_nulls().to_list()
            if len(prices) < 4:  # Precisamos de pelo menos 4 pontos para IQR
                return []
            
            q1 = np.percentile(prices, 25)
            q3 = np.percentile(prices, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            outliers = [p for p in prices if p < lower_bound or p > upper_bound]
            return outliers
            
        except Exception as e:
            logger.warning(f"Erro ao detectar outliers: {e}")
            return []
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Retorna resumo das validações realizadas."""
        if not self.validation_results:
            return {"status": "success", "issues": [], "total_issues": 0}
        
        issue_types = {}
        for issue in self.validation_results:
            issue_type = issue["type"]
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        return {
            "status": "issues_found" if self.validation_results else "success",
            "issues": self.validation_results,
            "total_issues": len(self.validation_results),
            "issue_types": issue_types
        }
    
    def clear_results(self):
        """Limpa resultados de validação anteriores."""
        self.validation_results = []

def validate_fund_ticker(ticker: str) -> bool:
    """
    Valida se um ticker de FII tem formato válido.
    
    Args:
        ticker: String do ticker
        
    Returns:
        True se válido, False caso contrário
    """
    import re
    pattern = r"^[A-Z]{4}\d{2}$"
    return bool(re.match(pattern, ticker))

def validate_cnpj(cnpj: str) -> bool:
    """
    Valida formato de CNPJ.
    
    Args:
        cnpj: String do CNPJ
        
    Returns:
        True se válido, False caso contrário
    """
    import re
    # Remove caracteres especiais
    cnpj_clean = re.sub(r'[^\d]', '', cnpj)
    
    # Verifica se tem 14 dígitos
    if len(cnpj_clean) != 14:
        return False
    
    # Verifica se não são todos iguais
    if len(set(cnpj_clean)) == 1:
        return False
    
    return True
