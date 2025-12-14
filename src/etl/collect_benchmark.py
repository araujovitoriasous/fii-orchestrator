import yfinance as yf
import pandas as pd
from pathlib import Path
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_ifix():
    """
    Coleta dados históricos do IFIX (Índice de Fundos de Investimentos Imobiliários)
    usando o yfinance e salva em formato Parquet para a camada Raw.
    """
    ticker_symbol = "^FIX"
    logger.info(f"Iniciando coleta do benchmark: {ticker_symbol}...")
    
    # Download data
    try:
        ifix = yf.download(ticker_symbol, start="2019-01-01", end="2025-01-01", progress=False)
    except Exception as e:
        logger.error(f"Erro ao baixar dados do yfinance: {e}")
        return

    if not ifix.empty:
        logger.info(f"Dados coletados com sucesso. Registros: {len(ifix)}")
        
        # Ajustes e limpeza
        # Yahoo Finance retorna MultiIndex nas colunas em versÃµes mais recentes, 
        # mas para um Ãºnico ticker pode ser simples. Vamos garantir.
        if isinstance(ifix.columns, pd.MultiIndex):
             ifix = ifix.xs(ticker_symbol, axis=1, level=1, drop_level=True)
             
        # Seleciona Adj Close se existir, caso contrÃ¡rio Close
        col_to_use = 'Adj Close' if 'Adj Close' in ifix.columns else 'Close'
        
        if col_to_use in ifix.columns:
            ifix = ifix[[col_to_use]].rename(columns={col_to_use: 'fechamento'})
        else:
             logger.warning(f"Coluna de fechamento nÃ£o encontrada. Colunas disponÃveis: {ifix.columns}")
             # Tenta pegar a primeira coluna se for sÃ©rie Ãºnica
             if ifix.shape[1] > 0:
                 ifix = ifix.iloc[:, 0].to_frame(name='fechamento')
        
        ifix.index.name = 'data'
        ifix = ifix.reset_index()
        
        # Salvar
        output_dir = Path("data/01_raw")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "benchmark.parquet"
        
        try:
            ifix.to_parquet(output_path, index=False)
            logger.info(f"IFIX salvo em {output_path}")
        except Exception as e:
             logger.error(f"Erro ao salvar arquivo parquet: {e}")

    else:
        logger.error("Nenhum dado retornado para o IFIX.")

if __name__ == "__main__":
    collect_ifix()
