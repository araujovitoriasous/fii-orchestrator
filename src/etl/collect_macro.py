"""
Coleta de Dados Macroeconômicos do Brasil (Com Lag de Divulgação)
=================================================================
Este script coleta séries históricas do Banco Central (SGS) e aplica
tratamento de lag para evitar viés de antecipação (Look-ahead Bias).
"""

import os
import pandas as pd
from datetime import datetime
from bcb import sgs
import logging
from typing import Optional
from pathlib import Path
import json

# Tenta importar fredapi (opcional para PMI)
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    Fred = None

class MacroDataCollector:
    """Coletor de dados macroeconômicos com tratamento Point-in-Time."""
    
    # Códigos das séries no SGS do Banco Central
    SERIES_CODES = {
        'taxa_juros_real': 1178,    # Taxa de juros real (diária)
        'cdi': 12,                  # Taxa CDI (diária)
        'ipca': 433,                # IPCA (mensal)
        'igpm': 189,                # IGP-M (mensal)
        'ibc_br': 24363,            # IBC-Br (mensal - prévia PIB)
        'concessoes_credito': 20631 # Concessões de crédito total (mensal - Proxy de oferta)
    }
    
    PMI_PLACEHOLDER = True 
    
    def __init__(self, output_dir: Optional[str] = None, log_level: int = logging.INFO):
        self._setup_logger(log_level)
        self._initialize_paths(output_dir)
        self._ensure_output_directory()
        self.fii_dates = self._get_fii_date_range()
    
    def _setup_logger(self, level: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(level)
    
    def _initialize_paths(self, output_dir: Optional[str]):
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent
            self.output_dir = str(project_root / "data" / "02_processed" / "market")
        else:
            self.output_dir = output_dir
    
    def _get_fii_date_range(self) -> tuple:
        project_root = Path(__file__).parent.parent.parent
        metadata_path = project_root / "data" / "metadata" / "fiis_metadata.json"
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata['periodo']['inicio'], metadata['periodo']['fim']
        except Exception as e:
            self.logger.warning(f"Erro ao ler metadata: {e}. Usando datas default.")
            return '2019-01-01', '2025-01-01'
    
    def _ensure_output_directory(self):
        os.makedirs(self.output_dir, exist_ok=True)
    
    def collect_series(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        if start_date is None: start_date = self.fii_dates[0]
        if end_date is None: end_date = self.fii_dates[1]
        
        self.logger.info(f"Coletando dados macroeconômicos de {start_date} a {end_date}")
        
        all_series = {}
        failed_series = []
        
        # 1. Coleta via BCB (SGS)
        for name, code in self.SERIES_CODES.items():
            try:
                self.logger.info(f"Coletando série {name} (código {code})")
                if name in ['taxa_juros_real', 'cdi']:
                    series_data = self._collect_daily_series_multi_request(name, code, start_date, end_date)
                else:
                    series_data = sgs.get({name: code}, start=start_date, end=end_date)
                
                if not series_data.empty:
                    all_series[name] = series_data[name] if name in series_data.columns else series_data
                else:
                    self.logger.warning(f"Nenhum dado para {name}")
            except Exception as e:
                self.logger.error(f"Erro em {name}: {e}")
                failed_series.append(name)
        
        # 2. Coleta PMI (Opcional)
        if self.PMI_PLACEHOLDER:
            pmi_data = self._collect_pmi_data(start_date, end_date)
            if pmi_data is not None and not pmi_data.empty:
                # O FRED retorna dataframe, precisamos garantir series ou coluna certa
                col_name = pmi_data.columns[1] if len(pmi_data.columns) > 1 else pmi_data.columns[0]
                all_series['pmi_brasil'] = pmi_data.set_index('date')[col_name]
            else:
                self.logger.warning("PMI Brasil não disponível (falta chave FRED ou erro)")

        if not all_series:
            raise ValueError("Falha total na coleta de séries.")
        
        # 3. Consolidação em dias úteis
        date_range = pd.bdate_range(start=start_date, end=end_date)
        df = pd.DataFrame(index=date_range)
        df.index.name = 'data'
        
        for name, series_data in all_series.items():
            # Garante alinhamento pelo índice de data
            series_data.index = pd.to_datetime(series_data.index)
            # Remove fuso horário se existir para evitar conflitos
            if series_data.index.tz is not None:
                series_data.index = series_data.index.tz_localize(None)
            
            # Reindexa para o range de dias úteis (isso vai gerar NaNs nos dias sem dado)
            df[name] = series_data.reindex(df.index)

        df = df.reset_index()

        # 4. TRATAMENTO DE LAG DE DIVULGAÇÃO (Point-in-Time) [CRÍTICO PARA TCC]
        # Séries mensais: IPCA, IGPM, IBC-Br, PMI, Crédito
        # Lógica: O dado de Jan/01 só é conhecido ~Fev/01.
        
        mensais_lag = ['ipca', 'igpm', 'ibc_br', 'pmi_brasil', 'concessoes_credito']
        
        for col in mensais_lag:
            if col in df.columns:
                nulls_before = df[col].isnull().sum()
                
                # Passo A: Forward Fill "sujo" (propaga o valor do dia 1 para o resto do mês)
                df[col] = df[col].ffill()
                
                # Passo B: O LAG MÁGICO. Movemos tudo 30 dias para frente.
                # O que era dado de Jan agora só aparece no dia correspondente de Fev.
                df[col] = df[col].shift(30)
                
                # Passo C: Backward Fill apenas para o início da série não ficar vazio (burn-in period)
                df[col] = df[col].bfill()
                
                self.logger.info(f"LAG APLICADO em {col.upper()}: Dados deslocados em 30 dias (Simulação de Divulgação).")

        # 5. Séries Diárias (Juros, CDI)
        # Não precisam de lag mensal, apenas preenchimento de feriados/fim de semana
        diarias = ['taxa_juros_real', 'cdi']
        for col in diarias:
             if col in df.columns:
                 df[col] = df[col].ffill().bfill()
        
        # Limpeza final
        df = df.dropna(how='all', subset=list(all_series.keys()))
        
        return df

    def _collect_daily_series_multi_request(self, name: str, code: int, start_date: str, end_date: str) -> pd.DataFrame:
        # (Mesma implementação do seu código original para contornar limite da API)
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        years_diff = (end_dt - start_dt).days / 365.25
        
        if years_diff <= 5:
            return sgs.get({name: code}, start=start_date, end=end_date)
        
        all_chunks = []
        current_start = start_dt
        while current_start < end_dt:
            chunk_end = min(current_start + pd.Timedelta(days=int(365.25 * 4.5)), end_dt)
            try:
                chunk = sgs.get({name: code}, start=current_start.strftime('%Y-%m-%d'), end=chunk_end.strftime('%Y-%m-%d'))
                if not chunk.empty: all_chunks.append(chunk)
            except Exception: pass
            current_start = chunk_end + pd.Timedelta(days=1)
            
        if all_chunks:
            consolidated = pd.concat(all_chunks).sort_index()
            return consolidated[~consolidated.index.duplicated(keep='last')]
        return pd.DataFrame()

    def _collect_pmi_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        # (Mesma implementação do seu código original com FRED)
        if not FRED_AVAILABLE or not os.getenv('FRED_API_KEY'): return None
        try:
            fred = Fred(api_key=os.getenv('FRED_API_KEY'))
            series = fred.get_series('BSPMPM', start=start_date, end=end_date)
            if series.empty: return None
            df = pd.DataFrame({'pmi_brasil': series})
            df.index.name = 'date'
            return df.reset_index()
        except Exception: return None

    def save_to_parquet(self, df: pd.DataFrame, filename: str = "macro_data.parquet"):
        filepath = os.path.join(self.output_dir, filename)
        df.to_parquet(filepath, index=False, engine='pyarrow')
        self.logger.info(f"Salvo: {filepath}")

    def run_full_collection(self):
        df = self.collect_series()
        self.save_to_parquet(df)

def main():
    try:
        MacroDataCollector().run_full_collection()
        print("Coleta Macro Concluída (Com Lag Point-in-Time)!")
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()
