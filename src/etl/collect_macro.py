"""
Coleta de Dados Macroeconômicos do Brasil
==========================================

Este script coleta séries históricas de indicadores macroeconômicos brasileiros
através do Sistema Gerenciador de Séries Temporais (SGS) do Banco Central do Brasil.

Funcionalidades:
- Coleta automática de séries temporais via API do BCB
- Sincronização com intervalo de datas dos dados FII
- Separação de séries por frequência (diárias e mensais)
- Tratamento de erros e dados faltantes
- Salvamento otimizado em formato Parquet

Séries coletadas:
- Taxa de Juros Real (código 1178): Taxa de juros real - Títulos públicos prefixados (diária)
- CDI (código 12): Taxa de juros - Certificado de Depósito Interbancário (diária)
- IPCA (código 433): Índice Nacional de Preços ao Consumidor Amplo (mensal)
- IGP-M (código 189): Índice Geral de Preços do Mercado (mensal)
- IBC-Br (código 24363): Índice de Atividade Econômica - Prévia do PIB (mensal)
- PMI Brasil: Índice de Gerentes de Compras (S&P Global) - placeholder implementado

Dados gerados:
- macro_data.parquet: Todas as séries combinadas
- Período: 2019-01-01 a 2025-01-01 (definido no fiis_metadata.json)
"""

import os
import pandas as pd
from datetime import datetime
from bcb import sgs
import logging
from typing import Optional
from pathlib import Path
import json

# Tenta importar fredapi, mas não falha se não disponível
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    Fred = None  # type: ignore


class MacroDataCollector:
    """Coletor de dados macroeconômicos do Brasil."""
    
    # Códigos das séries no SGS do Banco Central
    SERIES_CODES = {
        'taxa_juros_real': 1178,  # Taxa de juros real - Títulos públicos prefixados (proxy para NTN-B)
        'ipca': 433,               # Índice Nacional de Preços ao Consumidor Amplo
        'cdi': 12,                 # Taxa de juros - CDI diário
        'igpm': 189,               # Índice Geral de Preços do Mercado
        'ibc_br': 24363            # Índice de Atividade Econômica do Banco Central (prévia do PIB)
    }
    
    # PMI Brazil (usa placeholder - pode ser obtido de APIs externas)
    PMI_PLACEHOLDER = True  # Flag para indicar se deve tentar coletar PMI
    
    def __init__(self, output_dir: Optional[str] = None, log_level: int = logging.INFO):
        """
        Inicializa o coletor de dados macroeconômicos.
        
        Args:
            output_dir: Diretório de saída (padrão: data/02_processed/market)
            log_level: Nível de logging (padrão: logging.INFO)
        """
        self._setup_logger(log_level)
        self._initialize_paths(output_dir)
        self._ensure_output_directory()
        self.fii_dates = self._get_fii_date_range()
    
    def _setup_logger(self, level: int):
        """Configura logger específico da classe."""
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(level)
    
    def _initialize_paths(self, output_dir: Optional[str]):
        """Inicializa os caminhos do projeto."""
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent
            self.output_dir = str(project_root / "data" / "02_processed" / "market")
        else:
            self.output_dir = output_dir
    
    def _get_fii_date_range(self) -> tuple:
        """
        Obtém o intervalo de datas do fiis_metadata.json.
        
        Returns:
            tuple: (start_date, end_date) no formato 'YYYY-MM-DD'
        """
        project_root = Path(__file__).parent.parent.parent
        metadata_path = project_root / "data" / "metadata" / "fiis_metadata.json"
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        start_date = metadata['periodo']['inicio']
        end_date = metadata['periodo']['fim']
        self.logger.info(f"Usando intervalo de datas do metadata: {start_date} a {end_date}")
        return start_date, end_date
    
    def _ensure_output_directory(self):
        """Garante que o diretório de saída existe."""
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Diretório de saída: {self.output_dir}")
    
    def collect_series(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Coleta todas as séries macroeconômicas especificadas.
        
        Args:
            start_date: Data de início no formato 'YYYY-MM-DD' (padrão: mesma data dos dados FII)
            end_date: Data de fim no formato 'YYYY-MM-DD' (padrão: mesma data dos dados FII)
        
        Returns:
            DataFrame com as séries macroeconômicas
        """
        # Usa as datas dos dados FII para manter consistência
        if start_date is None:
            start_date = self.fii_dates[0]  # Data de início dos dados FII
        if end_date is None:
            end_date = self.fii_dates[1]    # Data de fim dos dados FII
        
        self.logger.info(f"Coletando dados macroeconômicos de {start_date} a {end_date}")
        
        # Coleta todas as séries
        all_series = {}
        failed_series = []
        
        for name, code in self.SERIES_CODES.items():
            try:
                self.logger.info(f"Coletando série {name} (código {code})")
                
                # Para séries diárias com período > 5 anos, faz múltiplas requisições
                if name in ['taxa_juros_real', 'cdi']:
                    series_data = self._collect_daily_series_multi_request(
                        name, code, start_date, end_date
                    )
                else:
                    # Séries mensais podem usar período completo
                    series_data = sgs.get({name: code}, start=start_date, end=end_date)
                
                if not series_data.empty:
                    all_series[name] = series_data[name] if name in series_data.columns else series_data
                    self.logger.info(f"Série {name} coletada com sucesso: {len(series_data)} registros")
                else:
                    self.logger.warning(f"Nenhum dado encontrado para a série {name}")
                    
            except Exception as e:
                self.logger.error(f"Erro ao coletar série {name}: {str(e)}")
                failed_series.append(name)
        
        # Tenta coletar PMI Brasil (se disponível)
        if self.PMI_PLACEHOLDER:
            try:
                self.logger.info("Tentando coletar PMI Brasil...")
                pmi_data = self._collect_pmi_data(start_date, end_date)
                if pmi_data is not None and not pmi_data.empty:
                    all_series['pmi_brasil'] = pmi_data
                    self.logger.info("PMI Brasil coletado com sucesso")
                else:
                    self.logger.warning("PMI Brasil não disponível ou sem dados")
            except Exception as e:
                self.logger.warning(f"Erro ao coletar PMI Brasil (usando placeholder): {e}")
        
        if not all_series:
            raise ValueError(f"Nenhuma série foi coletada com sucesso. Séries falharam: {failed_series}")
        
        # Cria índice de dias úteis (business days) - séries financeiras só existem em dias úteis
        # Isso evita criar muitas linhas vazias para fins de semana
        date_range = pd.bdate_range(start=start_date, end=end_date)
        df = pd.DataFrame(index=date_range)
        df.index.name = 'date'
        
        # Adiciona cada série ao DataFrame usando o índice de dias úteis
        for name, series_data in all_series.items():
            # Garante que temos uma Series
            if isinstance(series_data, pd.DataFrame):
                # Se for DataFrame, pega a primeira coluna ou a coluna com o nome da série
                if name in series_data.columns:
                    series_data = series_data[name]
                else:
                    series_data = series_data.iloc[:, 0]
            
            # Adiciona a série ao DataFrame (pandas faz o alinhamento automático por índice)
            df[name] = series_data
        
        # Reset index para ter coluna 'date'
        df = df.reset_index()
        
        # Renomeia 'date' para 'data' para compatibilidade com outros datasets
        df = df.rename(columns={'date': 'data'})
        
        # Forward fill e backward fill para preencher valores ausentes em todas as séries
        # IPCA, IGP-M, IBC-Br (mensais): propaga valor mensal para todos os dias do mês
        # Taxa Juros Real e CDI (diárias): preenche finais de semana, feriados e períodos iniciais
        for col in ['ipca', 'igpm', 'ibc_br', 'taxa_juros_real', 'cdi']:
            if col in df.columns:
                nulls_before = df[col].isnull().sum()
                # Forward fill: preenche para frente (dias/meses posteriores)
                df[col] = df[col].ffill()
                # Backward fill: preenche para trás (início da série)
                df[col] = df[col].bfill()
                nulls_after = df[col].isnull().sum()
                if nulls_before > 0:
                    self.logger.info(f"{col.upper()}: preenchidos {nulls_before - nulls_after} valores ausentes")
        
        # Forward fill para PMI se existir
        if 'pmi_brasil' in df.columns:
            nulls_before = df['pmi_brasil'].isnull().sum()
            df['pmi_brasil'] = df['pmi_brasil'].ffill()
            df['pmi_brasil'] = df['pmi_brasil'].bfill()
            nulls_after = df['pmi_brasil'].isnull().sum()
            if nulls_before > 0:
                self.logger.info(f"PMI_BRASIL: preenchidos {nulls_before - nulls_after} valores ausentes")
        
        # Remove linhas com todos os valores NaN
        df = df.dropna(how='all', subset=list(all_series.keys()))
        
        self.logger.info(f"Total de registros coletados: {len(df)}")
        if failed_series:
            self.logger.warning(f"Séries que falharam na coleta: {failed_series}")
        
        return df
    
    def _collect_daily_series_multi_request(
        self, 
        name: str, 
        code: int, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """
        Coleta série diária em múltiplas requisições se período > 5 anos.
        
        API do BCB tem limitação de ~5 anos para séries diárias.
        Este método divide em períodos de 4.5 anos e consolida.
        
        Args:
            name: Nome da série
            code: Código SGS da série
            start_date: Data de início
            end_date: Data de fim
        
        Returns:
            DataFrame consolidado com toda a série
        """
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        years_diff = (end_dt - start_dt).days / 365.25
        
        # Se período <= 5 anos, faz uma única requisição
        if years_diff <= 5:
            return sgs.get({name: code}, start=start_date, end=end_date)
        
        # Período > 5 anos: divide em chunks de 4.5 anos
        self.logger.info(f"Período de {years_diff:.1f} anos detectado. Coletando em múltiplas requisições...")
        
        all_chunks = []
        chunk_years = 4.5  # Usa 4.5 anos para ter margem de segurança
        current_start = start_dt
        
        chunk_num = 1
        while current_start < end_dt:
            # Calcula fim do chunk (4.5 anos ou data final, o que for menor)
            chunk_end = min(
                current_start + pd.Timedelta(days=int(365.25 * chunk_years)),
                end_dt
            )
            
            chunk_start_str = current_start.strftime('%Y-%m-%d')
            chunk_end_str = chunk_end.strftime('%Y-%m-%d')
            
            self.logger.info(f"  Chunk {chunk_num}: {chunk_start_str} → {chunk_end_str}")
            
            try:
                chunk_data = sgs.get({name: code}, start=chunk_start_str, end=chunk_end_str)
                if not chunk_data.empty:
                    all_chunks.append(chunk_data)
                    self.logger.info(f"  ✓ Chunk {chunk_num}: {len(chunk_data)} registros")
                else:
                    self.logger.warning(f"  ✗ Chunk {chunk_num}: sem dados")
            except Exception as e:
                self.logger.error(f"  ✗ Chunk {chunk_num}: erro - {e}")
            
            # Próximo chunk
            current_start = chunk_end + pd.Timedelta(days=1)
            chunk_num += 1
        
        # Consolida todos os chunks
        if all_chunks:
            consolidated = pd.concat(all_chunks)
            consolidated = consolidated.sort_index()
            consolidated = consolidated[~consolidated.index.duplicated(keep='last')]
            self.logger.info(f"Total consolidado: {len(consolidated)} registros")
            return consolidated
        else:
            self.logger.warning("Nenhum chunk retornou dados")
            return pd.DataFrame()
    
    def _collect_pmi_data(
        self, 
        start_date: str, 
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Coleta PMI Brasil via FRED API.
        
        Nota: PMI Brasil (Índice de Compras das Empresas) é divulgado pela S&P Global
        mensalmente. Código FRED: BSPMPM
        
        Args:
            start_date: Data de início
            end_date: Data de fim
        
        Returns:
            DataFrame com dados de PMI ou None se não disponível
        """
        if not FRED_AVAILABLE:
            self.logger.warning("fredapi não disponível. Instale com: pip install fredapi")
            return None
        
        # FRED API requer chave (key) - pode ser obtida em https://fred.stlouisfed.org/docs/api/api_key.html
        fred_key = os.getenv('FRED_API_KEY')
        
        if not fred_key:
            self.logger.warning("FRED_API_KEY não configurada. PMI Brasil não será coletado.")
            self.logger.info("Para coletar PMI Brasil, defina a variável de ambiente FRED_API_KEY")
            self.logger.info("Obtenha uma chave em: https://fred.stlouisfed.org/docs/api/api_key.html")
            return None
        
        try:
            self.logger.info("Coletando PMI Brasil via FRED API...")
            
            fred = Fred(api_key=fred_key)
            
            # Código da série PMI Brasil no FRED
            # BSPMPM: Markit Brazil Manufacturing PMI
            series_id = 'BSPMPM'
            
            # Tenta coletar dados
            series = fred.get_series(series_id, start=start_date, end=end_date)
            
            if series.empty:
                self.logger.warning("Nenhum dado de PMI Brasil encontrado no FRED")
                return None
            
            # Converte para DataFrame
            df = pd.DataFrame({series_id: series})
            df.index.name = 'date'
            df = df.reset_index()
            
            self.logger.info(f"PMI Brasil coletado: {len(df)} registros")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erro ao coletar PMI Brasil via FRED: {e}")
            self.logger.info("PMI Brasil pode não estar disponível para o período especificado")
            return None
    
    def save_to_parquet(self, df: pd.DataFrame, filename: str = "macro_data.parquet"):
        """
        Salva o DataFrame em formato Parquet.
        
        Args:
            df: DataFrame a ser salvo
            filename: Nome do arquivo (padrão: macro_data.parquet)
        """
        filepath = os.path.join(self.output_dir, filename)
        df.to_parquet(filepath, index=False, engine='pyarrow')
        self.logger.info(f"Dados salvos em Parquet: {filepath}")
        self.logger.info(f"Tamanho do arquivo: {os.path.getsize(filepath) / 1024:.2f} KB")
    
    def run_full_collection(self):
        """Executa a coleta completa de dados macroeconômicos."""
        self.logger.info("Iniciando coleta completa de dados macroeconômicos")
        
        try:
            all_data = self.collect_series()
            self.save_to_parquet(all_data, "macro_data.parquet")
            self.logger.info("Coleta completa finalizada com sucesso!")
            
        except Exception as e:
            self.logger.error(f"Erro durante a coleta completa: {str(e)}")
            raise


def main():
    """Função principal para execução do script."""
    try:
        # Inicializa o coletor
        collector = MacroDataCollector()
        
        # Executa a coleta completa
        collector.run_full_collection()
        
        print("✅ Coleta de dados macroeconômicos concluída com sucesso!")
        print(f"📁 Dados salvos em: {collector.output_dir}")
        
    except Exception as e:
        print(f"❌ Erro durante a execução: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
