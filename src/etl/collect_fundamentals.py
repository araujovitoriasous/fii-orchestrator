"""
Coleta de Fundamentos de FIIs - CVM (Trimestral e Mensal)
===========================================================

Sistema de coleta de fundamentos de FIIs usando dados da CVM (trimestrais e mensais).

Arquitetura OOP:
- FundamentalsCollector: Classe principal (orquestradora)
- CVMDataSource: Fonte de dados CVM (trimestrais)
- CVMMensalDataSource: Fonte de dados CVM (mensais)
- DataIntegrator: Normalização e validação

Funcionalidades:
- ✅ Coleta automática da CVM (trimestral e mensal)
- ✅ Processamento em memória (zero arquivos temporários)
- ✅ Mapeamento automático CNPJ → Ticker
- ✅ Suporte a CNPJs compartilhados
- ✅ Normalização e validação de dados
- ✅ Salvamento em Parquet otimizado

Uso:
    collector = FundamentalsCollector()
    collector.collect_all()  # Coleta e processa dados da CVM

Output: 
- fundamentals_trimestral.parquet (dados trimestrais consolidados com Taxa_Administracao, Valor_Ativo, Patrimonio_Liquido e Percentual_Inadimplencia)
"""

import requests
import zipfile
import pandas as pd
import logging
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from io import BytesIO
from abc import ABC, abstractmethod

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Classe base abstrata para fontes de dados."""
    
    @abstractmethod
    def collect(self) -> pd.DataFrame:
        """Coleta dados da fonte."""
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """Retorna o nome da fonte."""
        pass


class CVMDataSource(DataSource):
    """
    Fonte de dados da CVM (Comissão de Valores Mobiliários) - Foco TRIMESTRAL
    (Receitas, Lucro, Inadimplência)
    """
    
    BASE_URL = "https://dados.cvm.gov.br/dados/FII/DOC/INF_TRIMESTRAL/DADOS/"
    DEFAULT_START_YEAR = 2019
    DEFAULT_END_YEAR = 2025
    
    def __init__(self, fii_mapping: Dict[str, List[str]], start_year: int = None, end_year: int = None):
        self.fii_mapping = fii_mapping
        self.start_year = start_year or self.DEFAULT_START_YEAR
        self.end_year = end_year or self.DEFAULT_END_YEAR
    
    def get_source_name(self) -> str:
        return "cvm"
    
    def collect(self) -> pd.DataFrame:
        logger.info(f"\n{'='*60}")
        logger.info(f"COLETA CVM TRIMESTRAL ({self.start_year}-{self.end_year})")
        logger.info(f"{'='*60}")
        
        all_dataframes = []
        
        for year in range(self.start_year, self.end_year + 1):
            logger.info(f"\nProcessando ano TRIMESTRAL: {year}")
            
            zip_buffer = self._download_year(year)
            if zip_buffer is None:
                continue
            
            year_dfs = self._process_zip(zip_buffer, year)
            all_dataframes.extend(year_dfs)
            
            zip_buffer.close()
            logger.info(f"✓ Ano {year} trimestral concluído ({len(year_dfs)} arquivos)")
        
        if all_dataframes:
            df = pd.concat(all_dataframes, ignore_index=True)
            df = self._normalize_data(df)
            df['origem'] = self.get_source_name()
            
            logger.info(f"\n✅ CVM Trimestral: {len(df)} registros, {df['ticker'].nunique()} FIIs")
            return df
        
        logger.warning("⚠️  CVM Trimestral: Nenhum dado coletado")
        return pd.DataFrame()
    
    def _download_year(self, year: int) -> Optional[BytesIO]:
        filename = f"inf_trimestral_fii_{year}.zip"
        url = self.BASE_URL + filename
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return BytesIO(response.content)
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro ao baixar {year}: {e}")
            return None
    
    def _process_zip(self, zip_buffer: BytesIO, year: int) -> List[pd.DataFrame]:
        dataframes = []
        
        try:
            with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                
                for csv_file in csv_files:
                    # ATUALIZADO: Foco nos arquivos que contêm os dados-alvo
                    # 'resultado_contabil_financeiro' (Receita/Lucro) e 'imovel' (Inadimplência)
                    if not any(kw in csv_file for kw in ['resultado_contabil_financeiro', 'imovel']):
                        continue
                    
                    try:
                        with zip_ref.open(csv_file) as f:
                            df = self._read_csv(f, csv_file)
                            if df is not None:
                                dataframes.append(df)
                    except Exception as e:
                        logger.warning(f"  ✗ {csv_file}: {e}")
        
        except Exception as e:
            logger.error(f"Erro ao processar ZIP {year}: {e}")
        
        return dataframes
    
    def _read_csv(self, file_obj, filename: str) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(file_obj, sep=';', encoding='latin1', decimal=',')
            
            if df.empty:
                return None
            
            df.columns = df.columns.str.strip().str.lower()
            
            if 'cnpj_fundo_classe' in df.columns:
                cnpj_col = 'cnpj_fundo_classe'
            elif 'cnpj_fundo' in df.columns:
                df.rename(columns={'cnpj_fundo': 'cnpj_fundo_classe'}, inplace=True)
                cnpj_col = 'cnpj_fundo_classe'
            else:
                return None
            
            df['ticker_list'] = df[cnpj_col].map(self.fii_mapping)
            df = df[df['ticker_list'].notna()]
            if df.empty:
                return None
            
            df_expanded = df.explode('ticker_list')
            df_expanded.rename(columns={
                'ticker_list': 'ticker',
                cnpj_col: 'cnpj'
            }, inplace=True)
            
            return df_expanded if not df_expanded.empty else None
            
        except Exception:
            return None
    
    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'data_referencia' in df.columns:
            df['data_referencia'] = pd.to_datetime(df['data_referencia'], errors='coerce')
            df['data'] = df['data_referencia']
            df['ano'] = df['data_referencia'].dt.year
            df['trimestre'] = df['data_referencia'].dt.quarter
        
        if 'ticker' in df.columns and 'data_referencia' in df.columns:
            # ATUALIZADO: Mescla colunas de diferentes arquivos (ex: imovel e resultado)
            # Agrupa dados e pega o primeiro valor não nulo (combina colunas de 'imovel' e 'resultado')
            # NOTA: O 'percentual_inadimplencia' pode variar por imóvel. 
            # Aqui, pegamos a média por fundo/trimestre.
            
            # Define colunas numéricas para agregar
            numeric_cols = [
                'receita_aluguel_investimento_financeiro', 'receita_juros_tvm_financeiro', 
                'receita_juros_aplicacao_financeiro', 'resultado_trimestral_liquido_financeiro',
                'resultado_liquido_recurso_liquidez_financeiro', 'percentual_inadimplencia'
            ]
            
            # Garante que colunas numéricas existam
            for col in numeric_cols:
                if col not in df.columns:
                    df[col] = np.nan
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Agregações: Soma para receitas/lucros, Média para percentuais
            agg_rules = {
                'receita_aluguel_investimento_financeiro': 'sum',
                'receita_juros_tvm_financeiro': 'sum',
                'receita_juros_aplicacao_financeiro': 'sum',
                'resultado_trimestral_liquido_financeiro': 'sum',
                'resultado_liquido_recurso_liquidez_financeiro': 'sum',
                'percentual_inadimplencia': 'mean',
                'cnpj': 'first', # Mantém o CNPJ
                'ano': 'first',
                'trimestre': 'first'
            }
            
            # Filtra regras de agregação apenas para colunas que existem no DF
            agg_rules_filtered = {k: v for k, v in agg_rules.items() if k in df.columns}

            if not agg_rules_filtered:
                logger.warning("Nenhuma coluna de agregação encontrada. Normalização pode falhar.")
                return df
                
            df_grouped = df.groupby(['ticker', 'data_referencia']).agg(agg_rules_filtered).reset_index()
            
            df = df_grouped
        
        df = df.sort_values(['ticker', 'data_referencia'])
        
        return df


class CVMMensalDataSource(DataSource):
    """
    Fonte de dados da CVM (Comissão de Valores Mobiliários) - Foco MENSAL
    (Taxa Adm, Ativo Total, Patrimônio Líquido)
    """
    
    BASE_URL = "https://dados.cvm.gov.br/dados/FII/DOC/INF_MENSAL/DADOS/"
    DEFAULT_START_YEAR = 2019
    DEFAULT_END_YEAR = 2025
    
    def __init__(self, fii_mapping: Dict[str, List[str]], start_year: int = None, end_year: int = None):
        self.fii_mapping = fii_mapping
        self.start_year = start_year or self.DEFAULT_START_YEAR
        self.end_year = end_year or self.DEFAULT_END_YEAR
    
    def get_source_name(self) -> str:
        return "cvm_mensal"
    
    def collect(self) -> pd.DataFrame:
        logger.info(f"\n{'='*60}")
        logger.info(f"COLETA CVM MENSAL ({self.start_year}-{self.end_year})")
        logger.info(f"{'='*60}")
        
        all_dataframes = []
        
        for year in range(self.start_year, self.end_year + 1):
            logger.info(f"\nProcessando ano MENSAL: {year}")
            
            zip_buffer = self._download_year(year)
            if zip_buffer is None:
                continue
            
            year_dfs = self._process_zip(zip_buffer, year)
            all_dataframes.extend(year_dfs)
            
            zip_buffer.close()
            logger.info(f"✓ Ano {year} mensal concluído ({len(year_dfs)} arquivos)")
        
        if all_dataframes:
            df = pd.concat(all_dataframes, ignore_index=True)
            df = self._normalize_data(df)
            df['origem'] = self.get_source_name()
            
            logger.info(f"\n✅ CVM Mensal: {len(df)} registros, {df['ticker'].nunique()} FIIs")
            return df
        
        logger.warning("⚠️  CVM Mensal: Nenhum dado coletado")
        return pd.DataFrame()
    
    def _download_year(self, year: int) -> Optional[BytesIO]:
        filename = f"inf_mensal_fii_{year}.zip"
        url = self.BASE_URL + filename
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return BytesIO(response.content)
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro ao baixar {year}: {e}")
            return None
    
    def _process_zip(self, zip_buffer: BytesIO, year: int) -> List[pd.DataFrame]:
        dataframes = []
        
        try:
            with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                
                for csv_file in csv_files:
                    # ATUALIZADO: Lê 'complemento' (Taxa_Adm, Ativo, PL)
                    # O 'geral' e 'ativo_passivo' não são mais necessários
                    if 'complemento' not in csv_file:
                        continue
                    
                    try:
                        with zip_ref.open(csv_file) as f:
                            df = self._read_csv(f, csv_file)
                            if df is not None:
                                dataframes.append(df)
                    except Exception as e:
                        logger.warning(f"  ✗ {csv_file}: {e}")
        
        except Exception as e:
            logger.error(f"Erro ao processar ZIP mensal {year}: {e}")
        
        return dataframes
    
    def _read_csv(self, file_obj, filename: str) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(file_obj, sep=';', encoding='latin1', decimal=',')
            
            if df.empty:
                return None
            
            df.columns = df.columns.str.strip().str.lower()
            
            if 'cnpj_fundo_classe' in df.columns:
                cnpj_col = 'cnpj_fundo_classe'
            elif 'cnpj_fundo' in df.columns:
                df.rename(columns={'cnpj_fundo': 'cnpj_fundo_classe'}, inplace=True)
                cnpj_col = 'cnpj_fundo_classe'
            else:
                return None
            
            df['ticker_list'] = df[cnpj_col].map(self.fii_mapping)
            df = df[df['ticker_list'].notna()]
            if df.empty:
                return None
            
            df_expanded = df.explode('ticker_list')
            df_expanded.rename(columns={
                'ticker_list': 'ticker',
                cnpj_col: 'cnpj'
            }, inplace=True)
            
            return df_expanded if not df_expanded.empty else None
            
        except Exception:
            return None
    
    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'data_referencia' in df.columns:
            df['data_referencia'] = pd.to_datetime(df['data_referencia'], errors='coerce')
            df['data'] = df['data_referencia']
            df['ano'] = df['data_referencia'].dt.year
            df['mes'] = df['data_referencia'].dt.month
            df['trimestre'] = df['data_referencia'].dt.quarter
        
        if 'ticker' in df.columns and 'data_referencia' in df.columns:
            # Agrupa para consolidar (embora 'complemento' deva ser único)
            df_grouped = df.groupby(['ticker', 'data_referencia']).first().reset_index()
            if 'data_referencia' in df_grouped.columns:
                 df_grouped['ano'] = df_grouped['data_referencia'].dt.year
                 df_grouped['mes'] = df_grouped['data_referencia'].dt.month
                 df_grouped['trimestre'] = df_grouped['data_referencia'].dt.quarter
            df = df_grouped
        
        df = df.sort_values(['ticker', 'data_referencia'])
        
        return df


class DataIntegrator:
    """Normaliza e valida dados da CVM (Trimestral e Mensal)."""
    
    # ATUALIZADO: Schema alvo final
    TARGET_SCHEMA = {
        'ticker': 'str',                    # Chave de identificação
        'data_referencia': 'datetime',      # Chave de tempo (Trimestral)
        'ano': 'int',                       # Ano de referência
        'trimestre': 'int',                 # Trimestre
        'cnpj': 'str',                      # CNPJ do fundo
        'Receita_Caixa': 'float',           # (Trimestral) Receitas financeiras principais
        'Lucro_Caixa_Trimestral': 'float',  # (Trimestral) Resultado líquido trimestral
        'Liquidez_Caixa': 'float',          # (Trimestral) Recursos de liquidez
        'Taxa_Administracao': 'float',      # (Mensal, agregado) Taxa de administração
        'Valor_Ativo': 'float',             # (Mensal, agregado) Ativo Total para Alavancagem
        'Patrimonio_Liquido': 'float',      # (Mensal, agregado) PL para Alavancagem
        'Percentual_Inadimplencia': 'float' # (Trimestral) Inadimplência direta
    }
    
    # ATUALIZADO: Mapeamento combinado (Trimestral e Mensal)
    CVM_TO_TARGET_MAPPING = {
        # Dados Trimestrais (do CVMDataSource)
        'Receita_Caixa': {
            'formula': 'receita_aluguel_investimento_financeiro + receita_juros_tvm_financeiro + receita_juros_aplicacao_financeiro',
            'description': 'Soma das receitas financeiras'
        },
        'Lucro_Caixa_Trimestral': {
            'formula': 'resultado_trimestral_liquido_financeiro',
            'description': 'Resultado líquido trimestral financeiro'
        },
        'Liquidez_Caixa': {
            'formula': 'resultado_liquido_recurso_liquidez_financeiro',
            'description': 'Recursos de liquidez disponíveis'
        },
        'Percentual_Inadimplencia': { 
            'column': 'percentual_inadimplencia', # Do inf_trimestral_fii_imovel
            'description': 'Percentual de inadimplência (do arquivo imovel)'
        },
        
        # Dados Mensais (do CVMMensalDataSource)
        'Taxa_Administracao': {
            'column': 'percentual_despesas_taxa_administracao', # Do inf_mensal_fii_complemento
            'description': 'Taxa de administração (% ao mês)'
        },
        'Valor_Ativo': {
            'column': 'valor_ativo', # Do inf_mensal_fii_complemento
            'description': 'Valor Total Ativo (Mensal)'
        },
        'Patrimonio_Liquido': {
            'column': 'patrimonio_liquido', # Do inf_mensal_fii_complemento
            'description': 'Valor Patrimônio Líquido (Mensal)'
        }
    }
    
    @classmethod
    def normalize_cvm_data(cls, df_cvm: pd.DataFrame, is_mensal: bool = False) -> pd.DataFrame:
        """
        Converte dados CVM (Trimestral ou Mensal) para um esquema parcial.
        """
        logger.info(f"Normalizando dados CVM {'Mensal' if is_mensal else 'Trimestral'}...")
        
        if df_cvm.empty:
            logger.warning("DataFrame CVM vazio")
            return pd.DataFrame()
        
        df_normalized = pd.DataFrame()
        
        # Chaves obrigatórias
        df_normalized['ticker'] = df_cvm['ticker']
        df_normalized['data_referencia'] = df_cvm['data_referencia']
        df_normalized['ano'] = df_cvm['ano']
        if is_mensal:
            df_normalized['mes'] = df_cvm['mes']
            df_normalized['trimestre'] = df_cvm['data_referencia'].dt.quarter
        else:
            df_normalized['trimestre'] = df_cvm['trimestre']
        df_normalized['cnpj'] = df_cvm['cnpj']
        
        # Indicadores calculados (fórmulas ou colunas diretas)
        for target_col, mapping_info in cls.CVM_TO_TARGET_MAPPING.items():
            
            # Decide quais colunas processar baseado no tipo (Mensal vs Trimestral)
            is_trimestral_col = target_col in ['Receita_Caixa', 'Lucro_Caixa_Trimestral', 'Liquidez_Caixa', 'Percentual_Inadimplencia']
            is_mensal_col = target_col in ['Taxa_Administracao', 'Valor_Ativo', 'Patrimonio_Liquido']
            
            if (is_mensal and not is_mensal_col) or (not is_mensal and not is_trimestral_col):
                continue

            try:
                if 'column' in mapping_info:
                    column_name = mapping_info['column']
                    if column_name in df_cvm.columns:
                        df_normalized[target_col] = pd.to_numeric(df_cvm[column_name], errors='coerce')
                    else:
                        df_normalized[target_col] = 0.0 # Preenche com 0 se a coluna não existe
                    continue
                
                # É uma fórmula
                formula = mapping_info['formula']
                allowed_columns = set(df_cvm.columns)
                
                if '+' in formula:
                    cols_to_sum = [col.strip() for col in formula.split('+')]
                    calculated_value = None
                    
                    for col in cols_to_sum:
                        if col in allowed_columns:
                            col_value = pd.to_numeric(df_cvm[col], errors='coerce').fillna(0)
                            if calculated_value is None:
                                calculated_value = col_value
                            else:
                                calculated_value = calculated_value + col_value
                        else:
                            logger.warning(f"  ✗ Coluna {col} em fórmula para {target_col} não encontrada. Usando 0.")
                    
                    if calculated_value is None:
                        calculated_value = 0.0
                else:
                    temp_df = pd.DataFrame(index=df_cvm.index)
                    formula_safe = formula
                    found_cols = 0
                    for col in allowed_columns:
                        if col in formula:
                            temp_df[col] = pd.to_numeric(df_cvm[col], errors='coerce').fillna(0)
                            formula_safe = formula_safe.replace(col, f"temp_df['{col}']")
                            found_cols += 1
                    
                    if found_cols > 0:
                        calculated_value = eval(formula_safe)
                    else:
                        calculated_value = 0.0

                df_normalized[target_col] = calculated_value
                
            except Exception as e:
                logger.warning(f"  ✗ {target_col}: Erro ao calcular {formula} - {e}. Preenchendo com 0.")
                df_normalized[target_col] = 0.0
        
        df_normalized = cls._fill_missing_values(df_normalized, is_mensal)
        
        return df_normalized
    
    @classmethod
    def _fill_missing_values(cls, df: pd.DataFrame, is_mensal: bool = False) -> pd.DataFrame:
        """Preenche valores faltantes usando interpolação temporal por ticker."""
        
        if is_mensal:
            indicators = ['Taxa_Administracao', 'Valor_Ativo', 'Patrimonio_Liquido']
        else:
            indicators = ['Receita_Caixa', 'Lucro_Caixa_Trimestral', 'Liquidez_Caixa', 'Percentual_Inadimplencia']
        
        df = df.sort_values(['ticker', 'data_referencia'])
        
        df_filled_list = []
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy()
            
            for indicator in indicators:
                if indicator in ticker_data.columns:
                    ticker_data[indicator] = pd.to_numeric(ticker_data[indicator], errors='coerce')
                    # Interpolação linear para valores faltantes
                    ticker_data[indicator] = ticker_data[indicator].interpolate(method='linear', limit_direction='both')
                    # Preenche o que sobrou (ex: início da série)
                    ticker_data[indicator] = ticker_data[indicator].bfill().ffill()
                    ticker_data[indicator].fillna(0.0, inplace=True)
            
            df_filled_list.append(ticker_data)
        
        if not df_filled_list:
            return df
            
        df_filled = pd.concat(df_filled_list, ignore_index=True)
        return df_filled


class FundamentalsCollector:
    """Coletor de fundamentos de FIIs usando apenas dados da CVM."""
    
    def __init__(self, base_path: Optional[Path] = None):
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent / "data"
        
        self.base_path = Path(base_path)
        self.output_path = self.base_path / "02_processed" / "fundamentals"
        self.metadata_path = self.base_path / "metadata"
        
        self._ensure_directories()
        self.metadata = self._load_metadata()
        self.fii_mapping = self._create_cnpj_mapping()
    
    def _ensure_directories(self):
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def _load_metadata(self) -> Dict:
        metadata_file = self.metadata_path / "fiis_metadata.json"
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Metadata não encontrado: {metadata_file}")
            raise
        except Exception as e:
            logger.error(f"Erro ao carregar metadata: {e}")
            raise
    
    def _create_cnpj_mapping(self) -> Dict[str, List[str]]:
        mapping = {}
        
        for fii in self.metadata['fiis']:
            cnpj = fii['cnpj']
            ticker = fii['ticker']
            
            if cnpj not in mapping:
                mapping[cnpj] = []
            mapping[cnpj].append(ticker)
        
        shared = {cnpj: tickers for cnpj, tickers in mapping.items() if len(tickers) > 1}
        if shared:
            logger.info(f"✅ CNPJs compartilhados detectados: {shared}")
        
        logger.info(f"Mapeamento: {len(mapping)} CNPJs únicos → {len(self.metadata['fiis'])} FIIs")
        
        return mapping
    
    def _merge_trimestral_with_mensal(
        self, 
        df_trimestral: pd.DataFrame, 
        df_mensal: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Une dados trimestrais (base) com dados mensais agregados (Taxa_Adm, Ativo, PL).
        """
        logger.info("\nUnindo dados trimestrais com dados mensais agregados...")
        
        if df_trimestral.empty:
             logger.warning("DataFrame trimestral vazio. Retornando DataFrame vazio.")
             return pd.DataFrame(columns=list(DataIntegrator.TARGET_SCHEMA.keys()))

        if df_mensal.empty:
            logger.warning("DataFrame mensal vazio. Métricas mensais (Taxa_Adm, Alavancagem) ficarão zeradas.")
            # Garante que as colunas-alvo existem no DF trimestral
            for col in ['Taxa_Administracao', 'Valor_Ativo', 'Patrimonio_Liquido']:
                 if col not in df_trimestral.columns:
                      df_trimestral[col] = 0.0
            return df_trimestral.copy()
        
        # Agrega dados mensais por trimestre
        # Taxa_Adm: Média do trimestre
        # Ativo/PL: Pega o último valor reportado no trimestre
        
        df_mensal_agg = df_mensal.groupby(['ticker', 'ano', 'trimestre']).agg(
            Taxa_Administracao=('Taxa_Administracao', 'mean'),
            Valor_Ativo=('Valor_Ativo', 'last'),
            Patrimonio_Liquido=('Patrimonio_Liquido', 'last')
        ).reset_index()

        # Merge com dados trimestrais
        df_combined = df_trimestral.merge(
            df_mensal_agg,
            on=['ticker', 'ano', 'trimestre'],
            how='left'
        )
        
        # Preenche NaNs (para tickers/trimestres que não estavam no mensal)
        for col in ['Taxa_Administracao', 'Valor_Ativo', 'Patrimonio_Liquido']:
            if col not in df_combined.columns:
                 df_combined[col] = 0.0
            else:
                 # Interpola primeiro (preenche buracos)
                 df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
                 df_combined[col] = df_combined.groupby('ticker')[col].transform(
                     lambda x: x.interpolate(method='linear', limit_direction='both')
                 )
                 # Preenche o que sobrou (ex: início da série)
                 df_combined[col] = df_combined[col].bfill().ffill()
                 df_combined[col] = df_combined[col].fillna(0.0)


        logger.info(f"✓ Dados trimestrais e mensais unidos: {len(df_combined)} registros trimestrais")
        
        return df_combined
    
    def collect_all(self) -> pd.DataFrame:
        logger.info("\n" + "="*80)
        logger.info(" "*20 + "COLETA DE FUNDAMENTOS - FII-O (CVM)")
        logger.info("="*80)
        
        try:
            # 1. Coleta CVM Trimestral (Receita, Lucro, Inadimplência)
            cvm_source = CVMDataSource(self.fii_mapping)
            df_cvm_trim = cvm_source.collect()
            
            if df_cvm_trim.empty:
                logger.error("❌ Coleta CVM trimestral resultou em DataFrame vazio")
                df_normalized_trim = pd.DataFrame(columns=list(DataIntegrator.TARGET_SCHEMA.keys()))
                # Remove colunas que virão do mensal
                df_normalized_trim = df_normalized_trim.drop(columns=['Taxa_Administracao', 'Valor_Ativo', 'Patrimonio_Liquido'])
            else:
                df_normalized_trim = DataIntegrator.normalize_cvm_data(df_cvm_trim, is_mensal=False)
            
            # 2. Coleta CVM Mensal (Taxa_Adm, Ativo, PL)
            cvm_mensal_source = CVMMensalDataSource(self.fii_mapping)
            df_cvm_mensal = cvm_mensal_source.collect()
            
            if df_cvm_mensal.empty:
                logger.warning("⚠️  Coleta CVM mensal resultou em DataFrame vazio")
                df_normalized_mensal = pd.DataFrame()
            else:
                df_normalized_mensal = DataIntegrator.normalize_cvm_data(df_cvm_mensal, is_mensal=True)
            
            # 3. Une dados trimestrais (Base) com mensais (Agregados)
            df_combined = self._merge_trimestral_with_mensal(df_normalized_trim, df_normalized_mensal)
            
            # 4. Re-executa o _fill_missing_values no combinado final
            df_combined = DataIntegrator._fill_missing_values(df_combined, is_mensal=False) # Preenche os trimestrais
            df_combined = DataIntegrator._fill_missing_values(df_combined, is_mensal=True) # Preenche os mensais agregados

            # 5. Salva apenas o arquivo final consolidado
            self._save_to_parquet(df_combined, "fundamentals_trimestral.parquet")
            
            return df_combined
            
        except Exception as e:
            logger.error(f"\n❌ Erro durante a coleta: {e}")
            raise
    
    def _save_to_parquet(self, df: pd.DataFrame, filename: str = "fundamentals_trimestral.parquet"):
        if df.empty:
            logger.warning(f"DataFrame vazio, nada para salvar em {filename}")
            return
        
        expected_schema = set(DataIntegrator.TARGET_SCHEMA.keys())
        schema_to_use = DataIntegrator.TARGET_SCHEMA
        actual_schema = set(df.columns)
        
        missing_cols = expected_schema.difference(actual_schema)
        if missing_cols:
            logger.warning(f"⚠️  Esquema não corresponde ao alvo. Adicionando colunas faltantes com 0.0: {sorted(missing_cols)}")
            for col in missing_cols:
                df[col] = 0.0

        extra_cols = actual_schema.difference(expected_schema)
        if extra_cols:
             logger.warning(f"⚠️  Colunas extras detectadas. Removendo: {sorted(extra_cols)}")
             df = df.drop(columns=list(extra_cols))
        
        df = df[list(schema_to_use.keys())]
        
        for col, expected_type in schema_to_use.items():
            if col in df.columns:
                try:
                    if expected_type == 'datetime':
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    elif expected_type == 'float':
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                    elif expected_type == 'int':
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                    elif expected_type == 'str':
                        df[col] = df[col].astype(str)
                except Exception as e:
                     logger.error(f"Erro ao converter tipo da coluna {col} para {expected_type}: {e}")
                     df[col] = None

        parquet_path = self.output_path / filename
        
        df.to_parquet(
            parquet_path,
            index=False,
            engine='pyarrow',
            compression='snappy'
        )
        
        size_mb = parquet_path.stat().st_size / (1024 * 1024)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ DADOS SALVOS")
        logger.info(f"{'='*60}")
        logger.info(f"Arquivo: {parquet_path}")
        logger.info(f"Tamanho: {size_mb:.2f} MB")
        logger.info(f"Registros: {len(df)}")
        logger.info(f"Colunas: {len(df.columns)}")
        logger.info(f"FIIs: {df['ticker'].nunique()}")
        
        logger.info(f"\n🎯 DADOS CVM PROCESSADOS:")
        logger.info(f"   Sistema simplificado usando apenas dados oficiais da CVM!")


def main():
    try:
        collector = FundamentalsCollector()
        df_final = collector.collect_all()
        
        print("\n" + "="*80)
        print("✅ Coleta de fundamentos concluída!")
        print(f"📁 Dados salvos em: {collector.output_path}")
        print(f"📊 Dataset Final: {len(df_final)} registros, {df_final['ticker'].nunique() if not df_final.empty else 0} FIIs")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Erro durante a execução: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())