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
- fundamentals_trimestral.parquet (dados trimestrais consolidados com Taxa_Administracao)
"""

import requests
import zipfile
import pandas as pd
import logging
import json
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
    """Fonte de dados da CVM (Comissão de Valores Mobiliários)."""
    
    BASE_URL = "https://dados.cvm.gov.br/dados/FII/DOC/INF_TRIMESTRAL/DADOS/"
    DEFAULT_START_YEAR = 2019
    DEFAULT_END_YEAR = 2025
    
    def __init__(self, fii_mapping: Dict[str, List[str]], start_year: int = None, end_year: int = None):
        """
        Inicializa fonte CVM.
        
        Args:
            fii_mapping: Dicionário {CNPJ: [Ticker1, Ticker2, ...]}
            start_year: Ano inicial
            end_year: Ano final
        """
        self.fii_mapping = fii_mapping
        self.start_year = start_year or self.DEFAULT_START_YEAR
        self.end_year = end_year or self.DEFAULT_END_YEAR
    
    def get_source_name(self) -> str:
        return "cvm"
    
    def collect(self) -> pd.DataFrame:
        """
        Coleta dados da CVM.
        
        Returns:
            DataFrame com dados CVM
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"COLETA CVM ({self.start_year}-{self.end_year})")
        logger.info(f"{'='*60}")
        
        all_dataframes = []
        
        for year in range(self.start_year, self.end_year + 1):
            logger.info(f"\nProcessando ano: {year}")
            
            # Download e processamento em memória
            zip_buffer = self._download_year(year)
            if zip_buffer is None:
                continue
            
            year_dfs = self._process_zip(zip_buffer, year)
            all_dataframes.extend(year_dfs)
            
            zip_buffer.close()
            logger.info(f"✓ Ano {year} concluído ({len(year_dfs)} arquivos)")
        
        # Consolida dados
        if all_dataframes:
            df = pd.concat(all_dataframes, ignore_index=True)
            df = self._normalize_data(df)
            df['origem'] = self.get_source_name()
            
            logger.info(f"\n✅ CVM: {len(df)} registros, {df['ticker'].nunique()} FIIs")
            return df
        
        logger.warning("⚠️  CVM: Nenhum dado coletado")
        return pd.DataFrame()
    
    def _download_year(self, year: int) -> Optional[BytesIO]:
        """Baixa ZIP de um ano em memória."""
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
        """Processa ZIP diretamente da memória."""
        dataframes = []
        
        try:
            with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                
                for csv_file in csv_files:
                    # Filtra apenas arquivos relevantes para dados trimestrais
                    if not any(kw in csv_file for kw in ['geral', 'resultado_contabil_financeiro']):
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
        """Lê CSV e mapeia CNPJ → Tickers."""
        try:
            df = pd.read_csv(file_obj, sep=';', encoding='latin1', decimal=',')
            
            if df.empty:
                return None
            
            # Normaliza colunas
            df.columns = df.columns.str.strip().str.lower()
            
            # Identifica coluna CNPJ
            if 'cnpj_fundo_classe' in df.columns:
                cnpj_col = 'cnpj_fundo_classe'
            elif 'cnpj_fundo' in df.columns:
                df.rename(columns={'cnpj_fundo': 'cnpj_fundo_classe'}, inplace=True)
                cnpj_col = 'cnpj_fundo_classe'
            else:
                return None
            
            # Mapeia CNPJ → Lista de Tickers
            df['ticker_list'] = df[cnpj_col].map(self.fii_mapping)
            
            # Filtra não mapeados
            df = df[df['ticker_list'].notna()]
            if df.empty:
                return None
            
            # Expande: 1 linha por ticker (suporta CNPJs compartilhados)
            df_expanded = df.explode('ticker_list')
            df_expanded.rename(columns={
                'ticker_list': 'ticker',
                cnpj_col: 'cnpj'
            }, inplace=True)
            
            return df_expanded if not df_expanded.empty else None
            
        except Exception:
            return None
    
    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza dados CVM."""
        if 'data_referencia' in df.columns:
            df['data_referencia'] = pd.to_datetime(df['data_referencia'], errors='coerce')
            df['data'] = df['data_referencia']
            df['ano'] = df['data_referencia'].dt.year
            df['trimestre'] = df['data_referencia'].dt.quarter
        
        # Remove duplicatas
        if 'ticker' in df.columns and 'data_referencia' in df.columns:
            df = df.drop_duplicates(subset=['ticker', 'data_referencia'], keep='last')
        
        # Ordena
        df = df.sort_values(['ticker', 'data_referencia'])
        
        return df


class CVMMensalDataSource(DataSource):
    """Fonte de dados da CVM para relatórios MENS UAIS de FIIs."""
    
    BASE_URL = "https://dados.cvm.gov.br/dados/FII/DOC/INF_MENSAL/DADOS/"
    DEFAULT_START_YEAR = 2019
    DEFAULT_END_YEAR = 2025
    
    def __init__(self, fii_mapping: Dict[str, List[str]], start_year: int = None, end_year: int = None):
        """
        Inicializa fonte CVM mensal.
        
        Args:
            fii_mapping: Dicionário {CNPJ: [Ticker1, Ticker2, ...]}
            start_year: Ano inicial
            end_year: Ano final
        """
        self.fii_mapping = fii_mapping
        self.start_year = start_year or self.DEFAULT_START_YEAR
        self.end_year = end_year or self.DEFAULT_END_YEAR
    
    def get_source_name(self) -> str:
        return "cvm_mensal"
    
    def collect(self) -> pd.DataFrame:
        """
        Coleta dados mensais da CVM.
        
        Returns:
            DataFrame com dados CVM mensais
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"COLETA CVM MENSAL ({self.start_year}-{self.end_year})")
        logger.info(f"{'='*60}")
        
        all_dataframes = []
        
        for year in range(self.start_year, self.end_year + 1):
            logger.info(f"\nProcessando ano mensal: {year}")
            
            # Download e processamento em memória
            zip_buffer = self._download_year(year)
            if zip_buffer is None:
                continue
            
            year_dfs = self._process_zip(zip_buffer, year)
            all_dataframes.extend(year_dfs)
            
            zip_buffer.close()
            logger.info(f"✓ Ano {year} mensal concluído ({len(year_dfs)} arquivos)")
        
        # Consolida dados
        if all_dataframes:
            df = pd.concat(all_dataframes, ignore_index=True)
            df = self._normalize_data(df)
            df['origem'] = self.get_source_name()
            
            logger.info(f"\n✅ CVM Mensal: {len(df)} registros, {df['ticker'].nunique()} FIIs")
            return df
        
        logger.warning("⚠️  CVM Mensal: Nenhum dado coletado")
        return pd.DataFrame()
    
    def _download_year(self, year: int) -> Optional[BytesIO]:
        """Baixa ZIP de um ano em memória (mensal)."""
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
        """Processa ZIP diretamente da memória (mensal)."""
        dataframes = []
        
        try:
            with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                
                for csv_file in csv_files:
                    # Filtra apenas arquivo complemento para dados mensais (tem a taxa de administração)
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
        """Lê CSV e mapeia CNPJ → Tickers."""
        try:
            df = pd.read_csv(file_obj, sep=';', encoding='latin1', decimal=',')
            
            if df.empty:
                return None
            
            # Normaliza colunas
            df.columns = df.columns.str.strip().str.lower()
            
            # Identifica coluna CNPJ
            if 'cnpj_fundo_classe' in df.columns:
                cnpj_col = 'cnpj_fundo_classe'
            elif 'cnpj_fundo' in df.columns:
                df.rename(columns={'cnpj_fundo': 'cnpj_fundo_classe'}, inplace=True)
                cnpj_col = 'cnpj_fundo_classe'
            else:
                return None
            
            # Mapeia CNPJ → Lista de Tickers
            df['ticker_list'] = df[cnpj_col].map(self.fii_mapping)
            
            # Filtra não mapeados
            df = df[df['ticker_list'].notna()]
            if df.empty:
                return None
            
            # Expande: 1 linha por ticker (suporta CNPJs compartilhados)
            df_expanded = df.explode('ticker_list')
            df_expanded.rename(columns={
                'ticker_list': 'ticker',
                cnpj_col: 'cnpj'
            }, inplace=True)
            
            return df_expanded if not df_expanded.empty else None
            
        except Exception:
            return None
    
    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza dados CVM mensais."""
        if 'data_referencia' in df.columns:
            df['data_referencia'] = pd.to_datetime(df['data_referencia'], errors='coerce')
            df['data'] = df['data_referencia']
            df['ano'] = df['data_referencia'].dt.year
            df['mes'] = df['data_referencia'].dt.month
        
        # Remove duplicatas
        if 'ticker' in df.columns and 'data_referencia' in df.columns:
            df = df.drop_duplicates(subset=['ticker', 'data_referencia'], keep='last')
        
        # Ordena
        df = df.sort_values(['ticker', 'data_referencia'])
        
        return df


class DataIntegrator:
    """Normaliza e valida dados da CVM."""
    
    # ESQUEMA ALVO - Indicadores Financeiros da CVM com Taxa de Administração
    TARGET_SCHEMA = {
        'ticker': 'str',                    # Chave de identificação
        'data_referencia': 'datetime',      # Chave de tempo (Trimestral)
        'ano': 'int',                       # Ano de referência
        'trimestre': 'int',                 # Trimestre
        'cnpj': 'str',                      # CNPJ do fundo
        'Receita_Caixa': 'float',           # Receitas financeiras principais
        'Lucro_Caixa_Trimestral': 'float',  # Resultado líquido trimestral
        'Liquidez_Caixa': 'float',          # Recursos de liquidez
        'Taxa_Administracao': 'float'       # Taxa de administração (média trimestral)
    }
    
    # Mapeamento de conversão CVM → Indicadores Financeiros
    CVM_TO_TARGET_MAPPING = {
        'Receita_Caixa': {
            'formula': 'receita_aluguel_investimento_financeiro + receita_juros_tvm_financeiro + receita_juros_aplicacao_financeiro',
            'description': 'Soma das receitas financeiras (aluguel + juros TVM + juros aplicação)'
        },
        'Lucro_Caixa_Trimestral': {
            'formula': 'resultado_trimestral_liquido_financeiro',
            'description': 'Resultado líquido trimestral financeiro'
        },
        'Liquidez_Caixa': {
            'formula': 'resultado_liquido_recurso_liquidez_financeiro',
            'description': 'Recursos de liquidez disponíveis'
        },
        'Taxa_Administracao': {
            'column': 'percentual_despesas_taxa_administracao',
            'description': 'Taxa de administração (% ao mês)'
        }
    }
    
    @classmethod
    def normalize_cvm_data(cls, df_cvm: pd.DataFrame) -> pd.DataFrame:
        """
        Converte dados CVM para o esquema alvo.
        
        Args:
            df_cvm: DataFrame com dados brutos da CVM
        
        Returns:
            DataFrame com esquema alvo
        """
        logger.info("Normalizando dados CVM para esquema alvo...")
        
        if df_cvm.empty:
            logger.warning("DataFrame CVM vazio")
            return pd.DataFrame(columns=list(cls.TARGET_SCHEMA.keys()))
        
        df_normalized = pd.DataFrame()
        
        # Chaves obrigatórias
        df_normalized['ticker'] = df_cvm['ticker']
        df_normalized['data_referencia'] = df_cvm['data_referencia']
        df_normalized['ano'] = df_cvm['ano']
        df_normalized['trimestre'] = df_cvm['trimestre'] if 'trimestre' in df_cvm.columns else df_cvm['mes'] if 'mes' in df_cvm.columns else None
        df_normalized['cnpj'] = df_cvm['cnpj']
        
        # Indicadores calculados (fórmulas ou colunas diretas)
        for target_col, mapping_info in cls.CVM_TO_TARGET_MAPPING.items():
            description = mapping_info['description']
            
            try:
                # Se tem 'column', usa coluna direta; se tem 'formula', calcula
                if 'column' in mapping_info:
                    # É uma coluna direta
                    column_name = mapping_info['column']
                    if column_name in df_cvm.columns:
                        df_normalized[target_col] = pd.to_numeric(df_cvm[column_name], errors='coerce')
                        logger.debug(f"  ✓ {target_col}: {column_name} → {description}")
                    else:
                        logger.warning(f"  ✗ {target_col}: Coluna {column_name} não encontrada")
                        df_normalized[target_col] = None
                    continue
                
                # É uma fórmula
                formula = mapping_info['formula']
                allowed_columns = set(df_cvm.columns)
                formula_safe = formula
                
                # Para formulas que são somas, calculamos de forma especial
                if '+' in formula:
                    # Extrai colunas da fórmula de soma
                    cols_to_sum = [col.strip() for col in formula.split('+')]
                    calculated_value = None
                    
                    # Soma as colunas, preenchendo NaN com 0 para cada coluna
                    for col in cols_to_sum:
                        if col in allowed_columns:
                            col_value = pd.to_numeric(df_cvm[col], errors='coerce').fillna(0)
                            if calculated_value is None:
                                calculated_value = col_value
                            else:
                                calculated_value = calculated_value + col_value
                    
                    if calculated_value is None:
                        calculated_value = 0
                else:
                    # Para fórmulas que não são somas, usa o método antigo
                    for col in allowed_columns:
                        if col in formula:
                            formula_safe = formula_safe.replace(col, f"pd.to_numeric(df_cvm['{col}'], errors='coerce')")
                    
                    calculated_value = eval(formula_safe)
                
                df_normalized[target_col] = calculated_value
                
                logger.debug(f"  ✓ {target_col}: {formula} → {description}")
                
            except Exception as e:
                logger.warning(f"  ✗ {target_col}: Erro ao calcular {formula} - {e}")
                df_normalized[target_col] = 0  # Preenche com 0 em caso de erro
        
        # Preenche valores faltantes com interpolação temporal
        df_normalized = cls._fill_missing_values(df_normalized)
        
        logger.info(f"  ✓ CVM normalizado: {len(df_normalized)} registros, {len(df_normalized.columns)} colunas")
        
        return df_normalized
    
    @classmethod
    def _fill_missing_values(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preenche valores faltantes usando interpolação temporal por ticker.
        
        Args:
            df: DataFrame com dados normalizados
        
        Returns:
            DataFrame com valores faltantes preenchidos
        """
        # Indicadores que devem ser preenchidos
        indicators = ['Receita_Caixa', 'Lucro_Caixa_Trimestral', 'Liquidez_Caixa', 'Taxa_Administracao']
        
        # Ordena por ticker e data para garantir ordem correta
        df = df.sort_values(['ticker', 'data_referencia'])
        
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy()
            
            for indicator in indicators:
                if indicator in ticker_data.columns:
                    # Interpolação linear para valores faltantes
                    ticker_data[indicator] = ticker_data[indicator].interpolate(method='linear', limit_direction='both')
                    
                    # Se ainda houver NaN no início, preenche com o primeiro valor não nulo
                    first_valid = ticker_data[indicator].first_valid_index()
                    if first_valid is not None:
                        ticker_data[indicator] = ticker_data[indicator].bfill()
                        ticker_data[indicator] = ticker_data[indicator].ffill()
                    
                    # Preenche com 0 se ainda estiver vazio (caso de fundos sem histórico)
                    ticker_data[indicator].fillna(0, inplace=True)
            
            # Atualiza no dataframe original
            df.loc[df['ticker'] == ticker, indicators] = ticker_data[indicators]
        
        return df


class FundamentalsCollector:
    """
    Coletor de fundamentos de FIIs usando apenas dados da CVM.
    
    Responsável por:
    - Carregar metadata
    - Coletar dados da CVM
    - Normalizar e validar dados
    - Salvar resultado final
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Inicializa o coletor.
        
        Args:
            base_path: Caminho base do projeto (opcional)
        """
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent / "data"
        
        self.base_path = Path(base_path)
        self.output_path = self.base_path / "02_processed" / "fundamentals"
        self.metadata_path = self.base_path / "metadata"
        
        self._ensure_directories()
        self.metadata = self._load_metadata()
        self.fii_mapping = self._create_cnpj_mapping()
    
    def _ensure_directories(self):
        """Garante que diretórios existem."""
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def _load_metadata(self) -> Dict:
        """Carrega metadata dos FIIs."""
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
        """
        Cria mapeamento reverso CNPJ → Lista de Tickers.
        
        Returns:
            Dicionário {CNPJ: [Ticker1, Ticker2, ...]}
        """
        mapping = {}
        
        for fii in self.metadata['fiis']:
            cnpj = fii['cnpj']
            ticker = fii['ticker']
            
            if cnpj not in mapping:
                mapping[cnpj] = []
            mapping[cnpj].append(ticker)
        
        # Identifica CNPJs compartilhados
        shared = {cnpj: tickers for cnpj, tickers in mapping.items() if len(tickers) > 1}
        if shared:
            logger.info(f"✅ CNPJs compartilhados detectados: {shared}")
        
        logger.info(f"Mapeamento: {len(mapping)} CNPJs únicos → {len(self.metadata['fiis'])} FIIs")
        
        return mapping
    
    def _merge_trimestral_with_taxa_admin(
        self, 
        df_trimestral: pd.DataFrame, 
        df_mensal: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Une dados trimestrais com Taxa_Administracao mensal.
        
        Calcula média trimestral da taxa de administração a partir dos dados mensais.
        
        Args:
            df_trimestral: DataFrame com dados trimestrais
            df_mensal: DataFrame com dados mensais (inclui Taxa_Administracao)
        
        Returns:
            DataFrame trimestral com Taxa_Administracao agregada
        """
        logger.info("\nUnindo dados trimestrais com Taxa_Administracao mensal...")
        
        if df_trimestral.empty or df_mensal.empty or 'Taxa_Administracao' not in df_mensal.columns:
            logger.warning("Não foi possível unir dados - DataFrames vazios ou sem Taxa_Administracao")
            return df_trimestral.copy()
        
        # Agrega Taxa_Administracao por trimestre (média dos 3 meses)
        df_mensal_copy = df_mensal.copy()
        # Cria colunas trimestre e ano de uma vez para evitar fragmentação
        df_mensal_copy = df_mensal_copy.assign(
            trimestre=df_mensal_copy['data_referencia'].dt.quarter,
            ano=df_mensal_copy['data_referencia'].dt.year
        )
        
        # Calcula média da taxa de administração por FII, ano e trimestre
        taxa_trimestral = df_mensal_copy.groupby(['ticker', 'ano', 'trimestre'])['Taxa_Administracao'].mean().reset_index()
        taxa_trimestral.rename(columns={'Taxa_Administracao': 'Taxa_Administracao_Media'}, inplace=True)
        
        # Merge com dados trimestrais
        df_combined = df_trimestral.copy()
        df_combined = df_combined.merge(
            taxa_trimestral,
            on=['ticker', 'ano', 'trimestre'],
            how='left'
        )
        
        # Se a coluna já existia, substitui; se não, cria
        if 'Taxa_Administracao' in df_combined.columns:
            df_combined['Taxa_Administracao'] = df_combined['Taxa_Administracao_Media'].fillna(df_combined['Taxa_Administracao'])
        else:
            df_combined['Taxa_Administracao'] = df_combined['Taxa_Administracao_Media']
        
        # Remove coluna auxiliar
        df_combined = df_combined.drop(columns=['Taxa_Administracao_Media'], errors='ignore')
        
        logger.info(f"✓ Dados unidos: {len(df_combined)} registros trimestrais")
        
        return df_combined
    
    def collect_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Executa coleta completa da CVM (trimestral e mensal).
        
        Returns:
            Tupla (df_trimestral, df_mensal, df_combined) normalizados e validados
        """
        logger.info("\n" + "="*80)
        logger.info(" "*20 + "COLETA DE FUNDAMENTOS - FII-O (CVM)")
        logger.info("="*80)
        
        try:
            # 1. Coleta CVM Trimestral
            cvm_source = CVMDataSource(self.fii_mapping)
            df_cvm_trim = cvm_source.collect()
            
            if df_cvm_trim.empty:
                logger.error("❌ Coleta CVM trimestral resultou em DataFrame vazio")
                df_normalized_trim = pd.DataFrame()
            else:
                # 2. Normaliza dados trimestrais
                df_normalized_trim = DataIntegrator.normalize_cvm_data(df_cvm_trim)
                
                # Remove Taxa_Administracao se existir nos dados trimestrais (será adicionada depois)
                if 'Taxa_Administracao' in df_normalized_trim.columns:
                    df_normalized_trim = df_normalized_trim.drop(columns=['Taxa_Administracao'])
                
                if df_normalized_trim.empty:
                    logger.error("❌ Normalização trimestral resultou em DataFrame vazio")
            
            # 4. Coleta CVM Mensal
            cvm_mensal_source = CVMMensalDataSource(self.fii_mapping)
            df_cvm_mensal = cvm_mensal_source.collect()
            
            if df_cvm_mensal.empty:
                logger.warning("⚠️  Coleta CVM mensal resultou em DataFrame vazio")
                df_normalized_mensal = pd.DataFrame()
            else:
                # 5. Normaliza dados mensais (usa mesmo schema por enquanto)
                df_normalized_mensal = DataIntegrator.normalize_cvm_data(df_cvm_mensal)
                
                if df_normalized_mensal.empty:
                    logger.warning("⚠️  Normalização mensal resultou em DataFrame vazio")
                    df_normalized_mensal = pd.DataFrame()
            
            # 6. Une dados trimestrais com Taxa_Administracao mensal
            df_combined = self._merge_trimestral_with_taxa_admin(df_normalized_trim, df_normalized_mensal)
            
            # 7. Salva apenas o arquivo final consolidado
            self._save_to_parquet(df_combined, "fundamentals_trimestral.parquet")
            
            return df_combined
            
        except Exception as e:
            logger.error(f"\n❌ Erro durante a coleta: {e}")
            raise
    
    def _save_to_parquet(self, df: pd.DataFrame, filename: str = "fundamentals_trimestral.parquet"):
        """
        Salva DataFrame normalizado em Parquet.
        
        Args:
            df: DataFrame normalizado
            filename: Nome do arquivo
        """
        if df.empty:
            logger.warning(f"DataFrame vazio, nada para salvar em {filename}")
            return
        
        # Validação do esquema antes de salvar
        expected_schema = set(DataIntegrator.TARGET_SCHEMA.keys())
        schema_to_use = DataIntegrator.TARGET_SCHEMA
        
        actual_schema = set(df.columns)
        
        if expected_schema != actual_schema:
            logger.warning(f"⚠️  Esquema não corresponde ao alvo:")
            logger.warning(f"  Esperado: {sorted(expected_schema)}")
            logger.warning(f"  Atual: {sorted(actual_schema)}")
        
        # Garante tipos de dados corretos
        for col, expected_type in schema_to_use.items():
            if col in df.columns:
                if expected_type == 'datetime':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                elif expected_type == 'float':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif expected_type == 'int':
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                elif expected_type == 'str':
                    df[col] = df[col].astype(str)
        
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
    """Função principal para execução do script."""
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