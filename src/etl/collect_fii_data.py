"""
Coleta de Dados de Fundos de Investimento Imobiliário (FII) da B3
===================================================================

Este script coleta dados históricos de FIIs brasileiros através da API do Yahoo Finance
e dados adicionais via Fundamentus (web scraping).

Funcionalidades:
- Lista de FIIs conhecidos com mapeamento Ticker-CNPJ
- Coleta de séries temporais de preços (2019-2025)
- Preço diário ajustado (preço relevante para o investidor)
- Volume de negociação (liquidez diária)
- Histórico de proventos (dividendos)
- Cálculo de DY (Dividend Yield) de 12 meses
- Dados do Fundamentus: VP/cota, P/VP, Vacância, Tipo Gestão
- Tratamento automático de ajustes (splits e agrupamentos)
- Salvamento otimizado em formato Parquet

Dados coletados:
- Preços diários: ajustados (preço que o investidor acompanha)
- Volume: quantidade de cotas negociadas
- Proventos: histórico completo de dividendos distribuídos
- DY de 12 meses: calculado automaticamente
- CNPJ, Nome e Tipo do fundo
- VP/cota, P/VP, Vacância (atualizado diariamente)
- Taxa de Administração coletada via CVM (mensal)
- Período: Janeiro/2019 até Janeiro/2025
"""

import pandas as pd
import yfinance as yf
import time
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional
import requests
from bs4 import BeautifulSoup
import json

class FIIDataCollector:
    """Classe para coleta de dados de FIIs"""
    
    # Datas padrão para coleta de dados históricos
    DEFAULT_START_DATE = '2019-01-01'
    DEFAULT_END_DATE = '2025-01-01'
    
    # URL para scraping
    FUNDAMENTUS_BASE_URL = "https://www.fundamentus.com.br/detalhes.php"
    
    def __init__(self, log_level: int = logging.INFO):
        """
        Inicializa o coletor de dados de FIIs.
        
        Args:
            log_level: Nível de logging (padrão: logging.INFO)
        """
        self._setup_logger(log_level)
        self._initialize_paths()
        
    def _setup_logger(self, level: int):
        """Configura logger específico da classe."""
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(level)
    
    def _initialize_paths(self):
        """Inicializa os caminhos do projeto."""
        project_root = Path(__file__).parent.parent.parent
        self.base_path = project_root / "data"
        self.raw_path = self.base_path / "01_raw"
        self.metadata_path = self.base_path / "metadata"
            
    def load_fiis_list(self) -> pd.DataFrame:
        """
        Carrega lista de FIIs do arquivo de metadata centralizado.
        
        Returns:
            DataFrame com informações dos FIIs (ticker, nome, cnpj, tipo)
        
        Raises:
            FileNotFoundError: Se arquivo de metadata não existir
        """
        fiis_data = self._load_fiis_from_metadata()
        df = pd.DataFrame(fiis_data)
        df = df.drop_duplicates(subset=['ticker'])
        df = df.sort_values('ticker')
        self.logger.info(f'Carregados {len(df)} FIIs do metadata')
        return df
    
    def _load_fiis_from_metadata(self) -> list:
        """
        Carrega dados dos FIIs do arquivo metadata centralizado.
        
        Returns:
            Lista de dicionários com dados dos FIIs
        
        Raises:
            FileNotFoundError: Se arquivo não existir
            json.JSONDecodeError: Se arquivo JSON estiver malformado
        """
        metadata_path = self.metadata_path / "fiis_metadata.json"
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            self.logger.info(f"Metadata carregado de: {metadata_path}")
            return metadata['fiis']
        except FileNotFoundError:
            self.logger.error(f"Arquivo de metadata não encontrado: {metadata_path}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Erro ao decodificar JSON: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Erro ao carregar metadata: {e}")
            raise
    
    def collect_all_data(
        self, 
        fiis_df: pd.DataFrame, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Coleta preços e proventos para os FIIs especificados.
        
        Args:
            fiis_df: DataFrame com colunas 'ticker', 'nome', 'cnpj'
            start_date: Data de início no formato 'YYYY-MM-DD' (padrão: 2019-01-01)
            end_date: Data de fim no formato 'YYYY-MM-DD' (padrão: 2025-01-01)
        
        Returns:
            Tupla (prices_df, dividends_df) com os dados coletados
        
        Raises:
            ValueError: Se fiis_df estiver vazio ou sem colunas requeridas
            TypeError: Se fiis_df não for DataFrame
        """
        # Validações de entrada
        self._validate_fiis_dataframe(fiis_df)
        
        if start_date is None:
            start_date = self.DEFAULT_START_DATE
        if end_date is None:
            end_date = self.DEFAULT_END_DATE
        
        self.logger.info(f"Coletando dados completos: {len(fiis_df)} FIIs")
        
        # Cria dicionários de mapeamento
        mappings = self._create_ticker_mappings(fiis_df)
        
        all_prices = []
        all_dividends = []
        failed_tickers = []
        
        tickers = fiis_df['ticker'].tolist()
        
        for i, ticker in enumerate(tickers, 1):
            self.logger.info(f"[{i}/{len(tickers)}] {ticker}")
            
            price_data, div_data = self._collect_ticker_data(
                ticker, mappings, start_date, end_date
            )
            
            if price_data is not None:
                all_prices.append(price_data)
                if div_data is not None:
                    all_dividends.append(div_data)
            else:
                failed_tickers.append(ticker)
            
            time.sleep(0.5)
        
        # Enriquece dados de preços com dados do Fundamentus
        if all_prices:
            self.logger.info("Enriquecendo dados com Fundamentus...")
            prices_df = pd.concat(all_prices, ignore_index=True)
            prices_df = self._enrich_with_fundamentus(prices_df, fiis_df)
            all_prices = [prices_df]
        
        return self._consolidate_results(all_prices, all_dividends, failed_tickers)
    
    def _validate_fiis_dataframe(self, fiis_df: pd.DataFrame):
        """
        Valida o DataFrame de FIIs.
        
        Args:
            fiis_df: DataFrame a ser validado
        
        Raises:
            TypeError: Se não for DataFrame
            ValueError: Se estiver vazio ou faltar colunas
        """
        if not isinstance(fiis_df, pd.DataFrame):
            raise TypeError("fiis_df deve ser um pandas DataFrame")
        
        if fiis_df.empty:
            raise ValueError("fiis_df não pode estar vazio")
        
        required_cols = {'ticker', 'nome', 'cnpj'}
        if not required_cols.issubset(fiis_df.columns):
            missing = required_cols - set(fiis_df.columns)
            raise ValueError(f"fiis_df falta colunas: {missing}")
    
    def _create_ticker_mappings(self, fiis_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Cria dicionários de mapeamento ticker -> info.
        
        Args:
            fiis_df: DataFrame com dados dos FIIs
        
        Returns:
            Dicionário com mapeamentos 'cnpj' e 'nome'
        """
        return {
            'cnpj': dict(zip(fiis_df['ticker'], fiis_df['cnpj'])),
            'nome': dict(zip(fiis_df['ticker'], fiis_df['nome']))
        }
    
    def _collect_ticker_data(
        self, 
        ticker: str, 
        mappings: Dict, 
        start_date: str, 
        end_date: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Coleta dados de um ticker específico.
        
        Args:
            ticker: Código do ticker
            mappings: Dicionários de mapeamento
            start_date: Data de início
            end_date: Data de fim
        
        Returns:
            Tupla (price_data, dividend_data) ou (None, None) se falhar
        """
        try:
            yf_ticker = f"{ticker}.SA"
            stock = yf.Ticker(yf_ticker)
            
            # Coleta preços ajustados (auto_adjust=True já inclui ajustes)
            hist_adj = stock.history(start=start_date, end=end_date, auto_adjust=True)
            
            if hist_adj.empty:
                self.logger.warning(f"Nenhum dado encontrado para {ticker}")
                return None, None
            
            # Processa dados (apenas preço ajustado, que é o relevante)
            price_data = self._process_price_data(hist_adj, ticker, mappings)
            div_data = self._process_dividend_data(
                stock.dividends, ticker, mappings, start_date, end_date
            )
            
            return price_data, div_data
            
        except Exception as e:
            self.logger.error(f"Erro ao coletar dados para {ticker}: {e}")
            return None, None
    
    def _process_price_data(
        self, 
        hist_adj: pd.DataFrame, 
        ticker: str, 
        mappings: Dict
    ) -> pd.DataFrame:
        """
        Processa dados de preços ajustados (preço relevante para o investidor).
        
        Args:
            hist_adj: DataFrame com preços ajustados
            ticker: Código do ticker
            mappings: Dicionários de mapeamento
        
        Returns:
            DataFrame processado com preços
        """
        # Processa preços ajustados
        hist_adj = hist_adj.reset_index()
        hist_adj['Date'] = pd.to_datetime(hist_adj['Date'])
        hist_adj['ticker'] = ticker
        hist_adj.columns = [col.lower() for col in hist_adj.columns]
        hist_adj.rename(columns={'date': 'data'}, inplace=True)
        hist_adj['preco_ajustado'] = hist_adj['close']
        
        # Cria DataFrame com dados principais (preço ajustado, que é o relevante)
        combined = hist_adj[['data', 'ticker', 'preco_ajustado', 'volume']].copy()
        combined['cnpj'] = mappings['cnpj'].get(ticker)
        combined['nome'] = mappings['nome'].get(ticker)
        
        # Garante ordenação
        if not combined['data'].is_monotonic_increasing:
            combined = combined.sort_values('data')
        
        return combined
    
    def _process_dividend_data(
        self, 
        dividends: pd.Series, 
        ticker: str, 
        mappings: Dict, 
        start_date: str, 
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Processa dados de dividendos.
        
        Args:
            dividends: Série com dividendos
            ticker: Código do ticker
            mappings: Dicionários de mapeamento
            start_date: Data de início
            end_date: Data de fim
        
        Returns:
            DataFrame com dividendos ou None se vazio
        """
        if dividends.empty:
            return None
        
        dividends_df = dividends.reset_index()
        dividends_df.columns = ['data', 'dividendo']
        dividends_df['ticker'] = ticker
        dividends_df['cnpj'] = mappings['cnpj'].get(ticker)
        dividends_df['nome'] = mappings['nome'].get(ticker)
        dividends_df['data'] = pd.to_datetime(dividends_df['data']).dt.tz_localize(None)
        
        # Filtra por período
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        dividends_df = dividends_df[
            (dividends_df['data'] >= start_dt) & 
            (dividends_df['data'] <= end_dt)
        ]
        
        return dividends_df if not dividends_df.empty else None
    
    def _consolidate_results(
        self, 
        all_prices: list, 
        all_dividends: list, 
        failed_tickers: list
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Consolida resultados finais da coleta.
        
        Args:
            all_prices: Lista de DataFrames com preços
            all_dividends: Lista de DataFrames com dividendos
            failed_tickers: Lista de tickers que falharam
        
        Returns:
            Tupla (prices_df, dividends_df)
        """
        # Consolida preços
        if all_prices:
            prices_df = pd.concat(all_prices, ignore_index=True)
            prices_df = prices_df.drop_duplicates(subset=['ticker', 'data'])
            prices_df = prices_df.sort_values(['ticker', 'data'])
            self.logger.info(f"Preços: {len(all_prices)} sucessos, {len(failed_tickers)} falhas")
        else:
            prices_df = pd.DataFrame()
            self.logger.error("Nenhum dado de preços foi coletado")
        
        # Consolida dividendos
        if all_dividends:
            dividends_df = pd.concat(all_dividends, ignore_index=True)
            dividends_df = dividends_df.drop_duplicates(subset=['ticker', 'data'])
            dividends_df = dividends_df.sort_values(['ticker', 'data'])
            self.logger.info(f"Proventos: {len(dividends_df)} registros")
        else:
            dividends_df = pd.DataFrame()
            self.logger.warning("Nenhum dado de proventos foi coletado")
        
        # Calcula DY de 12 meses se temos dividendos e preços
        if not prices_df.empty and not dividends_df.empty:
            prices_df = self._calculate_dy_12_months(prices_df, dividends_df)
        
        return prices_df, dividends_df
    
    def _calculate_dy_12_months(
        self, 
        prices_df: pd.DataFrame, 
        dividends_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calcula Dividend Yield (DY) de 12 meses para cada data.
        
        Args:
            prices_df: DataFrame com preços
            dividends_df: DataFrame com dividendos
        
        Returns:
            DataFrame com coluna dy_12m adicionada
        """
        self.logger.info("Calculando DY de 12 meses...")
        
        # Cria cópia para não alterar o original
        prices_with_dy = prices_df.copy()
        prices_with_dy['dy_12m'] = 0.0
        
        for ticker in prices_df['ticker'].unique():
            ticker_prices = prices_with_dy[prices_with_dy['ticker'] == ticker].copy()
            ticker_dividends = dividends_df[dividends_df['ticker'] == ticker].copy()
            
            if ticker_dividends.empty:
                continue
            
            # Ordena por data
            ticker_prices = ticker_prices.sort_values('data').reset_index(drop=True)
            ticker_dividends = ticker_dividends.sort_values('data').reset_index(drop=True)
            
            # Remove timezone se houver para facilitar comparação
            if ticker_prices['data'].dt.tz is not None:
                ticker_prices['data'] = ticker_prices['data'].dt.tz_localize(None)
            if ticker_dividends['data'].dt.tz is not None:
                ticker_dividends['data'] = ticker_dividends['data'].dt.tz_localize(None)
            
            # Para cada data de preço, calcula soma de dividendos dos últimos 12 meses
            dy_values = []
            for idx, row in ticker_prices.iterrows():
                current_date = pd.to_datetime(row['data'])
                date_12m_ago = current_date - pd.DateOffset(months=12)
                
                # Filtra dividendos dos últimos 12 meses
                dividends_last_12m = ticker_dividends[
                    (ticker_dividends['data'] >= date_12m_ago) & 
                    (ticker_dividends['data'] <= current_date)
                ]
                
                # Soma os dividendos
                total_dividends = dividends_last_12m['dividendo'].sum()
                
                # Calcula DY: (soma dividendos últimos 12 meses / preço atual) * 100
                if row['preco_ajustado'] > 0:
                    dy = (total_dividends / row['preco_ajustado']) * 100
                else:
                    dy = 0.0
                
                dy_values.append(dy)
            
            # Atualiza valores de DY
            ticker_mask = prices_with_dy['ticker'] == ticker
            prices_with_dy.loc[ticker_mask, 'dy_12m'] = dy_values
        
        self.logger.info(f"DY de 12 meses calculado para {prices_with_dy['ticker'].nunique()} FIIs")
        
        return prices_with_dy
    
    def _enrich_with_fundamentus(
        self, 
        prices_df: pd.DataFrame, 
        fiis_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Enriquece dados com informações do Fundamentus.
        
        Args:
            prices_df: DataFrame com preços
            fiis_df: DataFrame com informações dos FIIs
        
        Returns:
            DataFrame enriquecido
        """
        # Coleta dados do Fundamentus para cada FII único
        tickers_unique = prices_df['ticker'].unique()
        fundamentus_data = {}
        
        for ticker in tickers_unique:
            try:
                data = self._scrape_fundamentus(ticker)
                if data:
                    fundamentus_data[ticker] = data
                time.sleep(1)  # Rate limiting
            except Exception as e:
                self.logger.warning(f"Erro ao coletar Fundamentus para {ticker}: {e}")
        
        if not fundamentus_data:
            self.logger.warning("Nenhum dado do Fundamentus foi coletado")
            return prices_df
        
        # Adiciona colunas de Fundamentus ao DataFrame de preços
        for col in ['vp_cota', 'pvp', 'vacancia', 'num_cotas', 'tipo_gestao']:
            prices_df[col] = None
        
        # Preenche dados do Fundamentus para cada linha (cada data)
        for ticker in fundamentus_data:
            ticker_mask = prices_df['ticker'] == ticker
            data = fundamentus_data[ticker]
            
            for col, value in data.items():
                if col != 'ticker' and col in prices_df.columns:
                    prices_df.loc[ticker_mask, col] = value
        
        self.logger.info(f"Dados Fundamentus adicionados para {len(fundamentus_data)} FIIs")
        return prices_df
    
    def _scrape_fundamentus(self, ticker: str) -> Optional[Dict]:
        """
        Faz scraping dos dados do Fundamentus para um FII.
        
        Args:
            ticker: Código do ticker
        
        Returns:
            Dicionário com dados do FII ou None
        """
        url = f"{self.FUNDAMENTUS_BASE_URL}?papel={ticker}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            data = {
                'ticker': ticker,
                'vp_cota': None,
                'pvp': None,
                'vacancia': None,
                'num_cotas': None,
                'tipo_gestao': None,
            }
            
            tables = soup.find_all('table')
            
            for table in tables:
                rows = table.find_all('tr')
                
                for row in rows:
                    cells = row.find_all('td')
                    
                    # Processa células em pares: label -> value
                    # A estrutura pode ter múltiplas células: <td label>texto</td><td data>valor</td>...
                    for i in range(len(cells) - 1):
                        # Verifica se a célula atual é um label (tem span.txt)
                        label_cell = cells[i]
                        value_cell = cells[i + 1]
                        
                        label_span = label_cell.find('span', class_='txt')
                        value_span = value_cell.find('span', class_='txt')
                        
                        if label_span and value_span:
                            label_text = label_span.get_text(strip=True).lower()
                            value_text = value_span.get_text(strip=True)
                            
                            # Busca case-insensitive
                            if 'valor patrimonial' in label_text or 'vp/cota' in label_text:
                                data['vp_cota'] = self._parse_fundamentus_value(value_text)
                            elif 'p/vp' in label_text:
                                data['pvp'] = self._parse_fundamentus_value(value_text)
                            elif 'vacância' in label_text or 'vacancia' in label_text:
                                data['vacancia'] = self._parse_fundamentus_percentage(value_text)
                            elif 'nro. cotas' in label_text or 'número de cotas' in label_text or 'numero de cotas' in label_text or 'quantidade de cotas' in label_text:
                                data['num_cotas'] = self._parse_fundamentus_value(value_text, is_integer=True)
                            elif 'tipo de gestão' in label_text or 'tipo de gestao' in label_text or 'gestão' in label_text:
                                # Tipo de gestão pode ser: Ativa, Passiva, Ativo-Passivo, etc.
                                data['tipo_gestao'] = value_text.strip()
            
            return data
            
        except Exception as e:
            self.logger.debug(f"Erro no scraping Fundamentus para {ticker}: {e}")
            return None
    
    def _parse_fundamentus_value(self, value_str: str, is_integer: bool = False) -> Optional[float]:
        """Parse de valores numéricos do Fundamentus."""
        if not value_str or value_str == '-':
            return None
        
        try:
            cleaned = value_str.replace('.', '').replace(',', '.').strip()
            for char in ['R$', '$', '%', ' ']:
                cleaned = cleaned.replace(char, '')
            
            if cleaned:
                return int(float(cleaned)) if is_integer else float(cleaned)
        except:
            pass
        
        return None
    
    def _parse_fundamentus_percentage(self, value_str: str) -> Optional[float]:
        """Parse de valores percentuais do Fundamentus."""
        if not value_str or value_str == '-':
            return None
        
        try:
            cleaned = value_str.replace('%', '').replace(',', '.').strip()
            if cleaned:
                return float(cleaned) / 100
        except:
            pass
        
        return None
    
    def save_price_data(
        self, 
        prices_df: pd.DataFrame, 
        dividends_df: Optional[pd.DataFrame] = None
    ):
        """
        Salva dados de preços e dividendos em formato Parquet.
        
        Args:
            prices_df: DataFrame com dados de preços
            dividends_df: DataFrame opcional com dados de dividendos
        
        Raises:
            IOError: Se houver erro ao salvar os arquivos
        """
        try:
            prices_path = self.raw_path / "prices.parquet"
            prices_df.to_parquet(prices_path, index=False, engine='pyarrow')
            self.logger.info(f"Preços salvos: {prices_path}")
            
            if dividends_df is not None and not dividends_df.empty:
                dividends_path = self.raw_path / "dividends.parquet"
                dividends_df.to_parquet(dividends_path, index=False, engine='pyarrow')
                self.logger.info(f"Proventos salvos: {dividends_path}")
        except Exception as e:
            self.logger.error(f"Erro ao salvar dados: {e}")
            raise IOError(f"Falha ao salvar arquivos Parquet: {e}")


if __name__ == "__main__":
    collector = FIIDataCollector()
    fiis_df = collector.load_fiis_list()
    prices_df, dividends_df = collector.collect_all_data(fiis_df)
    collector.save_price_data(prices_df, dividends_df)
    collector.logger.info(f"Concluído: {len(fiis_df)} FIIs, {len(prices_df)} preços, {len(dividends_df)} proventos")