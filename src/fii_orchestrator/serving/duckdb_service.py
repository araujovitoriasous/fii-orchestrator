import duckdb
from pathlib import Path
from loguru import logger

class DuckDBService:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or Path("./data")
        self.connection = None
        self._setup_connection()

    def _setup_connection(self):
        try:
            self.connection = duckdb.connect(":memory:")
            self._register_parquet_files()
            logger.info("✅ Conexão DuckDB estabelecida com dados Parquet")
        except Exception as e:
            logger.error(f"❌ Erro ao conectar DuckDB: {e}")
            raise

    def _register_parquet_files(self):
        """Registra arquivos Parquet como tabelas virtuais."""
        try:
            # Registrar diretório de fundos
            funds_dir = self.data_dir / "bronze" / "funds"
            if funds_dir.exists():
                self.connection.execute(f"CREATE VIEW funds AS SELECT * FROM read_parquet('{funds_dir}/*.parquet')")
                logger.info(f"✅ Tabela virtual 'funds' criada com dados de {funds_dir}")
            
            # Registrar diretório de preços
            prices_dir = self.data_dir / "bronze" / "prices"
            if prices_dir.exists():
                self.connection.execute(f"CREATE VIEW prices AS SELECT * FROM read_parquet('{prices_dir}/*.parquet')")
                logger.info(f"✅ Tabela virtual 'prices' criada com dados de {prices_dir}")
            
            # Registrar diretório de notícias
            news_dir = self.data_dir / "bronze" / "news"
            if news_dir.exists():
                self.connection.execute(f"CREATE VIEW news AS SELECT * FROM read_parquet('{prices_dir}/*.parquet')")
                logger.info(f"✅ Tabela virtual 'news' criada com dados de {news_dir}")
                
        except Exception as e:
            logger.warning(f"⚠️ Erro ao registrar arquivos Parquet: {e}")

    def get_funds(self, limit: int = 100, offset: int = 0):
        """Busca fundos com paginação."""
        try:
            query = f"""
                SELECT DISTINCT ticker, cnpj, razao_social, fonte, data_coleta
                FROM funds 
                ORDER BY ticker 
                LIMIT {limit} OFFSET {offset}
            """
            result = self.connection.execute(query)
            return result.fetchall()
        except Exception as e:
            logger.error(f"❌ Erro ao buscar fundos: {e}")
            return []

    def get_prices(self, ticker: str = None, limit: int = 100, offset: int = 0):
        """Busca preços com paginação e filtro por ticker."""
        try:
            if ticker:
                query = f"""
                    SELECT ticker, price, date, volume, fonte
                    FROM prices 
                    WHERE ticker = '{ticker}'
                    ORDER BY date DESC 
                    LIMIT {limit} OFFSET {offset}
                """
            else:
                query = f"""
                    SELECT ticker, price, date, volume, fonte
                    FROM prices 
                    ORDER BY date DESC, ticker
                    LIMIT {limit} OFFSET {offset}
                """
            result = self.connection.execute(query)
            return result.fetchall()
        except Exception as e:
            logger.error(f"❌ Erro ao buscar preços: {e}")
            return []

    def execute_query(self, query, params=None):
        try:
            if params:
                result = self.connection.execute(query, params)
            else:
                result = self.connection.execute(query)
            return result.fetchall()
        except Exception as e:
            logger.error(f"❌ Erro ao executar query: {e}")
            raise

    def close(self):
        if self.connection:
            self.connection.close()
            logger.info("✅ Conexão DuckDB fechada")
