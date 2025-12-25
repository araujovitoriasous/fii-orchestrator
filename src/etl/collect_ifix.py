"""
Coleta de Benchmark (IFIX) via Arquivos CSV
===========================================
Lê os arquivos de evolução diária do IFIX (ifix-YYYY.csv) que estão formatados
como uma matriz (Dia x Mês) e os converte para uma série temporal (Data, Fechamento).
"""

import pandas as pd
import logging
from pathlib import Path
import re

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class B3IfixCollector:
    
    def __init__(self):
        # Define caminhos
        self.base_path = Path(__file__).resolve().parents[2] / "data"
        self.input_path = self.base_path / "01_raw" / "reports_ifix" / "evolucao_diaria"
        self.output_path = self.base_path / "01_raw" / "benchmark-ifix.parquet"
        
        # Mapeamento de meses (cabeçalho do CSV) para números
        self.month_map = {
            'Jan': 1, 'Fev': 2, 'Mar': 3, 'Abr': 4, 'Mai': 5, 'Jun': 6,
            'Jul': 7, 'Ago': 8, 'Set': 9, 'Out': 10, 'Nov': 11, 'Dez': 12
        }

    def parse_value(self, val):
        """Converte string '3.320,47' para float 3320.47"""
        if pd.isna(val) or val == '':
            return None
        
        if isinstance(val, (int, float)):
            return float(val)
            
        try:
            # Remove pontos de milhar e troca vírgula por ponto
            clean_val = str(val).replace('.', '').replace(',', '.')
            return float(clean_val)
        except ValueError:
            return None

    def process_csv(self, file_path: Path) -> pd.DataFrame:
        """Lê um CSV anual e transforma em série temporal."""
        logger.info(f"Processando: {file_path.name}...")
        
        try:
            # Extrai ano do nome do arquivo (ifix-2024.csv)
            match = re.search(r'ifix-(\d{4})', file_path.name)
            if not match:
                logger.warning(f"  -> Ignorado: Ano não identificado no arquivo {file_path.name}")
                return pd.DataFrame()
            year = int(match.group(1))

            # Lê o CSV (delimitador ; e encoding possivelmente latin1 ou utf8)
            # O cabeçalho é: Dia;Jan;Fev;...
            # Pula a primeira linha que contém o título "IFIX - 20XX"
            df = pd.read_csv(file_path, sep=';', encoding='latin1', thousands='.', decimal=',', skiprows=1)
            
            # Normalizar colunas (strip whitespace)
            df.columns = [c.strip() for c in df.columns]

            # Lista para guardar os registros
            records = []

            # Itera sobre as linhas (dias)
            for _, row in df.iterrows():
                try:
                    day = int(row.get('Dia'))
                except (ValueError, TypeError):
                    continue # Pula linhas onde 'Dia' não é numérico (rodapés, vazios)

                # Itera sobre as colunas de meses
                for col_name, month_num in self.month_map.items():
                    if col_name in df.columns:
                        val = row[col_name]
                        price = self.parse_value(val)
                        
                        if price is not None:
                            # Cria a data
                            try:
                                date = pd.Timestamp(year=year, month=month_num, day=day)
                                records.append({
                                    'data': date,
                                    'fechamento': price
                                })
                            except ValueError:
                                # Data inválida (ex: 31 de Fevereiro), ignora
                                pass
            
            return pd.DataFrame(records)

        except Exception as e:
            logger.error(f"Erro ao processar {file_path}: {e}")
            return pd.DataFrame()

    def run(self):
        all_data = []
        
        # Busca todos os CSVS ifix-*.csv
        if not self.input_path.exists():
            logger.error(f"Diretório não encontrado: {self.input_path}")
            return

        csv_files = sorted(list(self.input_path.glob("ifix-*.csv")))
        
        if not csv_files:
            logger.warning(f"Nenhum arquivo 'ifix-*.csv' encontrado em {self.input_path}")
            return

        for csv_file in csv_files:
            df_year = self.process_csv(csv_file)
            if not df_year.empty:
                all_data.append(df_year)
                logger.info(f"  -> {len(df_year)} dias extraídos.")

        if all_data:
            df_final = pd.concat(all_data).sort_values('data').drop_duplicates(subset=['data'])
            
            # Salvar
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            df_final.to_parquet(self.output_path, index=False)
            
            logger.info(f"\n✅ SUCESSO! Histórico IFIX gerado.")
            logger.info(f"   Período: {df_final['data'].min().date()} até {df_final['data'].max().date()}")
            logger.info(f"   Total Dias: {len(df_final)}")
            logger.info(f"   Arquivo final: {self.output_path}")
        else:
            logger.warning("Nenhum dado extraído.")

if __name__ == "__main__":
    c = B3IfixCollector()
    c.run()
