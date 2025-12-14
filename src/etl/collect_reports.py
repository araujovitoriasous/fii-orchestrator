"""
Coleta de Documentos Financeiros de FIIs (DFIN) - CVM
=====================================================
Baseado na estrutura confirmada do Portal de Dados Abertos:
URL: https://dados.cvm.gov.br/dados/FII/DOC/DFIN/DADOS/
Arquivo: dfin_fii_YYYY.csv (CSV direto, sem ZIP)

Este módulo baixa as Demonstrações Financeiras (DFP/ITR) e extrai o texto 
dos PDFs disponíveis na coluna 'Link_Download'.
"""

import pandas as pd
import requests
import io
import logging
import json
import pdfplumber
import time
import base64
from pathlib import Path
from typing import List

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReportsCollector:
    # URL Base confirmada pelos prints e arquivos
    CVM_BASE_URL = "https://dados.cvm.gov.br/dados/FII/DOC/DFIN/DADOS/"
    
    def __init__(self):
        # Define caminho base relativo à raiz do projeto
        self.base_path = Path(__file__).resolve().parents[2] / "data"
        self.raw_reports_path = self.base_path / "01_raw" / "reports_text"
        self.raw_reports_path.mkdir(parents=True, exist_ok=True)
        
        self.metadata_path = self.base_path / "metadata" / "fiis_metadata.json"
        self.fiis_metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Erro ao carregar metadata: {e}")
            return {"fiis": []}

    def _get_target_cnpjs(self) -> List[str]:
        # Normaliza CNPJ (remove pontuação) para bater com o CSV da CVM
        return [fii['cnpj'].replace('.', '').replace('/', '').replace('-', '') 
                for fii in self.fiis_metadata['fiis']]

    def download_registry(self, year: int) -> pd.DataFrame:
        """
        Baixa o CSV de índice diretamente (sem ZIP).
        Padrão: dfin_fii_{ANO}.csv
        """
        filename = f"dfin_fii_{year}.csv"
        url = f"{self.CVM_BASE_URL}{filename}"
        
        logger.info(f"Baixando índice DFIN: {filename}...")
        try:
            # Timeout alto pois o servidor da CVM oscila
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            
            # Lê o CSV direto da memória (encoding padrão CVM é ISO-8859-1 ou cp1252)
            df = pd.read_csv(io.BytesIO(r.content), sep=';', encoding='ISO-8859-1')
            return df
            
        except Exception as e:
            logger.error(f"Erro ao baixar {url}: {e}")
            return pd.DataFrame()

    def extract_text_from_pdf_url(self, url: str) -> str:
        """Baixa e extrai texto do PDF (primeiras páginas)."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Correção de protocolo http -> https se necessário
            if url.startswith('http:'):
                url = url.replace('http:', 'https:')

            r = requests.get(url, headers=headers, timeout=20)
            
            if r.status_code != 200: 
                logger.warning(f"  Erro download PDF ({r.status_code}): {url}")
                return ""

            content = r.content
            # Tratamento para Base64 (CVM as vezes retorna assim)
            if content.startswith(b'"JVBERi'):
                try:
                    # Remove as aspas e decode
                    content_str = content.strip(b'"')
                    content = base64.b64decode(content_str)
                except Exception as e:
                    logger.warning(f"  Falha ao decodificar Base64: {e}")

            text_content = []
            try:
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    # Lê apenas as primeiras 15 páginas para otimizar
                    # (O relatório da administração costuma estar no início)
                    for i, page in enumerate(pdf.pages):
                        if i >= 15: break 
                        text = page.extract_text()
                        if text: text_content.append(text)
            except Exception as e:
                logger.warning(f"  Erro ao ler PDF (pode estar corrompido): {e}")
                return ""
            
            return "\n".join(text_content)
        except Exception as e:
            logger.error(f"  Erro de conexão ao baixar PDF: {e}")
            return ""

    def run(self, start_year: int = 2024, end_year: int = 2025):
        target_cnpjs = self._get_target_cnpjs()
        logger.info(f"Iniciando coleta DFIN para {len(target_cnpjs)} FIIs...")
        
        all_reports = []

        for year in range(start_year, end_year + 1):
            df_reg = self.download_registry(year)
            
            if df_reg.empty:
                logger.warning(f"Índice de {year} vazio ou não encontrado.")
                continue

            # 1. Normaliza colunas para evitar erros de Case Sensitivity
            df_reg.columns = [c.upper().strip() for c in df_reg.columns]
            
            # 2. Mapeamento Inteligente das Colunas (Adaptado ao seu CSV)
            # Tenta encontrar o nome certo da coluna baseado no padrão
            col_cnpj = next((c for c in df_reg.columns if 'CNPJ' in c), None)
            col_link = next((c for c in df_reg.columns if 'LINK' in c), None)
            col_data_ref = next((c for c in df_reg.columns if 'REFER' in c), None)
            col_data_ent = next((c for c in df_reg.columns if 'ENTREGA' in c or 'RECEB' in c), None)

            if not col_cnpj or not col_link:
                logger.error(f"Colunas essenciais (CNPJ/LINK) não encontradas. Colunas: {df_reg.columns}")
                continue

            # 3. Limpeza e Filtro
            # Remove pontuação do CNPJ no DataFrame para bater com o metadata
            df_reg['cnpj_limpo'] = df_reg[col_cnpj].astype(str).str.replace(r'[^\d]', '', regex=True)
            
            # Filtra apenas os FIIs do nosso portfólio
            df_target = df_reg[df_reg['cnpj_limpo'].isin(target_cnpjs)].copy()
            
            # Ordena pelos mais recentes
            if col_data_ref:
                df_target = df_target.sort_values(col_data_ref, ascending=False)
            
            logger.info(f"Ano {year}: {len(df_target)} documentos encontrados para os FIIs alvo.")

            count = 0
            for idx, row in df_target.iterrows():
                url = row.get(col_link)
                if not url or pd.isna(url): continue
                
                # Limite de segurança para testes (remova ou aumente em produção)
                if count >= 20: 
                    logger.info("Limite de teste atingido (20 docs) para este ano.")
                    break 

                cnpj_formatado = row[col_cnpj]
                data_ref = row.get(col_data_ref, 'S/D')
                
                logger.info(f"Processando [{count+1}]: {cnpj_formatado} - {data_ref}")
                
                text = self.extract_text_from_pdf_url(url)
                
                # Só salva se extraiu algum texto útil (> 50 caracteres)
                if text and len(text) > 50:
                    all_reports.append({
                        'cnpj': cnpj_formatado,
                        'data_referencia': data_ref,
                        'data_entrega': row.get(col_data_ent, None),
                        'url': url,
                        'conteudo_texto': text, # Aqui está o conteúdo para a LLM
                        'ano_competencia': year,
                        'tipo_documento': 'DFIN'
                    })
                    count += 1
                    time.sleep(0.5) # Respeito ao servidor da CVM/B3

        # Salvamento Final
        if all_reports:
            output_file = self.raw_reports_path / "fii_dfin_text.parquet"
            pd.DataFrame(all_reports).to_parquet(output_file, index=False)
            logger.info(f"\n✅ SUCESSO TOTAL! {len(all_reports)} documentos salvos em:")
            logger.info(f"📁 {output_file}")
        else:
            logger.warning("\n⚠️ Nenhum documento coletado. Verifique conexão ou CNPJs.")

def main():
    collector = ReportsCollector()
    collector.run(start_year=2019, end_year=2025)

if __name__ == "__main__":
    main()
