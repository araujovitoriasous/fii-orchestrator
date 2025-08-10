from loguru import logger
# TODO: baixar "Informe Mensal", "Fatos Relevantes" etc. do portal CVM/FundInfo
# extrair indicadores (vacância, P/VP, DY), mapear FII <-> CNPJ

def run():
    logger.info("ETL CVM (relatórios) - TODO")
    # 1) baixar PDFs/HTML
    # 2) extrair campos (pdfplumber/BeautifulSoup)
    # 3) salvar features em parquet: fund_id (ticker), cnpj, ref_date, metrics...

if __name__ == "__main__":
    run()
