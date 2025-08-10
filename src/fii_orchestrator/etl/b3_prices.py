from loguru import logger
# TODO: implementar coleta de preços/proventos (ex.: B3 API/raspagem autorizada)
# salvar em data/bronze/prices com particionamento: year, month, ticker

def run():
    logger.info("ETL B3 (prices/dividends) - TODO")
    # 1) baixar dados
    # 2) normalizar esquema: date, ticker, close, volume, kind=price|dividend, value
    # 3) salvar parquet particionado (year, month, ticker)

if __name__ == "__main__":
    run()
