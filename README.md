# FII Orchestrator

FII Orchestrator coleta e normaliza dados de mercado de Fundos Imobiliários (FIIs).
Ele oferece um ETL simples que baixa cotações e proventos de provedores de
mercado e grava em arquivos Parquet particionados.

## Principais recursos
- **Provedores plugáveis** via interface `MarketDataProvider`.
- **Adapters inclusos**:
  - `YFinanceProvider` para usos de desenvolvimento.
  - `B3VendorProvider` para integração com CSVs oficiais da B3 com
    mapeamento de colunas configurável.
- **ETL de preços e dividendos** executado por `make etl-prices`.
- Configuração por variáveis de ambiente em `.env`.

## Instalação
```bash
make setup
```

## Execução do ETL
Crie um arquivo `.env` com as configurações desejadas:

```env
PROVIDER=yf                 # ou b3_vendor
PRICES_START=2018-01-01     # data inicial

# Parâmetros do B3VendorProvider (se PROVIDER=b3_vendor)
B3_VENDOR_DIR=./vendor/b3
B3_PRICE_GLOB=prices_*.csv
B3_DIV_GLOB=dividends_*.csv
B3_PRICE_COLMAP={"date":"DATA","ticker":"TICKER","close":"FECHAMENTO","volume":"VOLUME"}
B3_DIV_COLMAP={"ex_date":"DATA_EX","payment_date":"DATA_PAG","ticker":"TICKER","value":"VALOR"}
```

Rodar o ETL:
```bash
make etl-prices
```
Os arquivos Parquet serão gerados em `data/bronze/prices/` e `data/bronze/dividends/`.

## Estrutura do projeto
- `src/fii_orchestrator/etl/providers/` – implementações de provedores de dados.
- `src/fii_orchestrator/etl/b3_prices.py` – ponto de entrada do ETL de preços.
- `Makefile` – tarefas comuns (`setup`, `etl-prices`).

## Licença
Distribuído sob a licença MIT.
