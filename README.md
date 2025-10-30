# FII-O: Sistema de Coleta de Dados de FIIs

Sistema completo de coleta, processamento e armazenamento de dados de Fundos de Investimento Imobiliário (FIIs) brasileiros, dados macroeconômicos e fundamentos financeiros.

## 📊 Dados Coletados

### 1️⃣ Dados de FIIs (`collect_fii_data.py`)

**Fonte**: Yahoo Finance + Fundamentus (web scraping)

**Período**: 2019-01-01 a 2025-01-01 (diário)

**Arquivo gerado**: `data/01_raw/prices.parquet`

**Colunas**:
- `data` (datetime): Data de referência
- `ticker` (str): Código do FII (ex: HGLG11)
- `preco_ajustado` (float): Preço da cota ajustado (distribuições e splits)
- `volume` (float): Volume negociado no dia
- `cnpj` (str): CNPJ do fundo
- `nome` (str): Nome do fundo
- `vp_cota` (float): Valor patrimonial por cota (VP/cota)
- `pvp` (float): Preço sobre valor patrimonial (P/VP)
- `vacancia` (float): Taxa de vacância (em decimal, ex: 0.035 = 3.5%)
- `num_cotas` (int): Número total de cotas
- `tipo_gestao` (str): Tipo de gestão (Ativa, Definida)
- `dy_12m` (float): Dividend Yield dos últimos 12 meses (em decimal)

**Arquivo gerado**: `data/01_raw/dividends.parquet`

**Colunas**:
- `data` (datetime): Data de distribuição
- `dividendo` (float): Valor do dividendo por cota
- `ticker` (str): Código do FII
- `cnpj` (str): CNPJ do fundo
- `nome` (str): Nome do fundo

---

### 2️⃣ Dados Macroeconômicos (`collect_macro.py`)

**Fonte**: Banco Central do Brasil (SGS) + FRED API (PMI Brasil)

**Período**: 2019-01-01 a 2025-01-01 (diário e mensal)

**Arquivo gerado**: `data/02_processed/market/macro_data.parquet`

**Colunas**:
- `data` (datetime): Data de referência
- `taxa_juros_real` (float): Taxa de juros real - Títulos públicos prefixados (código SGS 1178) - diária
- `ipca` (float): Índice Nacional de Preços ao Consumidor Amplo (código SGS 433) - mensal
- `cdi` (float): Taxa de juros - Certificado de Depósito Interbancário (código SGS 12) - diária
- `igpm` (float): Índice Geral de Preços do Mercado (código SGS 189) - mensal
- `ibc_br` (float): Índice de Atividade Econômica - Prévia do PIB (código SGS 24363) - mensal
- `pmi_brasil` (float): Índice de Gerentes de Compras - Brasil (S&P Global) - mensal (opcional, requer API key FRED)

**Observações**:
- Séries mensais (IPCA, IGP-M, IBC-Br, PMI) são propagadas para todos os dias do mês usando forward fill
- Séries diárias têm valores preenchidos para fins de semana e feriados

---

### 3️⃣ Fundamentos Financeiros (`collect_fundamentals.py`)

**Fonte**: CVM (Comissão de Valores Mobiliários)

**Período**: 2019-Q1 a 2025-Q2 (trimestral)

**Arquivo gerado**: `data/02_processed/fundamentals/fundamentals_trimestral.parquet`

**Colunas**:
- `ticker` (str): Código do FII
- `data_referencia` (datetime): Data de referência do trimestre
- `ano` (int): Ano
- `trimestre` (int): Trimestre (1-4)
- `cnpj` (str): CNPJ do fundo
- `Receita_Caixa` (float): Receitas financeiras principais (aluguel + juros TVM + juros aplicação)
- `Lucro_Caixa_Trimestral` (float): Resultado líquido trimestral financeiro
- `Liquidez_Caixa` (float): Recursos de liquidez disponíveis
- `Taxa_Administracao` (float): Taxa de administração (agregada como média dos 3 meses do trimestre a partir de dados mensais da CVM)

**Observações**:
- Dados trimestrais obtidos de relatórios `INF_TRIMESTRAL` da CVM
- Taxa de Administração é obtida de relatórios mensais (`INF_MENSAL/complemento`) e agregada por trimestre
- FIIs cobertos: 16 FIIs ativos

---

## 🚀 Como Usar

### Requisitos

```bash
pip install -r requirements.txt
```

### Coletar Dados de FIIs

```bash
cd src/etl
python3 collect_fii_data.py
```

### Coletar Dados Macroeconômicos

```bash
cd src/etl
python3 collect_macro.py
```

**Observação**: Para coletar PMI Brasil, é necessário configurar a variável de ambiente `FRED_API_KEY`:

```bash
export FRED_API_KEY="sua_chave_aqui"
```

### Coletar Fundamentos Financeiros

```bash
cd src/etl
python3 collect_fundamentals.py
```

---

## 📁 Estrutura de Diretórios

```
fii-o/
├── data/
│   ├── 01_raw/                     # Dados brutos
│   │   ├── prices.parquet          # Preços e dados Fundamentus
│   │   └── dividends.parquet       # Histórico de proventos
│   ├── 02_processed/               # Dados processados
│   │   ├── market/
│   │   │   └── macro_data.parquet  # Dados macroeconômicos
│   │   └── fundamentals/
│   │       └── fundamentals_trimestral.parquet  # Fundamentos CVM
│   └── metadata/
│       └── fiis_metadata.json      # Metadata centralizado (tickers, CNPJs)
├── src/
│   └── etl/
│       ├── collect_fii_data.py     # Coleta FIIs
│       ├── collect_macro.py        # Coleta macro
│       └── collect_fundamentals.py # Coleta fundamentos
└── requirements.txt
```

---

## 📈 FIIs Coletados (16 Fundos)

| Ticker | Nome | Tipo |
|--------|------|------|
| ALZR11 | Alianza Trust Renda Imobiliária | Híbrido |
| BRCR11 | BTG Pactual Corporate Office Fund | Lajes Corporativas |
| BTAL11 | BTG Pactual Logística | Logística |
| BTLG11 | BTG Pactual Logística | Logística |
| HGBS11 | CSHG Brasil Shopping | Shopping |
| HGCR11 | CSHG Recebíveis Imobiliários | Recebíveis |
| HGLG11 | CSHG Logística | Logística |
| HGRU11 | CSHG Renda Urbana | Tijolo |
| KNHY11 | Kinea High Yield CRI | Recebíveis |
| KNIP11 | Kinea Índices de Preços | Recebíveis |
| KNRI11 | Kinea Renda Imobiliária | Híbrido |
| VISC11 | Vinci Shopping Centers | Shopping |
| VSLH11 | Vinci Shopping Centers | Shopping |
| XPCM11 | XP Corporate Office | Lajes Corporativas |
| XPIN11 | XP Industrial | Logística |
| XPML11 | XP Malls | Shopping |

---

## 🔧 Detalhes Técnicos

### Rate Limiting

- **Fundamentus**: Delay de 1 segundo entre requisições para evitar bloqueios
- **CVM**: Sem limite explícito, mas processamento em lotes

### Tratamento de Dados Faltantes

- **Preços**: Preços ajustados são calculados automaticamente pelo Yahoo Finance
- **Macro**: Forward fill e backward fill para séries mensais e diárias
- **Fundamentos**: Interpolação linear para valores faltantes

### Validação de Dados

- Metadata centralizado (`fiis_metadata.json`) com CNPJs validados com CVM
- Mapeamento automático CNPJ → Ticker
- Suporte a CNPJs compartilhados (ex: BTAL11 e BTLG11)

---

## 📝 Observações Importantes

1. **PMI Brasil**: Requer API key gratuita do FRED. Se não disponível, o sistema continua sem essa série.

2. **Taxa de Administração**: Coletada via CVM (mensal) e agregada para análise trimestral.

3. **Tipo de Gestão**: Coletado do Fundamentus (valores: Ativa, Definida).

4. **Periodicidade**:
   - Dados FIIs: Diários (apenas dias úteis)
   - Macro: Diários (séries diárias) e Mensais (séries mensais propragadas)
   - Fundamentos: Trimestrais (com Taxa de Administração agregada mensalmente)

5. **Valor de Mercado**: Pode ser calculado multiplicando `preco_ajustado * num_cotas`.

---

## 📅 Última Atualização

- **Metadata**: 2025-10-12
- **Código**: 2025-10-29

---

## 📄 Licença

Este projeto é de uso interno para análise de FIIs.

