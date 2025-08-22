# 🏗️ FII Orchestrator

**Sistema de orquestração de dados para Fundos Imobiliários (FIIs) com arquitetura Clean Architecture**

## 🎯 **Visão Geral**

O FII Orchestrator é uma solução robusta para coleta, processamento e análise de dados de Fundos Imobiliários brasileiros. O projeto implementa uma arquitetura moderna baseada em **Clean Architecture** e **Domain-Driven Design (DDD)**.

## ✨ **Funcionalidades Principais**

- **Coleta de Cotações**: Preços e volumes via Yahoo Finance
- **Coleta de Dividendos**: Histórico de proventos por cota
- **Relatórios CVM**: Dados regulatórios e indicadores financeiros
- **Notícias RSS**: Coleta automática de fontes especializadas
- **Mapeamento FII ↔ CNPJ**: Referência completa de fundos
- **Processamento Paralelo**: Coleta simultânea de múltiplos FIIs
- **Sistema de Retry**: Recuperação automática de falhas
- **Validação de Dados**: Detecção de outliers e inconsistências

## 🚀 **Instalação e Uso**

```bash
# Instalar dependências
make setup

# Gerar referência de fundos
make ref-funds

# Executar pipeline completo
make etl-pipeline-complete

# Executar apenas coleta de preços
make etl-prices-refactored

# Executar apenas coleta de notícias
make etl-news-refactored

# Executar apenas coleta CVM
make etl-cvm-refactored

# Validar qualidade dos dados
make quality-validation

# Executar testes
make test
```

## 🏗️ **Arquitetura**

O sistema segue os princípios de Clean Architecture com as seguintes camadas:

- **Domain**: Entidades e regras de negócio
- **Application**: Casos de uso e lógica de aplicação
- **Infrastructure**: Repositórios e adaptadores externos
- **ETL**: Orquestradores de coleta de dados

## 📊 **Estrutura de Dados**

Os dados são salvos em formato Parquet com particionamento por:
- Ticker do FII
- Ano e mês de coleta
- Tipo de dado (preços, dividendos, notícias)

## 🧪 **Testes**

```bash
# Testes da camada de domínio
make test-domain

# Testes da camada de aplicação
make test-application

# Todos os testes
make test
```

## 📝 **Comandos Disponíveis**

Execute `make help` para ver todos os comandos disponíveis.

## 🔧 **Desenvolvimento**

```bash
# Formatar código
make fmt

# Verificar qualidade
make lint

# Limpar arquivos temporários
make clean
```
