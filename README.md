# 🏗️ FII Orchestrator

**Sistema de orquestração de dados para Fundos Imobiliários (FIIs)**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-green.svg)](https://fastapi.tiangolo.com)
[![Architecture](https://img.shields.io/badge/Architecture-Hexagonal-orange.svg)](https://en.wikipedia.org/wiki/Hexagonal_architecture_(software))
[![Tests](https://img.shields.io/badge/Tests-42%20passing-brightgreen.svg)](https://pytest.org)
[![Coverage](https://img.shields.io/badge/Coverage-28%25-yellow.svg)](https://coverage.readthedocs.io)

## 🎯 **Visão Geral**

O FII Orchestrator é uma solução **empresarial de nível mundial** para coleta, processamento e análise de dados de Fundos Imobiliários brasileiros.

## 🚀 **Instalação e Uso**

```bash
# Instalar dependências
make setup

# Iniciar API
make api-start

# Executar testes
make test                    # Todos os testes
make test-unit              # Testes unitários
make test-integration       # Testes de integração
make test-coverage          # Testes com cobertura

# Desenvolvimento
make fmt                    # Formatar código
make lint                   # Verificar qualidade
make clean                  # Limpar arquivos temporários
```

## 🏗️ **Arquitetura**

O sistema implementa uma **arquitetura hexagonal híbrida**:

### **📊 Camadas da Aplicação**
- **Presentation**: FastAPI com Swagger, rate limiting e tratamento de erros
- **Application**: Use Cases, CQRS, Event Sourcing e Sagas
- **Domain**: Entidades, Value Objects e regras de negócio
- **Infrastructure**: Bancos especializados, DI Container e adaptadores

### **🗄️ Estratégia de Dados**
- **PostgreSQL**: Dados operacionais (fundos, preços, dividendos)
- **ClickHouse**: Analytics e séries temporais
- **Redis**: Cache de alta performance
- **DuckDB**: Desenvolvimento e testes locais

### **🔄 Padrões Implementados**
- **CQRS**: Separação de comandos e consultas
- **Event Sourcing**: Rastreamento de mudanças
- **Sagas**: Transações distribuídas
- **Dependency Injection**: Inversão de dependências

## 📊 **Estrutura de Dados**

### **🗄️ Bancos de Dados**
- **PostgreSQL**: Schema normalizado para operações CRUD
- **ClickHouse**: Tabelas otimizadas para analytics
- **Redis**: Cache de queries frequentes
- **Parquet**: Arquivos para desenvolvimento local

### **📁 Organização**
- **Operacional**: Dados transacionais e relacionamentos
- **Analítico**: Histórico de preços e métricas
- **Cache**: Respostas de API e sessões
- **Desenvolvimento**: Arquivos Parquet para testes

## 🧪 **Testes**

```bash
# Testes unitários
make test-unit

# Testes de integração
make test-integration

# Testes end-to-end
make test-e2e

# Todos os testes
make test

# Testes com cobertura
make test-coverage
```

## 🚀 **API REST**

### **📡 Endpoints Disponíveis**
- **`/`**: Status da API
- **`/health`**: Health check dos serviços
- **`/stats`**: Estatísticas da API
- **`/swagger`**: Documentação interativa
- **`/api/funds/`**: Listagem de fundos com paginação
- **`/api/prices/`**: Histórico de preços com filtros

### **⚡ Recursos da API**
- **Rate Limiting**: 100 requests/minuto
- **Cache Redis**: TTL configurável
- **Paginação**: Limit/offset para grandes datasets
- **Tratamento de Erros**: Respostas padronizadas
- **CORS**: Suporte a cross-origin requests
- **Logging**: Estruturado com Loguru

## 📝 **Comandos Disponíveis**

Execute `make help` para ver todos os comandos disponíveis.

## 🔧 **Desenvolvimento**

### **🛠️ Ferramentas**
- **Poetry**: Gerenciamento de dependências
- **Black**: Formatação de código
- **Ruff**: Linting e formatação
- **Pytest**: Framework de testes
- **Pre-commit**: Hooks de qualidade

### **📋 Comandos**
```bash
# Formatar código
make fmt

# Verificar qualidade
make lint

# Limpar arquivos temporários
make clean

# Validar arquitetura
make arch-test
```

## 🌍 **Configuração de Ambiente**

### **📋 Variáveis de Ambiente**
```bash
# API
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false

# Redis
REDIS_URL=redis://localhost:6379
REDIS_ENABLED=true

# PostgreSQL
POSTGRES_URL=postgresql://user:pass@localhost/fii_db
POSTGRES_ENABLED=true

# ClickHouse
CLICKHOUSE_URL=http://localhost:8123
CLICKHOUSE_ENABLED=true

# Cache
CACHE_TTL=300
RATE_LIMIT=100/minute
```

### **📁 Estrutura do Projeto**
```
fii-orchestrator/
├── src/fii_orchestrator/
│   ├── application/          # Use Cases, CQRS, Events, Sagas
│   ├── domain/              # Entities, Value Objects
│   ├── infrastructure/      # Database, DI Container
│   ├── presentation/        # FastAPI, Routers, Middleware
│   └── serving/            # Services (Cache, DuckDB)
├── tests/                  # Testes unitários, integração, e2e
├── data/                   # Dados Parquet para desenvolvimento
└── docs/                   # Documentação
```

## 🚀 **Deploy e Produção**

### **🐳 Docker (Recomendado)**
```bash
# Construir imagem
docker build -t fii-orchestrator .

# Executar container
docker run -p 8000:8000 fii-orchestrator
```

### **☁️ Cloud Native**
- **Kubernetes**: Deploy com Helm charts
- **AWS**: RDS PostgreSQL, ElastiCache Redis
- **GCP**: Cloud SQL, Memorystore Redis
- **Azure**: Azure Database, Azure Cache

## 📊 **Monitoramento e Observabilidade**

### **📈 Métricas**
- **Health Checks**: `/health` endpoint
- **Estatísticas**: `/stats` endpoint
- **Logs**: Estruturados com Loguru
- **Tracing**: Preparado para OpenTelemetry

### **🔍 Dashboards**
- **API Metrics**: Requests, response times, errors
- **Database Performance**: Query times, connections
- **Cache Hit Rate**: Redis performance
- **Business Metrics**: FIIs processados, dados coletados

## 📄 **Licença**

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.