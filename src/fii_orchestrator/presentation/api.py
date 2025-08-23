from fastapi import FastAPI, Depends
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fii_orchestrator.presentation.middleware.rate_limit import setup_rate_limiting
from fii_orchestrator.presentation.exceptions import global_exception_handler, FIIOrchestratorException
from fii_orchestrator.infrastructure.di_container import Container

# Criar container de DI
container = Container()

app = FastAPI(
    title="FII Orchestrator API",
    description="""
    ## 🏗️ **API de Orquestração de Fundos Imobiliários (FIIs)**
    
    Esta API fornece acesso aos dados coletados e processados de Fundos Imobiliários brasileiros.
    
    ### 🚀 **Funcionalidades**
    - **Fundos**: Listagem e consulta de fundos disponíveis
    - **Preços**: Histórico de cotações e volumes
    - **Cache**: Sistema de cache Redis para performance
    - **Rate Limiting**: Proteção contra abuso (100 req/min)
    - **Paginação**: Suporte a paginação em todos os endpoints
    
    ### 📊 **Dados Disponíveis**
    - Informações dos fundos (ticker, CNPJ, razão social)
    - Histórico de preços e volumes
    - Metadados de coleta e fontes
    
    ### 🔧 **Tecnologias**
    - **Backend**: FastAPI + Python
    - **Banco**: DuckDB com arquivos Parquet
    - **Cache**: Redis
    - **Rate Limiting**: SlowAPI
    """,
    version="1.0.0",
    contact={
        "name": "FII Orchestrator Team",
        "email": "vitoria.sousa@stone.com.br",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    docs_url="/swagger",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configurar rate limiting
setup_rate_limiting(app)

# Configurar tratamento de erros global
app.add_exception_handler(Exception, global_exception_handler)

# Configurar container de DI
app.container = container

@app.get("/", tags=["Status"])
async def root():
    """
    ## 🏠 **Status da API**
    
    Endpoint raiz que retorna informações básicas sobre a API.
    
    ### 📋 **Resposta**
    - **message**: Mensagem de boas-vindas
    - **status**: Status da API
    - **version**: Versão atual
    - **timestamp**: Horário da requisição
    """
    from datetime import datetime
    return {
        "message": "FII Orchestrator API",
        "status": "operational",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", tags=["Status"])
async def health_check():
    """
    ## 🩺 **Health Check**
    
    Endpoint para verificar a saúde da API e serviços dependentes.
    
    ### 📋 **Resposta**
    - **status**: Status geral (healthy/unhealthy)
    - **services**: Status de cada serviço
    - **timestamp**: Horário da verificação
    """
    from datetime import datetime
    import os
    
    # Verificar serviços
    services_status = {
        "duckdb": "healthy",  # Sempre healthy por enquanto
        "redis": "unknown",   # Será verificado quando implementado
        "parquet_files": "unknown"  # Será verificado quando implementado
    }
    
    # Verificar se há dados Parquet
    data_dir = os.path.join(os.getcwd(), "data", "bronze")
    if os.path.exists(data_dir):
        services_status["parquet_files"] = "available"
    else:
        services_status["parquet_files"] = "not_found"
    
    overall_status = "healthy" if all(s == "healthy" or s == "available" for s in services_status.values()) else "degraded"
    
    return {
        "status": overall_status,
        "services": services_status,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stats", tags=["Status"])
async def get_stats():
    """
    ## 📊 **Estatísticas da API**
    
    Endpoint para obter estatísticas sobre o uso da API e dados disponíveis.
    
    ### 📋 **Resposta**
    - **api_stats**: Estatísticas da API
    - **data_stats**: Estatísticas dos dados
    - **cache_stats**: Estatísticas do cache
    - **timestamp**: Horário da requisição
    """
    from datetime import datetime
    import os
    
    # Estatísticas da API
    api_stats = {
        "version": "1.0.0",
        "uptime": "running",
        "endpoints": 4,  # root, health, funds, prices
        "rate_limit": "100/minute"
    }
    
    # Estatísticas dos dados
    data_stats = {
        "parquet_files": 0,
        "data_size_mb": 0,
        "last_update": "unknown"
    }
    
    # Verificar dados Parquet
    data_dir = os.path.join(os.getcwd(), "data", "bronze")
    if os.path.exists(data_dir):
        for subdir in ["funds", "prices", "news"]:
            subdir_path = os.path.join(data_dir, subdir)
            if os.path.exists(subdir_path):
                parquet_files = [f for f in os.listdir(subdir_path) if f.endswith('.parquet')]
                data_stats["parquet_files"] += len(parquet_files)
    
    # Estatísticas do cache (serão implementadas quando Redis estiver ativo)
    cache_stats = {
        "redis_status": "not_configured",
        "cache_hits": 0,
        "cache_misses": 0
    }
    
    return {
        "api_stats": api_stats,
        "data_stats": data_stats,
        "cache_stats": cache_stats,
        "timestamp": datetime.now().isoformat()
    }

from fii_orchestrator.presentation.routers import funds

app.include_router(funds.router, prefix="/api/funds", tags=["Fundos"])

from fii_orchestrator.presentation.routers import prices

app.include_router(prices.router, prefix="/api/prices", tags=["Preços"])
