from fastapi import APIRouter, Query, Depends, Request
from typing import List, Optional
from fii_orchestrator.presentation.schemas.responses import FundResponse
from fii_orchestrator.serving.duckdb_service import DuckDBService
from fii_orchestrator.serving.cache_service import RedisCacheService
from slowapi.util import get_remote_address
from slowapi import Limiter
from fii_orchestrator.infrastructure.di_container import Container

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

# Funções de dependência
def get_duckdb_service() -> DuckDBService:
    return Container().duckdb_service()

def get_cache_service() -> RedisCacheService:
    return Container().cache_service()

@router.get("/", response_model=List[FundResponse])
@limiter.limit("100/minute")
async def list_funds(
    limit: int = Query(100, ge=1, le=1000, description="Número máximo de resultados"),
    offset: int = Query(0, ge=0, description="Número de resultados para pular"),
    request: Request = None,
    duckdb_service: DuckDBService = Depends(get_duckdb_service),
    cache_service: RedisCacheService = Depends(get_cache_service)
):
    """
    ## 📊 **Listar Fundos Imobiliários**
    
    Retorna uma lista paginada de todos os fundos imobiliários disponíveis no sistema.
    
    ### 🔍 **Parâmetros**
    - **limit**: Número máximo de resultados (1-1000)
    - **offset**: Número de resultados para pular (paginação)
    
    ### 📋 **Resposta**
    Lista de fundos com informações básicas (ticker, CNPJ, razão social, fonte)
    
    ### 💾 **Cache**
    Resultados são cacheados por 5 minutos para melhor performance.
    
    ### 🚦 **Rate Limiting**
    Máximo de 100 requests por minuto por IP.
    """
    # Gerar chave de cache
    cache_key = f"funds:list:{limit}:{offset}"
    
    # Tentar buscar do cache
    cached_result = cache_service.get(cache_key)
    if cached_result:
        return cached_result
    
    # Buscar do DuckDB
    funds_data = duckdb_service.get_funds(limit=limit, offset=offset)
    
    # Converter para schema de resposta
    funds = []
    for fund_data in funds_data:
        fund = FundResponse(
            ticker=fund_data[0],
            cnpj=fund_data[1],
            razao_social=fund_data[2],
            fonte=fund_data[3]
        )
        funds.append(fund)
    
    # Salvar no cache
    cache_service.set(cache_key, funds, ttl=300)
    
    return funds
