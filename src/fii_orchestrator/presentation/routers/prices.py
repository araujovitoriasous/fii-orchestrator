from fastapi import APIRouter, Query, Depends, Request
from typing import List, Optional
from datetime import date
from pydantic import BaseModel
from fii_orchestrator.serving.duckdb_service import DuckDBService
from fii_orchestrator.serving.cache_service import RedisCacheService
from slowapi.util import get_remote_address
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

class PriceResponse(BaseModel):
    ticker: str
    price: float
    date: date
    volume: Optional[int] = None

router = APIRouter()

# Inicializar serviços
duckdb_service = DuckDBService()
cache_service = RedisCacheService()

@router.get("/", response_model=List[PriceResponse])
@limiter.limit("100/minute")
async def list_prices(
    ticker: Optional[str] = Query(None, description="Filtrar por ticker específico"),
    limit: int = Query(100, ge=1, le=1000, description="Número máximo de resultados"),
    offset: int = Query(0, ge=0, description="Número de resultados para pular"),
    request: Request = None
):
    """
    ## 💰 **Listar Preços de Fundos Imobiliários**
    
    Retorna uma lista paginada de preços históricos dos fundos imobiliários.
    
    ### 🔍 **Parâmetros**
    - **ticker**: Filtro opcional por ticker específico (ex: HGLG11)
    - **limit**: Número máximo de resultados (1-1000)
    - **offset**: Número de resultados para pular (paginação)
    
    ### 📋 **Resposta**
    Lista de preços com informações (ticker, preço, data, volume, fonte)
    
    ### 💾 **Cache**
    Resultados são cacheados por 5 minutos para melhor performance.
    
    ### 🚦 **Rate Limiting**
    Máximo de 100 requests por minuto por IP.
    
    ### 📊 **Exemplos de Uso**
    - `/api/prices/` - Todos os preços
    - `/api/prices/?ticker=HGLG11` - Preços do HGLG11
    - `/api/prices/?limit=50&offset=100` - Paginação
    """
    # Gerar chave de cache
    cache_key = f"prices:list:{ticker}:{limit}:{offset}"
    
    # Tentar buscar do cache
    cached_result = cache_service.get(cache_key)
    if cached_result:
        return cached_result
    
    # Buscar do DuckDB
    prices_data = duckdb_service.get_prices(ticker=ticker, limit=limit, offset=offset)
    
    # Converter para schema de resposta
    prices = []
    for price_data in prices_data:
        price = PriceResponse(
            ticker=price_data[0],
            price=price_data[1],
            date=price_data[2],
            volume=price_data[3]
        )
        prices.append(price)
    
    # Salvar no cache
    cache_service.set(cache_key, prices, ttl=300)
    
    return prices
