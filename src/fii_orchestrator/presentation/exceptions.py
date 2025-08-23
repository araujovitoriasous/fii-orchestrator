from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger

class FIIOrchestratorException(HTTPException):
    """Exceção base para erros da API."""
    pass

async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handler global para exceções não tratadas."""
    logger.error(f"❌ Erro não tratado: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "Ocorreu um erro interno no servidor",
            "timestamp": "2025-08-23T09:00:00Z"
        }
    )
