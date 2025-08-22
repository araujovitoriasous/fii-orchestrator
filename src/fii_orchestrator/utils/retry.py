import time
import random
from functools import wraps
from typing import Callable, Any, Optional
from loguru import logger

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,)
):
    """
    Decorator para retry automático com backoff exponencial.
    
    Args:
        max_retries: Número máximo de tentativas
        base_delay: Delay inicial em segundos
        max_delay: Delay máximo em segundos
        exponential_base: Base para cálculo exponencial
        jitter: Adicionar variação aleatória para evitar thundering herd
        exceptions: Tupla de exceções que devem triggerar retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Falha final após {max_retries + 1} tentativas: {e}")
                        raise last_exception
                    
                    # Calcular delay com backoff exponencial
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    # Adicionar jitter se habilitado
                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)
                    
                    logger.warning(
                        f"Tentativa {attempt + 1}/{max_retries + 1} falhou: {e}. "
                        f"Tentando novamente em {delay:.2f}s..."
                    )
                    
                    time.sleep(delay)
            
            # Nunca deve chegar aqui, mas por segurança
            raise last_exception
        
        return wrapper
    return decorator

def retry_on_specific_errors(
    error_mapping: dict[type, dict],
    max_retries: int = 3
):
    """
    Decorator para retry com tratamento específico por tipo de erro.
    
    Args:
        error_mapping: Dict com mapeamento de exceção -> configuração de retry
        max_retries: Número máximo de tentativas por tipo de erro
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_type = type(e)
                    
                    # Verificar se temos configuração específica para este erro
                    if error_type in error_mapping:
                        config = error_mapping[error_type]
                        delay = config.get('delay', 1.0)
                        should_retry = config.get('retry', True)
                        
                        if not should_retry or attempt == max_retries:
                            logger.error(f"Erro não recuperável após {attempt + 1} tentativas: {e}")
                            raise e
                        
                        logger.warning(
                            f"Erro específico {error_type.__name__} na tentativa {attempt + 1}: {e}. "
                            f"Tentando novamente em {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        # Erro não mapeado, não retry
                        logger.error(f"Erro não mapeado: {e}")
                        raise e
            
            return None
        
        return wrapper
    return decorator
