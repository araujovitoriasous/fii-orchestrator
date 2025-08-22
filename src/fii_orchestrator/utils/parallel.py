import asyncio
import concurrent.futures
from typing import List, Callable, Any, Dict, Optional
from functools import partial
import time
from loguru import logger
import polars as pl

class ParallelProcessor:
    """Classe para processamento paralelo de dados."""
    
    def __init__(self, max_workers: int = 4, chunk_size: int = 10):
        """
        Inicializa o processador paralelo.
        
        Args:
            max_workers: Número máximo de workers paralelos
            chunk_size: Tamanho dos chunks para processamento
        """
        self.max_workers = max_workers
        self.chunk_size = chunk_size
    
    def process_chunks(
        self, 
        items: List[Any], 
        process_func: Callable, 
        chunk_size: Optional[int] = None
    ) -> List[Any]:
        """
        Processa itens em chunks paralelos.
        
        Args:
            items: Lista de itens para processar
            process_func: Função para processar cada item
            chunk_size: Tamanho do chunk (usa self.chunk_size se None)
            
        Returns:
            Lista com resultados do processamento
        """
        chunk_size = chunk_size or self.chunk_size
        
        if not items:
            return []
        
        # Dividir em chunks
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        logger.info(f"Processando {len(items)} itens em {len(chunks)} chunks com {self.max_workers} workers")
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submeter chunks para processamento
            future_to_chunk = {
                executor.submit(self._process_chunk, chunk, process_func): chunk 
                for chunk in chunks
            }
            
            # Coletar resultados
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    logger.debug(f"Chunk processado: {len(chunk_results)} resultados")
                except Exception as e:
                    logger.error(f"Erro ao processar chunk: {e}")
                    # Continuar com outros chunks
        
        logger.info(f"Processamento concluído: {len(results)} resultados")
        return results
    
    def _process_chunk(self, chunk: List[Any], process_func: Callable) -> List[Any]:
        """
        Processa um chunk de itens.
        
        Args:
            chunk: Lista de itens do chunk
            process_func: Função para processar cada item
            
        Returns:
            Lista com resultados do chunk
        """
        results = []
        for item in chunk:
            try:
                result = process_func(item)
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.error(f"Erro ao processar item {item}: {e}")
                continue
        return results
    
    def process_with_progress(
        self, 
        items: List[Any], 
        process_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> List[Any]:
        """
        Processa itens com callback de progresso.
        
        Args:
            items: Lista de itens para processar
            process_func: Função para processar cada item
            progress_callback: Função chamada com progresso (opcional)
            
        Returns:
            Lista com resultados do processamento
        """
        total_items = len(items)
        processed = 0
        results = []
        
        def process_with_progress(item):
            nonlocal processed
            try:
                result = process_func(item)
                processed += 1
                
                if progress_callback:
                    progress_callback(processed, total_items)
                
                return result
            except Exception as e:
                logger.error(f"Erro ao processar item {item}: {e}")
                processed += 1
                return None
        
        return self.process_chunks(items, process_with_progress)

class AsyncProcessor:
    """Classe para processamento assíncrono de dados."""
    
    def __init__(self, max_concurrent: int = 10):
        """
        Inicializa o processador assíncrono.
        
        Args:
            max_concurrent: Número máximo de tarefas concorrentes
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_items_async(
        self, 
        items: List[Any], 
        process_func: Callable,
        delay_between: float = 0.1
    ) -> List[Any]:
        """
        Processa itens de forma assíncrona.
        
        Args:
            items: Lista de itens para processar
            process_func: Função assíncrona para processar cada item
            delay_between: Delay entre processamentos (para respeitar rate limits)
            
        Returns:
            Lista com resultados do processamento
        """
        if not items:
            return []
        
        logger.info(f"Processando {len(items)} itens de forma assíncrona (max {self.max_concurrent} concorrentes)")
        
        async def process_with_semaphore(item):
            async with self.semaphore:
                try:
                    result = await process_func(item)
                    await asyncio.sleep(delay_between)  # Rate limiting
                    return result
                except Exception as e:
                    logger.error(f"Erro ao processar item {item}: {e}")
                    return None
        
        # Criar todas as tarefas
        tasks = [process_with_semaphore(item) for item in items]
        
        # Executar e aguardar conclusão
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrar resultados válidos
        valid_results = [r for r in results if r is not None and not isinstance(r, Exception)]
        
        logger.info(f"Processamento assíncrono concluído: {len(valid_results)} resultados válidos")
        return valid_results
    
    async def process_batches_async(
        self, 
        items: List[Any], 
        process_func: Callable,
        batch_size: int = 5,
        delay_between_batches: float = 1.0
    ) -> List[Any]:
        """
        Processa itens em lotes assíncronos.
        
        Args:
            items: Lista de itens para processar
            process_func: Função assíncrona para processar cada item
            batch_size: Tamanho de cada lote
            delay_between_batches: Delay entre lotes
            
        Returns:
            Lista com resultados do processamento
        """
        if not items:
            return []
        
        # Dividir em lotes
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        logger.info(f"Processando {len(items)} itens em {len(batches)} lotes de {batch_size}")
        
        all_results = []
        
        for i, batch in enumerate(batches):
            logger.debug(f"Processando lote {i+1}/{len(batches)} com {len(batch)} itens")
            
            # Processar lote atual
            batch_results = await self.process_items_async(batch, process_func)
            all_results.extend(batch_results)
            
            # Delay entre lotes (exceto no último)
            if i < len(batches) - 1:
                await asyncio.sleep(delay_between_batches)
        
        logger.info(f"Processamento em lotes concluído: {len(all_results)} resultados")
        return all_results

def run_parallel_processing(
    items: List[Any], 
    process_func: Callable, 
    max_workers: int = 4,
    chunk_size: int = 10
) -> List[Any]:
    """
    Função de conveniência para processamento paralelo simples.
    
    Args:
        items: Lista de itens para processar
        process_func: Função para processar cada item
        max_workers: Número máximo de workers
        chunk_size: Tamanho dos chunks
        
    Returns:
        Lista com resultados do processamento
    """
    processor = ParallelProcessor(max_workers=max_workers, chunk_size=chunk_size)
    return processor.process_chunks(items, process_func)

async def run_async_processing(
    items: List[Any], 
    process_func: Callable, 
    max_concurrent: int = 10
) -> List[Any]:
    """
    Função de conveniência para processamento assíncrono simples.
    
    Args:
        items: Lista de itens para processar
        process_func: Função assíncrona para processar cada item
        max_concurrent: Número máximo de tarefas concorrentes
        
    Returns:
        Lista com resultados do processamento
    """
    processor = AsyncProcessor(max_concurrent=max_concurrent)
    return await processor.process_items_async(items, process_func)
