"""
Módulo refatorado de coleta de preços e dividendos da B3.
Usa a nova arquitetura Clean Architecture.
"""

from datetime import datetime, timedelta
from typing import List, Optional
from loguru import logger

from fii_orchestrator.infrastructure.config import get_config, get_container
from fii_orchestrator.etl.orchestrators import ETLOrchestrator

def run():
    """Função principal para execução do módulo refatorado."""
    logger.info("🚀 Iniciando ETL de preços e dividendos (versão refatorada)")
    
    try:
        # Usar orquestrador da nova arquitetura
        orchestrator = ETLOrchestrator()
        
        # Executar coleta de dados de fundos
        result = orchestrator.run_funds_data_collection()
        
        if result and result.get("status") != "no_tickers":
            logger.info("✅ ETL de preços e dividendos concluído com sucesso")
            logger.info(f"📊 Resumo: {result['successful_collections']} sucessos, {result['failed_collections']} falhas")
        else:
            logger.warning("⚠️ Nenhum ticker encontrado para coleta")
            
    except Exception as e:
        logger.exception(f"❌ Erro fatal no ETL de preços e dividendos: {e}")
        raise

def run_sequential():
    """Executa coleta de forma sequencial (para compatibilidade)."""
    logger.info("🐌 Executando coleta sequencial")
    
    # Configurar para modo sequencial
    config = get_config()
    config.processing.use_parallel = False
    
    run()

def run_parallel():
    """Executa coleta de forma paralela (para compatibilidade)."""
    logger.info("⚡ Executando coleta paralela")
    
    # Configurar para modo paralelo
    config = get_config()
    config.processing.use_parallel = True
    
    run()

if __name__ == "__main__":
    run()
