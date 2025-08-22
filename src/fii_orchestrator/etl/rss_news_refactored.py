"""
Módulo refatorado de coleta de notícias RSS.
Usa a nova arquitetura Clean Architecture.
"""

from typing import List, Optional
from loguru import logger

from fii_orchestrator.infrastructure.config import get_config
from fii_orchestrator.etl.orchestrators import ETLOrchestrator

def run():
    """Função principal para execução do módulo refatorado."""
    logger.info("📰 Iniciando ETL de notícias RSS (versão refatorada)")
    
    try:
        # Usar orquestrador da nova arquitetura
        orchestrator = ETLOrchestrator()
        
        # Executar coleta de notícias
        result = orchestrator.run_news_collection()
        
        if result:
            logger.info("✅ ETL de notícias RSS concluído com sucesso")
            logger.info(f"📊 Resumo: {result['news_collected']} notícias coletadas")
        else:
            logger.warning("⚠️ Nenhuma notícia coletada")
            
    except Exception as e:
        logger.exception(f"❌ Erro fatal no ETL de notícias RSS: {e}")
        raise

def run_with_custom_sources(sources: List[str]):
    """Executa coleta com fontes RSS personalizadas."""
    logger.info(f"📡 Coletando de fontes personalizadas: {sources}")
    
    try:
        orchestrator = ETLOrchestrator()
        result = orchestrator.run_news_collection(sources=sources)
        
        if result:
            logger.info("✅ Coleta com fontes personalizadas concluída")
            logger.info(f"📊 Resumo: {result['news_collected']} notícias coletadas")
        else:
            logger.warning("⚠️ Nenhuma notícia coletada")
            
    except Exception as e:
        logger.exception(f"❌ Erro na coleta com fontes personalizadas: {e}")
        raise

if __name__ == "__main__":
    run()
