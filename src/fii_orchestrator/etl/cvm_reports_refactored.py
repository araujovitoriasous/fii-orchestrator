"""
Módulo refatorado de coleta de relatórios CVM.
Usa a nova arquitetura Clean Architecture.
"""

from typing import List, Optional
from loguru import logger

from fii_orchestrator.infrastructure.config import get_config
from fii_orchestrator.etl.orchestrators import ETLOrchestrator

def run():
    """Função principal para execução do módulo refatorado."""
    logger.info("🏛️ Iniciando ETL de relatórios CVM (versão refatorada)")
    
    try:
        # Usar orquestrador da nova arquitetura
        orchestrator = ETLOrchestrator()
        
        # Executar coleta de relatórios CVM
        result = orchestrator.run_cvm_reports_collection()
        
        if result and result.get("status") != "no_tickers":
            logger.info("✅ ETL de relatórios CVM concluído com sucesso")
            logger.info(f"📊 Resumo: {result['successful_collections']} sucessos, {result['failed_collections']} falhas")
        else:
            logger.warning("⚠️ Nenhum ticker encontrado para coleta CVM")
            
    except Exception as e:
        logger.exception(f"❌ Erro fatal no ETL de relatórios CVM: {e}")
        raise

def run_for_specific_tickers(tickers: List[str]):
    """Executa coleta para tickers específicos."""
    logger.info(f"📋 Coletando relatórios CVM para tickers específicos: {tickers}")
    
    try:
        orchestrator = ETLOrchestrator()
        result = orchestrator.run_cvm_reports_collection(tickers=tickers)
        
        if result:
            logger.info("✅ Coleta para tickers específicos concluída")
            logger.info(f"📊 Resumo: {result['successful_collections']} sucessos, {result['failed_collections']} falhas")
        else:
            logger.warning("⚠️ Nenhum resultado obtido")
            
    except Exception as e:
        logger.exception(f"❌ Erro na coleta para tickers específicos: {e}")
        raise

if __name__ == "__main__":
    run()
