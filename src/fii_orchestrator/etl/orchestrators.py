"""
Orquestradores de ETL.
Coordena a execução dos processos de extração, transformação e carregamento.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger
import time

from fii_orchestrator.domain.entities import Fund
from fii_orchestrator.application.use_cases import (
    CollectFundDataUseCase, CollectNewsUseCase, 
    AnalyzeFundPerformanceUseCase, ValidateDataQualityUseCase
)
from fii_orchestrator.infrastructure.adapters import DataProviderFactory
from fii_orchestrator.infrastructure.config import get_config, get_container
from fii_orchestrator.utils.parallel import ParallelProcessor, AsyncProcessor
from fii_orchestrator.utils.io import write_metadata

class ETLOrchestrator:
    """Orquestrador principal de ETL."""
    
    def __init__(self):
        self.config = get_config()
        self.container = get_container()
        self.parallel_processor = ParallelProcessor(
            max_workers=self.config.processing.max_workers,
            chunk_size=self.config.processing.chunk_size
        )
    
    def run_funds_data_collection(
        self, 
        tickers: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Executa coleta de dados de fundos.
        
        Args:
            tickers: Lista de tickers para coletar (None para todos)
            start_date: Data de início (None para padrão)
            end_date: Data de fim (None para hoje)
            
        Returns:
            Dicionário com resultados da coleta
        """
        logger.info("🚀 Iniciando orquestração de coleta de dados de fundos")
        
        # Configurar datas padrão
        if not start_date:
            start_date = datetime(2018, 1, 1)
        if not end_date:
            end_date = datetime.now()
        
        # Obter lista de tickers
        if not tickers:
            fund_repo = self.container.get_repository('fund')
            funds = fund_repo.get_all()
            tickers = [str(fund.ticker) for fund in funds]
        
        if not tickers:
            logger.warning("Nenhum ticker encontrado para coleta")
            return {"status": "no_tickers", "message": "Nenhum ticker disponível"}
        
        logger.info(f"📊 Coletando dados para {len(tickers)} fundos")
        logger.info(f"📅 Período: {start_date.date()} - {end_date.date()}")
        
        # Configurar provedor de dados
        data_provider = DataProviderFactory.create_yahoo_finance(
            rate_limit=self.config.api.yahoo_finance_rate_limit
        )
        
        # Criar caso de uso
        collect_use_case = CollectFundDataUseCase(
            fund_repo=self.container.get_repository('fund'),
            price_repo=self.container.get_repository('price'),
            dividend_repo=self.container.get_repository('dividend'),
            data_provider=data_provider
        )
        
        # Executar coleta
        results = []
        successful = 0
        failed = 0
        
        def process_ticker(ticker: str) -> Optional[Dict[str, Any]]:
            try:
                result = collect_use_case.execute(ticker, start_date, end_date)
                return result
            except Exception as e:
                logger.error(f"Erro ao processar {ticker}: {e}")
                return None
        
        if self.config.processing.use_parallel:
            logger.info(f"⚡ Usando processamento paralelo ({self.config.processing.max_workers} workers)")
            results = self.parallel_processor.process_chunks(tickers, process_ticker)
        else:
            logger.info("🐌 Usando processamento sequencial")
            for ticker in tickers:
                result = process_ticker(ticker)
                if result:
                    results.append(result)
                    successful += 1
                else:
                    failed += 1
                time.sleep(self.config.api.yahoo_finance_rate_limit)
        
        # Contar resultados
        successful = len([r for r in results if r])
        failed = len(tickers) - successful
        
        # Salvar metadados
        metadata = {
            "collection_start": datetime.now().isoformat(),
            "total_tickers": len(tickers),
            "successful_collections": successful,
            "failed_collections": failed,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "processing_mode": "parallel" if self.config.processing.use_parallel else "sequential",
            "results": results
        }
        
        write_metadata(
            self.config.database.meta_dir / "etl", 
            "funds_data_collection", 
            metadata
        )
        
        logger.info(f"✅ Coleta concluída: {successful} sucessos, {failed} falhas")
        return metadata
    
    def run_news_collection(self, sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Executa coleta de notícias RSS.
        
        Args:
            sources: Lista de fontes RSS (None para configuração padrão)
            
        Returns:
            Dicionário com resultados da coleta
        """
        logger.info("📰 Iniciando orquestração de coleta de notícias")
        
        if not sources:
            sources = self.config.rss.sources
        
        logger.info(f"📡 Coletando de {len(sources)} fontes RSS")
        
        # Configurar provedor de dados
        data_provider = DataProviderFactory.create_rss_feed(
            timeout=self.config.api.cvm_timeout
        )
        
        # Criar caso de uso
        collect_news_use_case = CollectNewsUseCase(
            news_repo=self.container.get_repository('news'),
            data_provider=data_provider
        )
        
        try:
            result = collect_news_use_case.execute(sources)
            
            # Salvar metadados
            metadata = {
                "collection_date": datetime.now().isoformat(),
                "sources": sources,
                "result": result
            }
            
            write_metadata(
                self.config.database.meta_dir / "etl", 
                "news_collection", 
                metadata
            )
            
            logger.info(f"✅ Coleta de notícias concluída: {result['news_collected']} itens")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro na coleta de notícias: {e}")
            raise
    
    def run_cvm_reports_collection(self, tickers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Executa coleta de relatórios CVM.
        
        Args:
            tickers: Lista de tickers para coletar (None para todos)
            
        Returns:
            Dicionário com resultados da coleta
        """
        logger.info("🏛️ Iniciando orquestração de coleta de relatórios CVM")
        
        # Obter lista de tickers
        if not tickers:
            fund_repo = self.container.get_repository('fund')
            funds = fund_repo.get_all()
            tickers = [str(fund.ticker) for fund in funds]
        
        if not tickers:
            logger.warning("Nenhum ticker encontrado para coleta CVM")
            return {"status": "no_tickers", "message": "Nenhum ticker disponível"}
        
        logger.info(f"📋 Coletando relatórios para {len(tickers)} fundos")
        
        # Configurar provedor de dados
        data_provider = DataProviderFactory.create_cvm_api(
            timeout=self.config.api.cvm_timeout,
            max_retries=self.config.api.max_retries
        )
        
        # Executar coleta
        results = []
        successful = 0
        failed = 0
        
        for i, ticker in enumerate(tickers, 1):
            try:
                logger.info(f"📊 Processando {i}/{len(tickers)}: {ticker}")
                
                report_data = data_provider.fetch_fund_reports(ticker)
                results.append(report_data)
                successful += 1
                
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Erro ao processar {ticker}: {e}")
                failed += 1
                results.append({
                    "ticker": ticker,
                    "status": "error",
                    "error": str(e)
                })
        
        # Salvar metadados
        metadata = {
            "collection_date": datetime.now().isoformat(),
            "total_tickers": len(tickers),
            "successful_collections": successful,
            "failed_collections": failed,
            "results": results
        }
        
        write_metadata(
            self.config.database.meta_dir / "etl", 
            "cvm_reports_collection", 
            metadata
        )
        
        logger.info(f"✅ Coleta CVM concluída: {successful} sucessos, {failed} falhas")
        return metadata
    
    def run_complete_etl_pipeline(self) -> Dict[str, Any]:
        """
        Executa pipeline completo de ETL.
        
        Returns:
            Dicionário com resultados de todos os processos
        """
        logger.info("🚀 Iniciando pipeline completo de ETL")
        
        start_time = datetime.now()
        results = {}
        
        try:
            # 1. Coleta de dados de fundos
            logger.info("📊 Etapa 1/4: Coleta de dados de fundos")
            results['funds_data'] = self.run_funds_data_collection()
            
            # 2. Coleta de notícias
            logger.info("📰 Etapa 2/4: Coleta de notícias RSS")
            results['news'] = self.run_news_collection()
            
            # 3. Coleta de relatórios CVM
            logger.info("🏛️ Etapa 3/4: Coleta de relatórios CVM")
            results['cvm_reports'] = self.run_cvm_reports_collection()
            
            # 4. Validação de qualidade
            logger.info("🔍 Etapa 4/4: Validação de qualidade dos dados")
            quality_use_case = self.container.get_use_case('validate_quality')
            results['quality_validation'] = quality_use_case.execute()
            
            # Resumo final
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            pipeline_summary = {
                "pipeline_start": start_time.isoformat(),
                "pipeline_end": end_time.isoformat(),
                "duration_seconds": duration,
                "status": "success",
                "results": results
            }
            
            # Salvar metadados do pipeline
            write_metadata(
                self.config.database.meta_dir / "etl", 
                "complete_pipeline", 
                pipeline_summary
            )
            
            logger.info(f"🎉 Pipeline completo concluído em {duration:.2f} segundos")
            return pipeline_summary
            
        except Exception as e:
            logger.error(f"❌ Erro no pipeline de ETL: {e}")
            
            # Salvar metadados de erro
            error_summary = {
                "pipeline_start": start_time.isoformat(),
                "pipeline_end": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
                "partial_results": results
            }
            
            write_metadata(
                self.config.database.meta_dir / "etl", 
                "complete_pipeline_error", 
                error_summary
            )
            
            raise

class DataQualityOrchestrator:
    """Orquestrador para validação de qualidade dos dados."""
    
    def __init__(self):
        self.config = get_config()
        self.container = get_container()
    
    def run_quality_validation(self) -> Dict[str, Any]:
        """
        Executa validação completa de qualidade.
        
        Returns:
            Dicionário com resultados da validação
        """
        logger.info("🔍 Iniciando orquestração de validação de qualidade")
        
        # Obter caso de uso
        quality_use_case = self.container.get_use_case('validate_quality')
        
        try:
            result = quality_use_case.execute()
            
            # Salvar metadados
            write_metadata(
                self.config.database.meta_dir / "quality", 
                "quality_validation", 
                result
            )
            
            logger.info("✅ Validação de qualidade concluída")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro na validação de qualidade: {e}")
            raise
    
    def run_data_freshness_check(self) -> Dict[str, Any]:
        """
        Executa verificação de frescor dos dados.
        
        Returns:
            Dicionário com resultados da verificação
        """
        logger.info("⏰ Verificando frescor dos dados")
        
        from fii_orchestrator.utils.monitoring import DataQualityMonitor
        
        monitor = DataQualityMonitor()
        report = monitor.generate_quality_report()
        
        # Salvar metadados
        write_metadata(
            self.config.database.meta_dir / "quality", 
            "freshness_check", 
            report
        )
        
        logger.info("✅ Verificação de frescor concluída")
        return report

class PerformanceAnalysisOrchestrator:
    """Orquestrador para análise de performance de fundos."""
    
    def __init__(self):
        self.config = get_config()
        self.container = get_container()
    
    def run_performance_analysis(
        self, 
        tickers: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Executa análise de performance de fundos.
        
        Args:
            tickers: Lista de tickers para analisar (None para todos)
            start_date: Data de início (None para 1 ano atrás)
            end_date: Data de fim (None para hoje)
            
        Returns:
            Dicionário com resultados da análise
        """
        logger.info("📈 Iniciando análise de performance de fundos")
        
        # Configurar datas padrão
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)
        if not end_date:
            end_date = datetime.now()
        
        # Obter lista de tickers
        if not tickers:
            fund_repo = self.container.get_repository('fund')
            funds = fund_repo.get_all()
            tickers = [str(fund.ticker) for fund in funds]
        
        if not tickers:
            logger.warning("Nenhum ticker encontrado para análise")
            return {"status": "no_tickers", "message": "Nenhum ticker disponível"}
        
        logger.info(f"📊 Analisando performance de {len(tickers)} fundos")
        logger.info(f"📅 Período: {start_date.date()} - {end_date.date()}")
        
        # Obter caso de uso
        performance_use_case = self.container.get_use_case('analyze_performance')
        
        # Executar análise
        results = []
        successful = 0
        failed = 0
        
        for ticker in tickers:
            try:
                result = performance_use_case.execute(ticker, start_date, end_date)
                results.append(result)
                successful += 1
                
            except Exception as e:
                logger.error(f"❌ Erro ao analisar {ticker}: {e}")
                failed += 1
                results.append({
                    "ticker": ticker,
                    "status": "error",
                    "error": str(e)
                })
        
        # Resumo da análise
        analysis_summary = {
            "analysis_date": datetime.now().isoformat(),
            "total_tickers": len(tickers),
            "successful_analyses": successful,
            "failed_analyses": failed,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "results": results
        }
        
        # Salvar metadados
        write_metadata(
            self.config.database.meta_dir / "analysis", 
            "performance_analysis", 
            analysis_summary
        )
        
        logger.info(f"✅ Análise de performance concluída: {successful} sucessos, {failed} falhas")
        return analysis_summary
