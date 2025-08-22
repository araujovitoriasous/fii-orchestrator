POETRY=poetry

setup:
	$(POETRY) install

# 🚀 ETL (Nova Arquitetura)
etl-news-refactored:
	$(POETRY) run python -m fii_orchestrator.etl.rss_news_refactored

etl-prices-refactored:
	$(POETRY) run python -m fii_orchestrator.etl.b3_prices_refactored

etl-cvm-refactored:
	$(POETRY) run python -m fii_orchestrator.etl.cvm_reports_refactored

etl-pipeline-complete:
	$(POETRY) run python -c "from fii_orchestrator.etl.orchestrators import ETLOrchestrator; ETLOrchestrator().run_complete_etl_pipeline()"

etl-pipeline-funds:
	$(POETRY) run python -c "from fii_orchestrator.etl.orchestrators import ETLOrchestrator; ETLOrchestrator().run_funds_data_collection()"

etl-pipeline-news:
	$(POETRY) run python -c "from fii_orchestrator.etl.orchestrators import ETLOrchestrator; ETLOrchestrator().run_news_collection()"

etl-pipeline-cvm:
	$(POETRY) run python -c "from fii_orchestrator.etl.orchestrators import ETLOrchestrator; ETLOrchestrator().run_cvm_reports_collection()"

# 🔍 Qualidade e Análise
monitor-quality:
	$(POETRY) run python -c "from fii_orchestrator.etl.orchestrators import DataQualityOrchestrator; DataQualityOrchestrator().run_quality_validation()"

quality-validation:
	$(POETRY) run python -c "from fii_orchestrator.etl.orchestrators import DataQualityOrchestrator; DataQualityOrchestrator().run_quality_validation()"

performance-analysis:
	$(POETRY) run python -c "from fii_orchestrator.etl.orchestrators import PerformanceAnalysisOrchestrator; PerformanceAnalysisOrchestrator().run_performance_analysis()"

# 🧪 Testes
test:
	$(POETRY) run python -m pytest tests/ -v

test-domain:
	$(POETRY) run python -m pytest tests/test_domain.py -v

test-application:
	$(POETRY) run python -m pytest tests/test_application.py -v

test-coverage:
	$(POETRY) run python -m pytest tests/ --cov=fii_orchestrator --cov-report=html --cov-report=term

# 🔧 Desenvolvimento
ref-funds:
	$(POETRY) run python -m fii_orchestrator.etl.reference_funds

fmt:
	$(POETRY) run black src/ tests/
	$(POETRY) run ruff check --fix src/ tests/

lint:
	$(POETRY) run ruff check src/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -delete

# 🏗️ Arquitetura
arch-test:
	$(POETRY) run python -c "from fii_orchestrator.infrastructure.config import get_config, get_container; print('✅ Arquitetura funcionando')"

arch-validate:
	$(POETRY) run python -c "from fii_orchestrator.domain.entities import Fund, FundTicker; from fii_orchestrator.application.use_cases import CollectFundDataUseCase; print('✅ Camadas validadas')"

help:
	@echo "Comandos disponíveis:"
	@echo ""
	@echo "🚀 ETL (Nova Arquitetura):"
	@echo "  etl-news-refactored   - Coletar notícias (refatorado)"
	@echo "  etl-prices-refactored - Coletar preços (refatorado)"
	@echo "  etl-cvm-refactored    - Coletar CVM (refatorado)"
	@echo "  etl-pipeline-complete - Pipeline completo de ETL"
	@echo "  etl-pipeline-funds    - Coleta apenas de fundos"
	@echo "  etl-pipeline-news     - Coleta apenas de notícias"
	@echo "  etl-pipeline-cvm      - Coleta apenas de CVM"
	@echo ""
	@echo "🔍 Qualidade e Análise:"
	@echo "  monitor-quality       - Monitorar qualidade dos dados"
	@echo "  quality-validation    - Validação de qualidade"
	@echo "  performance-analysis  - Análise de performance"
	@echo ""
	@echo "🧪 Testes:"
	@echo "  test                  - Executar todos os testes"
	@echo "  test-domain           - Testes da camada de domínio"
	@echo "  test-application      - Testes da camada de aplicação"
	@echo "  test-coverage         - Testes com cobertura de código"
	@echo ""
	@echo "🔧 Desenvolvimento:"
	@echo "  setup                 - Instalar dependências"
	@echo "  ref-funds             - Gerar referência de fundos"
	@echo "  fmt                   - Formatar código"
	@echo "  lint                  - Verificar qualidade do código"
	@echo "  clean                 - Limpar arquivos temporários"
	@echo "  arch-test             - Testar arquitetura"
	@echo "  arch-validate         - Validar camadas"
	@echo "  help                  - Mostrar esta ajuda"