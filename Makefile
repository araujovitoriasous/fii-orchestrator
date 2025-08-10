POETRY=poetry

setup:
	$(POETRY) install

etl-news:
	$(POETRY) run python -m fii_orchestrator.etl.rss_news

etl-prices:
	$(POETRY) run python -m fii_orchestrator.etl.b3_prices

etl-cvm:
	$(POETRY) run python -m fii_orchestrator.etl.cvm_reports

fmt:
	$(POETRY) run ruff check --fix .
	$(POETRY) run black .
