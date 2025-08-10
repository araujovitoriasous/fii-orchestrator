POETRY=poetry

setup:
	$(POETRY) install

etl-news:
	$(POETRY) run python -m fii_orchestrator.etl.rss_news

etl-prices:
	poetry run python -m fii_orchestrator.etl.b3_prices

etl-cvm:
	$(POETRY) run python -m fii_orchestrator.etl.cvm_reports

fmt:
	$(POETRY) run ruff check --fix .
	$(POETRY) run black .
ref-funds:
	poetry run python -m fii_orchestrator.etl.reference_funds