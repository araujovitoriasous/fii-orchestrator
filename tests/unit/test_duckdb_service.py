import pytest
from unittest.mock import Mock, patch
from fii_orchestrator.serving.duckdb_service import DuckDBService

class TestDuckDBService:
    """Testes unitários para DuckDBService."""

    @pytest.fixture
    def mock_duckdb_connection(self):
        """Mock da conexão DuckDB."""
        mock_conn = Mock()
        mock_conn.execute.return_value.fetchall.return_value = []
        return mock_conn

    def test_get_funds_success(self, mock_duckdb_connection):
        """Testa busca de fundos com sucesso."""
        with patch("duckdb.connect", return_value=mock_duckdb_connection):
            service = DuckDBService()
            result = service.get_funds(limit=10, offset=0)
            assert result == []
