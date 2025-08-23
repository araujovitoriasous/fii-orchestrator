import pytest
from fastapi.testclient import TestClient
from fii_orchestrator.presentation.api import app

class TestAPIIntegration:
    """Testes de integração para a API."""

    def test_root_endpoint(self):
        """Testa o endpoint raiz da API."""
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
