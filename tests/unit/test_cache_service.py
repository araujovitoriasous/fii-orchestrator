import pytest
from unittest.mock import Mock, patch
from fii_orchestrator.serving.cache_service import RedisCacheService

class TestRedisCacheService:
    """Testes unitários para RedisCacheService."""

    def test_set_and_get_cache(self):
        """Testa operações de set e get do cache."""
        with patch("redis.from_url") as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client

            cache_service = RedisCacheService()
            cache_service.set("test_key", "test_value")
            mock_client.setex.assert_called_once()
