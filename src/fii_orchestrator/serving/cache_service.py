import redis
import json
import hashlib
from typing import Any, Optional
from loguru import logger

class RedisCacheService:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.default_ttl = 300  # 5 minutos

    def get(self, key: str) -> Optional[Any]:
        """Busca valor no cache."""
        try:
            value = self.redis_client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.warning(f"⚠️ Erro ao buscar cache: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Define valor no cache com TTL."""
        try:
            ttl = ttl or self.default_ttl
            self.redis_client.setex(key, ttl, json.dumps(value))
            return True
        except Exception as e:
            logger.warning(f"⚠️ Erro ao definir cache: {e}")
            return False
