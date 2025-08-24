#!/usr/bin/env python3

import asyncio
import asyncpg
from fii_orchestrator.infrastructure.config import get_config

async def test_connection():
    config = get_config()
    print(f"🔍 Config: {config.postgres}")
    print(f"🔍 DSN: {config.postgres.dsn}")
    
    try:
        # Testar conexão direta
        print("🔌 Testando conexão direta...")
        conn = await asyncpg.connect(config.postgres.connection_string)
        print("✅ Conexão direta OK!")
        await conn.close()
        
        # Testar pool
        print("🏊 Testando pool...")
        pool = await asyncpg.create_pool(config.postgres.connection_string)
        print("✅ Pool criado OK!")
        await pool.close()
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_connection())
