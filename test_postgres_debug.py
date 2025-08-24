#!/usr/bin/env python3

import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

print("🔍 Debugando configuração PostgreSQL:")
print(f"POSTGRES_HOST: {os.getenv('POSTGRES_HOST')}")
print(f"POSTGRES_PORT: {os.getenv('POSTGRES_PORT')}")
print(f"POSTGRES_DB: {os.getenv('POSTGRES_DB')}")
print(f"POSTGRES_USER: {os.getenv('POSTGRES_USER')}")
print(f"POSTGRES_PASSWORD: {os.getenv('POSTGRES_PASSWORD')}")
print(f"POSTGRES_SSL_MODE: {os.getenv('POSTGRES_SSL_MODE')}")

print("\n🔧 Testando configuração Python:")
try:
    from fii_orchestrator.infrastructure.config import get_config
    config = get_config()
    print(f"✅ Config carregada: {config}")
    print(f"✅ Postgres config: {config.postgres}")
    print(f"✅ DSN: {config.postgres.dsn}")
    print(f"✅ Connection string: {config.postgres.connection_string}")
except Exception as e:
    print(f"❌ Erro ao carregar config: {e}")
    import traceback
    traceback.print_exc()
