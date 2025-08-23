#!/usr/bin/env python3
"""Script principal para executar a API FastAPI do FII Orchestrator."""

import uvicorn
from fii_orchestrator.presentation.api import app
from fii_orchestrator.presentation.config import APISettings

settings = APISettings()

if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)
