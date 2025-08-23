import logging
import sys
from pathlib import Path
from loguru import logger

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Configura logging estruturado com Loguru."""
    # Remover handlers padrão
    logger.remove()

    # Handler para console
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
