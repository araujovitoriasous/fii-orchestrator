from abc import ABC, abstractmethod
from typing import Any

class Command(ABC):
    """Classe base para comandos CQRS."""
    pass

class CreateFundCommand(Command):
    """Comando para criar um novo fundo."""

    def __init__(self, ticker: str, cnpj: str, razao_social: str):
        self.ticker = ticker
        self.cnpj = cnpj
        self.razao_social = razao_social
