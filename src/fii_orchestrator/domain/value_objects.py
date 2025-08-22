"""
Value Objects do domínio de Fundos Imobiliários.

Este módulo contém os value objects que representam conceitos
fundamentais do domínio de FIIs, garantindo invariantes e
regras de negócio através de validações automáticas.
"""

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Union, Optional

@dataclass(frozen=True)
class FundTicker:
    """
    Value Object representando o ticker de um Fundo Imobiliário.
    
    Um ticker válido deve seguir o padrão ABCD11 onde:
    - ABCD: 4 letras maiúsculas (código do fundo)
    - 11: 2 dígitos (tipo de fundo)
    
    Examples:
        >>> FundTicker("HGLG11")
        FundTicker(value='HGLG11')
        >>> FundTicker("INVALID")
        ValueError: Ticker inválido: INVALID
    
    Attributes:
        value: String representando o ticker do fundo
        
    Raises:
        ValueError: Se o ticker não seguir o formato esperado
    """
    value: str
    
    def __post_init__(self):
        if not self._is_valid_format(self.value):
            raise ValueError(f"Ticker inválido: {self.value}")
    
    @staticmethod
    def _is_valid_format(ticker: str) -> bool:
        """
        Valida se o formato do ticker está correto.
        
        Args:
            ticker: String a ser validada
            
        Returns:
            True se o formato for válido, False caso contrário
        """
        pattern = r"^[A-Z]{4}\d{2}$"
        return bool(re.match(pattern, ticker))
    
    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return f"FundTicker(value='{self.value}')"

@dataclass(frozen=True)
class CNPJ:
    """
    Value Object representando um CNPJ válido.
    
    Valida o formato e os dígitos verificadores do CNPJ
    seguindo as regras da Receita Federal.
    
    Examples:
        >>> CNPJ("08.441.966/0001-53")
        CNPJ(value='08.441.966/0001-53')
        >>> CNPJ("11.111.111/1111-11")
        ValueError: CNPJ inválido: todos os dígitos são iguais
        
    Attributes:
        value: String representando o CNPJ
        
    Raises:
        ValueError: Se o CNPJ não for válido
    """
    value: str
    
    def __post_init__(self):
        if not self._is_valid_cnpj(self.value):
            raise ValueError(f"CNPJ inválido: {self.value}")
    
    @staticmethod
    def _is_valid_cnpj(cnpj: str) -> bool:
        """
        Valida se o CNPJ é válido.
        
        Args:
            cnpj: String do CNPJ a ser validada
            
        Returns:
            True se o CNPJ for válido, False caso contrário
        """
        # Remove caracteres especiais
        cnpj_clean = re.sub(r'[^\d]', '', cnpj)
        
        # Verifica comprimento
        if len(cnpj_clean) != 14:
            return False
        
        # Verifica se todos os dígitos são iguais
        if len(set(cnpj_clean)) == 1:
            return False
        
        # Validação dos dígitos verificadores
        return CNPJ._validate_check_digits(cnpj_clean)
    
    @staticmethod
    def _validate_check_digits(cnpj: str) -> bool:
        """
        Valida os dígitos verificadores do CNPJ.
        
        Args:
            cnpj: CNPJ sem formatação
            
        Returns:
            True se os dígitos verificadores forem válidos
        """
        # Primeiro dígito verificador
        weights1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
        sum1 = sum(int(cnpj[i]) * weights1[i] for i in range(12))
        digit1 = (11 - (sum1 % 11)) % 10
        
        if int(cnpj[12]) != digit1:
            return False
        
        # Segundo dígito verificador
        weights2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
        sum2 = sum(int(cnpj[i]) * weights2[i] for i in range(13))
        digit2 = (11 - (sum2 % 11)) % 10
        
        return int(cnpj[13]) == digit2
    
    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return f"CNPJ(value='{self.value}')"

@dataclass(frozen=True)
class Money:
    """
    Value Object para valores monetários.
    
    Garante que valores monetários sejam sempre positivos
    e permite operações matemáticas seguras entre valores
    da mesma moeda.
    
    Examples:
        >>> Money(100.50)
        Money(amount=Decimal('100.50'), currency='BRL')
        >>> Money(-50.00)
        ValueError: Valor monetário não pode ser negativo
        >>> Money(100.00) + Money(50.00)
        Money(amount=Decimal('150.00'), currency='BRL')
        
    Attributes:
        amount: Valor decimal do dinheiro
        currency: Código da moeda (padrão: BRL)
        
    Raises:
        ValueError: Se o valor for negativo
        ValueError: Se tentar operar com moedas diferentes
    """
    amount: Decimal
    currency: str = "BRL"
    
    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("Valor monetário não pode ser negativo")
    
    def __add__(self, other: 'Money') -> 'Money':
        """
        Soma dois valores monetários.
        
        Args:
            other: Outro valor monetário
            
        Returns:
            Nova instância com a soma dos valores
            
        Raises:
            ValueError: Se as moedas forem diferentes
        """
        if self.currency != other.currency:
            raise ValueError("Moedas diferentes não podem ser somadas")
        return Money(self.amount + other.amount, self.currency)
    
    def __sub__(self, other: 'Money') -> 'Money':
        """
        Subtrai dois valores monetários.
        
        Args:
            other: Outro valor monetário
            
        Returns:
            Nova instância com a diferença dos valores
            
        Raises:
            ValueError: Se as moedas forem diferentes
            ValueError: Se o resultado for negativo
        """
        if self.currency != other.currency:
            raise ValueError("Moedas diferentes não podem ser subtraídas")
        result = self.amount - other.amount
        if result < 0:
            raise ValueError("Resultado da subtração não pode ser negativo")
        return Money(result, self.currency)
    
    def __mul__(self, multiplier: Union[int, float, Decimal]) -> 'Money':
        """
        Multiplica o valor por um multiplicador.
        
        Args:
            multiplier: Número pelo qual multiplicar
            
        Returns:
            Nova instância com o valor multiplicado
        """
        return Money(self.amount * Decimal(str(multiplier)), self.currency)
    
    def __truediv__(self, divisor: Union[int, float, Decimal]) -> 'Money':
        """
        Divide o valor por um divisor.
        
        Args:
            divisor: Número pelo qual dividir
            
        Returns:
            Nova instância com o valor dividido
            
        Raises:
            ValueError: Se o divisor for zero
        """
        if divisor == 0:
            raise ValueError("Divisor não pode ser zero")
        return Money(self.amount / Decimal(str(divisor)), self.currency)
    
    def __str__(self) -> str:
        return f"{self.amount:.2f} {self.currency}"
    
    def __repr__(self) -> str:
        return f"Money(amount={self.amount}, currency='{self.currency}')"
    
    def to_float(self) -> float:
        """
        Converte o valor para float.
        
        Returns:
            Valor como float
        """
        return float(self.amount)
    
    def is_zero(self) -> bool:
        """
        Verifica se o valor é zero.
        
        Returns:
            True se o valor for zero
        """
        return self.amount == 0

@dataclass(frozen=True)
class Percentage:
    """
    Value Object para valores percentuais.
    
    Garante que valores percentuais estejam no intervalo [0, 100]
    e permite operações matemáticas seguras.
    
    Examples:
        >>> Percentage(5.5)
        Percentage(value=Decimal('5.5'))
        >>> Percentage(150.0)
        ValueError: Percentual deve estar entre 0 e 100
        
    Attributes:
        value: Valor decimal do percentual (0-100)
        
    Raises:
        ValueError: Se o valor estiver fora do intervalo [0, 100]
    """
    value: Decimal
    
    def __post_init__(self):
        if not (0 <= self.value <= 100):
            raise ValueError("Percentual deve estar entre 0 e 100")
    
    def to_decimal(self) -> Decimal:
        """
        Converte para decimal (0.0 a 1.0).
        
        Returns:
            Valor como decimal entre 0.0 e 1.0
        """
        return self.value / 100
    
    def to_float(self) -> float:
        """
        Converte para float (0.0 a 1.0).
        
        Returns:
            Valor como float entre 0.0 e 1.0
        """
        return float(self.to_decimal())
    
    def __str__(self) -> str:
        return f"{self.value:.2f}%"
    
    def __repr__(self) -> str:
        return f"Percentage(value={self.value})"

@dataclass(frozen=True)
class PositiveFloat:
    """
    Value Object para valores float positivos.
    
    Garante que valores sejam sempre positivos e permite
    operações matemáticas seguras.
    
    Examples:
        >>> PositiveFloat(1.05)
        PositiveFloat(value=1.05)
        >>> PositiveFloat(-0.5)
        ValueError: Valor deve ser positivo
        
    Attributes:
        value: Valor float positivo
        
    Raises:
        ValueError: Se o valor não for positivo
    """
    value: float
    
    def __post_init__(self):
        if self.value <= 0:
            raise ValueError("Valor deve ser positivo")
    
    def __add__(self, other: Union[int, float, 'PositiveFloat']) -> 'PositiveFloat':
        """
        Soma com outro valor.
        
        Args:
            other: Valor a ser somado
            
        Returns:
            Nova instância com a soma
        """
        if isinstance(other, PositiveFloat):
            return PositiveFloat(self.value + other.value)
        return PositiveFloat(self.value + float(other))
    
    def __sub__(self, other: Union[int, float, 'PositiveFloat']) -> 'PositiveFloat':
        """
        Subtrai outro valor.
        
        Args:
            other: Valor a ser subtraído
            
        Returns:
            Nova instância com a subtração
            
        Raises:
            ValueError: Se o resultado for negativo
        """
        if isinstance(other, PositiveFloat):
            result = self.value - other.value
        else:
            result = self.value - float(other)
        
        if result <= 0:
            raise ValueError("Resultado da subtração deve ser positivo")
        
        return PositiveFloat(result)
    
    def __mul__(self, multiplier: Union[int, float]) -> 'PositiveFloat':
        """
        Multiplica por um multiplicador.
        
        Args:
            multiplier: Número pelo qual multiplicar
            
        Returns:
            Nova instância com o valor multiplicado
        """
        return PositiveFloat(self.value * float(multiplier))
    
    def __truediv__(self, divisor: Union[int, float]) -> 'PositiveFloat':
        """
        Divide por um divisor.
        
        Args:
            divisor: Número pelo qual dividir
            
        Returns:
            Nova instância com o valor dividido
            
        Raises:
            ValueError: Se o divisor for zero ou se o resultado for negativo
        """
        if divisor == 0:
            raise ValueError("Divisor não pode ser zero")
        
        result = self.value / float(divisor)
        if result <= 0:
            raise ValueError("Resultado da divisão deve ser positivo")
        
        return PositiveFloat(result)
    
    def __str__(self) -> str:
        return f"{self.value:.4f}"
    
    def __repr__(self) -> str:
        return f"PositiveFloat(value={self.value})"
    
    def to_float(self) -> float:
        """
        Converte para float.
        
        Returns:
            Valor como float
        """
        return self.value
