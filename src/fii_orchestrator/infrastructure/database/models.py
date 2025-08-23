from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Fund(Base):
    """Modelo para fundos imobiliários."""
    __tablename__ = "funds"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(10), unique=True, index=True, nullable=False)
    cnpj = Column(String(18), unique=True, index=True, nullable=False)
    razao_social = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
