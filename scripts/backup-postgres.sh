#!/bin/bash

# Script de backup do PostgreSQL para FII Orchestrator
set -e

# Configurações
BACKUP_DIR="data/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="fii_orchestrator_${TIMESTAMP}.sql"
COMPRESSED_FILE="${BACKUP_FILE}.gz"
RETENTION_DAYS=30

echo "💾 Criando backup do PostgreSQL..."

# Verificar se Docker está rodando
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker não está rodando."
    exit 1
fi

# Verificar se PostgreSQL está rodando
if ! docker-compose ps postgres | grep -q "Up"; then
    echo "❌ PostgreSQL não está rodando. Inicie primeiro com: docker-compose up -d postgres"
    exit 1
fi

# Criar diretório de backup se não existir
mkdir -p "$BACKUP_DIR"

# Criar backup
echo "📦 Criando backup: ${BACKUP_FILE}..."
if docker-compose exec -T postgres pg_dump \
    -U fii_user \
    -d fii_orchestrator \
    --clean \
    --if-exists \
    --create \
    --verbose \
    --no-password \
    > "${BACKUP_DIR}/${BACKUP_FILE}"; then
    
    echo "✅ Backup criado com sucesso!"
    
    # Comprimir backup
    echo "🗜️  Comprimindo backup..."
    gzip "${BACKUP_DIR}/${BACKUP_FILE}"
    
    # Verificar tamanho do arquivo comprimido
    BACKUP_SIZE=$(du -h "${BACKUP_DIR}/${COMPRESSED_FILE}" | cut -f1)
    echo "📊 Tamanho do backup: ${BACKUP_SIZE}"
    
    # Limpar backups antigos
    echo "🧹 Limpando backups antigos (mais de ${RETENTION_DAYS} dias)..."
    find "${BACKUP_DIR}" -name "fii_orchestrator_*.sql.gz" -mtime +${RETENTION_DAYS} -delete
    
    echo ""
    echo "🎉 Backup concluído com sucesso!"
    echo "📁 Arquivo: ${BACKUP_DIR}/${COMPRESSED_FILE}"
    echo "📊 Tamanho: ${BACKUP_SIZE}"
    echo "🗓️  Data: $(date)"
    
else
    echo "❌ Erro ao criar backup!"
    exit 1
fi

# Listar backups disponíveis
echo ""
echo "📋 Backups disponíveis:"
ls -lh "${BACKUP_DIR}"/fii_orchestrator_*.sql.gz 2>/dev/null || echo "   Nenhum backup encontrado"

echo ""
echo "🔧 Para restaurar um backup:"
echo "   docker-compose exec -T postgres psql -U fii_user -d fii_orchestrator < backup_file.sql"
