#!/bin/bash

# Script de configuração do PostgreSQL para FII Orchestrator
set -e

echo "🚀 Configurando PostgreSQL para FII Orchestrator..."

# Verificar se Docker está rodando
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker não está rodando. Inicie o Docker primeiro."
    exit 1
fi

# Verificar se docker-compose está disponível
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose não está instalado. Instale primeiro."
    exit 1
fi

# Criar diretórios necessários
echo "📁 Criando diretórios..."
mkdir -p data/postgres
mkdir -p data/backups
mkdir -p infrastructure/database/init
mkdir -p infrastructure/database/backups

# Verificar se arquivo .env existe
if [ ! -f .env ]; then
    echo "📝 Arquivo .env não encontrado. Copiando de env.example..."
    if [ -f env.example ]; then
        cp env.example .env
        echo "✅ Arquivo .env criado a partir de env.example"
        echo "🔑 Configure as variáveis de ambiente no arquivo .env se necessário"
    else
        echo "⚠️  Arquivo env.example não encontrado. Crie o arquivo .env manualmente."
    fi
else
    echo "✅ Arquivo .env já existe"
fi

# Verificar se docker-compose.yml existe
if [ ! -f docker-compose.yml ]; then
    echo "❌ docker-compose.yml não encontrado!"
    exit 1
fi

echo "🔑 Configurações padrão:"
echo "   - PostgreSQL: localhost:5432"
echo "   - Usuário: fii_user"
echo "   - Senha: fii_secure_password_2024"
echo "   - Banco: fii_orchestrator"
echo "   - PgAdmin: http://localhost:8080"
echo "   - Email: admin@fii.com"
echo "   - Senha: admin_secure_2024"

# Iniciar serviços
echo "🐳 Iniciando PostgreSQL..."
docker-compose up -d postgres

# Aguardar PostgreSQL estar pronto
echo "⏳ Aguardando PostgreSQL estar pronto..."
attempts=0
max_attempts=30

while [ $attempts -lt $max_attempts ]; do
    if docker-compose exec -T postgres pg_isready -U fii_user -d fii_orchestrator > /dev/null 2>&1; then
        echo "✅ PostgreSQL está pronto!"
        break
    fi
    
    attempts=$((attempts + 1))
    echo "⏳ Aguardando... (tentativa $attempts/$max_attempts)"
    sleep 2
done

if [ $attempts -eq $max_attempts ]; then
    echo "❌ Timeout aguardando PostgreSQL"
    echo "📋 Verificando logs..."
    docker-compose logs postgres
    exit 1
fi

# Iniciar PgAdmin
echo "🐳 Iniciando PgAdmin..."
docker-compose up -d pgadmin

# Aguardar PgAdmin estar pronto
echo "⏳ Aguardando PgAdmin estar pronto..."
sleep 10

# Verificar se os serviços estão rodando
echo "🔍 Verificando status dos serviços..."
docker-compose ps

echo ""
echo "🎉 PostgreSQL configurado com sucesso!"
echo ""
echo "📊 Acessos:"
echo "   - PostgreSQL: localhost:5432"
echo "   - PgAdmin: http://localhost:8080"
echo ""
echo "🔧 Comandos úteis:"
echo "   - Ver logs: docker-compose logs -f postgres"
echo "   - Parar: docker-compose stop"
echo "   - Reiniciar: docker-compose restart"
echo "   - Backup: ./scripts/backup-postgres.sh"
echo ""
echo "✅ Configuração concluída!"
