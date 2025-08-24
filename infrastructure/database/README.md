# 🗄️ PostgreSQL - FII Orchestrator

## 📋 Visão Geral

Este diretório contém toda a configuração e infraestrutura do PostgreSQL para o FII Orchestrator, seguindo as melhores práticas de DevOps.

## 🏗️ Arquitetura

### **Estrutura de Diretórios**
```
infrastructure/database/
├── init/                    # Scripts de inicialização
│   ├── 01-init.sql         # Configurações de performance e extensões
│   └── 02-schema.sql       # Schema completo do banco
├── backups/                 # Diretório para backups
└── README.md               # Esta documentação
```

### **Serviços Docker**
- **PostgreSQL 15**: Banco principal com configurações otimizadas
- **PgAdmin 4**: Interface web para administração

## 🚀 Configuração Rápida

### **1. Configurar Ambiente**
```bash
# Copiar arquivo de exemplo
cp env.example .env

# Editar variáveis se necessário
nano .env
```

### **2. Iniciar Serviços**
```bash
# Configuração inicial (primeira vez)
make postgres-setup

# Iniciar PostgreSQL
make postgres-start

# Verificar status
make postgres-status
```

### **3. Testar Conexão**
```bash
# Testar se está funcionando
make test-db-connection
```

## 🔧 Comandos Disponíveis

### **Gerenciamento de Serviços**
```bash
make postgres-start      # Iniciar PostgreSQL
make postgres-stop       # Parar PostgreSQL
make postgres-restart    # Reiniciar PostgreSQL
make postgres-status     # Ver status
make postgres-logs       # Ver logs
```

### **Operações de Dados**
```bash
make postgres-backup     # Criar backup
make test-db-connection  # Testar conexão
```

## 📊 Acessos

### **PostgreSQL**
- **Host**: localhost
- **Porta**: 5432
- **Banco**: fii_orchestrator
- **Usuário**: fii_user
- **Senha**: fii_secure_password_2024

### **PgAdmin**
- **URL**: http://localhost:8080
- **Email**: admin@fii.com
- **Senha**: admin_secure_2024

## 🗂️ Schema do Banco

### **Tabelas Principais**
- **`fii_core.funds`**: Fundos imobiliários
- **`fii_core.prices`**: Histórico de preços
- **`fii_core.dividends`**: Histórico de dividendos
- **`fii_core.news`**: Notícias do mercado
- **`fii_core.fund_news`**: Relacionamento fundo-notícia

### **Índices e Performance**
- Índices otimizados para consultas por ticker e data
- Índice GIN para tags de notícias
- Configurações de performance otimizadas

### **Funções SQL**
- **`insert_fund()`**: Inserção com validação
- **`get_fund_stats()`**: Estatísticas de fundos
- **`update_updated_at_column()`**: Trigger para updated_at

## 🔒 Segurança

### **Configurações de Segurança**
- Usuário dedicado para aplicação
- SSL habilitado por padrão
- Conexões limitadas por pool
- Timeouts configurados

### **Backup e Recuperação**
- Backups automáticos com compressão
- Retenção configurável (padrão: 30 dias)
- Scripts de backup automatizados

## 📈 Monitoramento

### **Health Checks**
- Verificação automática a cada 30 segundos
- Logs de conexão e desconexão
- Estatísticas do pool de conexões

### **Métricas Disponíveis**
- Tamanho do pool de conexões
- Conexões livres
- Status de saúde
- Última verificação

## 🛠️ Desenvolvimento

### **Conectar via Python**
```python
from fii_orchestrator.infrastructure.database.postgres_service import PostgresService, PostgresConfig
from fii_orchestrator.infrastructure.config import get_config

config = get_config()
service = PostgresService(config.postgres)
await service.initialize()

# Executar queries
result = await service.fetch("SELECT * FROM fii_core.funds")
```

### **Conectar via psql**
```bash
# Conectar ao banco
docker-compose exec postgres psql -U fii_user -d fii_orchestrator

# Ver tabelas
\dt fii_core.*

# Ver dados de exemplo
SELECT * FROM fii_core.funds;
```

## 🚨 Troubleshooting

### **Problemas Comuns**

#### **PostgreSQL não inicia**
```bash
# Verificar logs
make postgres-logs

# Verificar se Docker está rodando
docker info

# Verificar se porta 5432 está livre
lsof -i :5432
```

#### **Erro de conexão**
```bash
# Verificar se serviço está rodando
make postgres-status

# Testar conexão
make test-db-connection

# Verificar variáveis de ambiente
cat .env
```

#### **Problemas de permissão**
```bash
# Verificar permissões dos diretórios
ls -la data/postgres/
ls -la infrastructure/database/

# Corrigir permissões se necessário
chmod 755 data/postgres/
chmod 755 infrastructure/database/
```

## 📚 Recursos Adicionais

### **Documentação Oficial**
- [PostgreSQL 15 Documentation](https://www.postgresql.org/docs/15/)
- [asyncpg Python Driver](https://magicstack.github.io/asyncpg/)
- [Docker PostgreSQL](https://hub.docker.com/_/postgres)

### **Ferramentas Úteis**
- **pgAdmin**: Interface web para administração
- **psql**: Cliente de linha de comando
- **pg_dump/pg_restore**: Backup e restauração

## 🤝 Contribuição

Para contribuir com melhorias na configuração do PostgreSQL:

1. Teste as mudanças localmente
2. Atualize a documentação
3. Execute os testes de conexão
4. Crie um pull request com descrição detalhada

---

**⚠️ Importante**: Nunca commite senhas ou configurações sensíveis no repositório. Use sempre variáveis de ambiente.
