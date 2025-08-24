-- Configuração de acesso externo para o usuário fii_user
-- Este script configura as permissões necessárias para conexões externas

-- Garantir que o usuário fii_user existe e tem as permissões corretas
DO $$
BEGIN
    -- Verificar se o usuário existe
    IF NOT EXISTS (SELECT 1 FROM pg_user WHERE usename = 'fii_user') THEN
        CREATE USER fii_user WITH PASSWORD 'fii_secure_password_2024';
    END IF;
    
    -- Garantir que o usuário tem acesso ao banco
    IF NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'fii_orchestrator') THEN
        CREATE DATABASE fii_orchestrator OWNER postgres;
    END IF;
END
$$;

-- Conceder todas as permissões necessárias ao usuário fii_user
GRANT ALL PRIVILEGES ON DATABASE fii_orchestrator TO fii_user;
GRANT ALL PRIVILEGES ON SCHEMA fii_core TO fii_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA fii_core TO fii_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA fii_core TO fii_user;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA fii_core TO fii_user;

-- Configurar permissões para tabelas futuras
ALTER DEFAULT PRIVILEGES IN SCHEMA fii_core GRANT ALL ON TABLES TO fii_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA fii_core GRANT ALL ON SEQUENCES TO fii_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA fii_core GRANT ALL ON FUNCTIONS TO fii_user;

-- Garantir que o usuário pode conectar
ALTER USER fii_user WITH LOGIN;

-- Configurar pg_hba.conf para permitir conexões externas
-- Nota: Estas configurações serão aplicadas via variáveis de ambiente

-- Verificar se as configurações foram aplicadas
SELECT 
    usename,
    usecreatedb,
    usesuper,
    usebypassrls,
    valuntil
FROM pg_user 
WHERE usename = 'fii_user';

-- Verificar permissões do banco
SELECT 
    datname,
    datacl
FROM pg_database 
WHERE datname = 'fii_orchestrator';
