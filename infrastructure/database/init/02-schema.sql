-- Schema principal
CREATE SCHEMA IF NOT EXISTS fii_core;

-- Tabela de fundos
CREATE TABLE IF NOT EXISTS fii_core.funds (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(10) NOT NULL UNIQUE,
    cnpj VARCHAR(18) NOT NULL UNIQUE,
    razao_social VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true
);

-- Tabela de preços
CREATE TABLE IF NOT EXISTS fii_core.prices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    fund_id UUID NOT NULL REFERENCES fii_core.funds(id),
    date DATE NOT NULL,
    open_price DECIMAL(10,4) NOT NULL,
    close_price DECIMAL(10,4) NOT NULL,
    high_price DECIMAL(10,4) NOT NULL,
    low_price DECIMAL(10,4) NOT NULL,
    volume BIGINT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(fund_id, date)
);

-- Tabela de dividendos
CREATE TABLE IF NOT EXISTS fii_core.dividends (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    fund_id UUID NOT NULL REFERENCES fii_core.funds(id),
    payment_date DATE NOT NULL,
    amount DECIMAL(10,4) NOT NULL,
    type VARCHAR(50) DEFAULT 'rendimento',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Tabela de notícias
CREATE TABLE IF NOT EXISTS fii_core.news (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(500) NOT NULL,
    content TEXT,
    source VARCHAR(255),
    url VARCHAR(1000),
    published_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tags TEXT[]
);

-- Tabela de relacionamento fundo-notícia
CREATE TABLE IF NOT EXISTS fii_core.fund_news (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    fund_id UUID NOT NULL REFERENCES fii_core.funds(id),
    news_id UUID NOT NULL REFERENCES fii_core.news(id),
    relevance_score DECIMAL(3,2) DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(fund_id, news_id)
);

-- Índices para performance
CREATE INDEX IF NOT EXISTS idx_funds_ticker ON fii_core.funds(ticker);
CREATE INDEX IF NOT EXISTS idx_funds_cnpj ON fii_core.funds(cnpj);
CREATE INDEX IF NOT EXISTS idx_funds_active ON fii_core.funds(is_active);
CREATE INDEX IF NOT EXISTS idx_prices_fund_date ON fii_core.prices(fund_id, date);
CREATE INDEX IF NOT EXISTS idx_prices_date ON fii_core.prices(date);
CREATE INDEX IF NOT EXISTS idx_dividends_fund_date ON fii_core.dividends(fund_id, payment_date);
CREATE INDEX IF NOT EXISTS idx_dividends_date ON fii_core.dividends(payment_date);
CREATE INDEX IF NOT EXISTS idx_news_published ON fii_core.news(published_at);
CREATE INDEX IF NOT EXISTS idx_news_tags ON fii_core.news USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_fund_news_fund ON fii_core.fund_news(fund_id);
CREATE INDEX IF NOT EXISTS idx_fund_news_relevance ON fii_core.fund_news(relevance_score);

-- Triggers para updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_funds_updated_at 
    BEFORE UPDATE ON fii_core.funds 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Função para inserir fundo com validação
CREATE OR REPLACE FUNCTION insert_fund(
    p_ticker VARCHAR(10),
    p_cnpj VARCHAR(18),
    p_razao_social VARCHAR(255)
) RETURNS UUID AS $$
DECLARE
    v_fund_id UUID;
BEGIN
    -- Validar CNPJ (formato básico)
    IF p_cnpj !~ '^\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}$' THEN
        RAISE EXCEPTION 'CNPJ inválido: %', p_cnpj;
    END IF;
    
    -- Validar ticker (formato básico)
    IF p_ticker !~ '^[A-Z]{4}[0-9]{1,2}$' THEN
        RAISE EXCEPTION 'Ticker inválido: %', p_ticker;
    END IF;
    
    -- Inserir fundo
    INSERT INTO fii_core.funds (ticker, cnpj, razao_social)
    VALUES (p_ticker, p_cnpj, p_razao_social)
    RETURNING id INTO v_fund_id;
    
    RETURN v_fund_id;
END;
$$ LANGUAGE plpgsql;

-- Função para buscar estatísticas de fundo
CREATE OR REPLACE FUNCTION get_fund_stats(p_fund_id UUID)
RETURNS TABLE(
    total_prices INTEGER,
    total_dividends INTEGER,
    avg_price DECIMAL(10,4),
    total_dividend_amount DECIMAL(12,4),
    last_price_date DATE,
    last_dividend_date DATE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(p.id)::INTEGER as total_prices,
        COUNT(d.id)::INTEGER as total_dividends,
        AVG(p.close_price) as avg_price,
        COALESCE(SUM(d.amount), 0) as total_dividend_amount,
        MAX(p.date) as last_price_date,
        MAX(d.payment_date) as last_dividend_date
    FROM fii_core.funds f
    LEFT JOIN fii_core.prices p ON f.id = p.fund_id
    LEFT JOIN fii_core.dividends d ON f.id = d.fund_id
    WHERE f.id = p_fund_id
    GROUP BY f.id;
END;
$$ LANGUAGE plpgsql;

-- Comentários nas tabelas
COMMENT ON TABLE fii_core.funds IS 'Tabela principal de fundos imobiliários';
COMMENT ON TABLE fii_core.prices IS 'Histórico de preços dos fundos';
COMMENT ON TABLE fii_core.dividends IS 'Histórico de dividendos dos fundos';
COMMENT ON TABLE fii_core.news IS 'Notícias relacionadas ao mercado';
COMMENT ON TABLE fii_core.fund_news IS 'Relacionamento entre fundos e notícias';

-- Inserir dados de exemplo (opcional para desenvolvimento)
INSERT INTO fii_core.funds (ticker, cnpj, razao_social) VALUES
    ('HGLG11', '12.345.678/0001-90', 'HGLG Investimentos Ltda'),
    ('XPML11', '98.765.432/0001-10', 'XP Malls Fundo de Investimento Imobiliário')
ON CONFLICT (ticker) DO NOTHING;
