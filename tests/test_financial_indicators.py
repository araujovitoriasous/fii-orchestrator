
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import from the processor
from etl.processors.financial_indicators import process_financial_indicators

def test_integration():
    print("Creating mock data...")
    prices = pd.DataFrame({
        "ticker": ["HGLG11", "KNRI11"],
        # Mocking generic dates
        "data": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-01")],
        "pvp": [1.05, 0.95],
        "vacancia": [0.05, 0.02],
        "dy_12m": [0.08, 0.09],
        "vp_cota": [100.0, 150.0],
        "tipo_gestao": ["Ativa", "Ativa"]
    })

    funds = pd.DataFrame({
        "ticker": ["HGLG11", "KNRI11"],
        "data_referencia": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-01")],
        "Lucro_Caixa_Trimestral": [1000, 2000],
        "Receita_Caixa": [5000, 8000],
        "Liquidez_Caixa": [200, 500],
        "Taxa_Administracao": [0.01, 0.01],
        "Passivo_Total": [100, 200],
        "Contas_Receber_Aluguel": [50, 80]
    })

    print("Calling process_financial_indicators...")
    # Now it's deterministic, NO API KEY needed
    result = process_financial_indicators(prices, funds, cutoff_date="2023-12-31")
    
    print("Result columns:", result.columns)
    print("Result head:")
    print(result.head())
    
    assert "score_fundamentalista" in result.columns
    # Check if scores are not 0.0 (unless they should be)
    # With these inputs, they should be normalized values.
    # pvp 0.95 (KNRI) is better than 1.05 (HGLG). 
    
    # HGLG PVP adj = 1 + (1-1.05) = 0.95
    # KNRI PVP adj = 1 + (1-0.95) = 1.05
    # KNRI should have better pvp_score.
    
    knri_score = result[result['ticker'] == 'KNRI11']['pvp_score'].values[0]
    hglg_score = result[result['ticker'] == 'HGLG11']['pvp_score'].values[0]
    
    print(f"KNRI PVP Score: {knri_score}, HGLG PVP Score: {hglg_score}")
    assert knri_score >= hglg_score
    
    print("Test passed!")

if __name__ == "__main__":
    test_integration()
