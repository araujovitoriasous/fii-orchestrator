
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
        "Passivo_Total": [100, 200], # Old column, but good to keep
        "Contas_Receber_Aluguel": [50, 80],
        
        # New columns for Leverage
        "Valor_Ativo": [10000, 20000],
        "Patrimonio_Liquido": [9000, 15000] 
        # HGLG: Alavancagem = (10000 - 9000)/10000 = 0.10 (10%)
        # KNRI: Alavancagem = (20000 - 15000)/20000 = 0.25 (25%)
        # KNRI is MORE leveraged, so should have LOWER leverage score.
    })

    print("Calling process_financial_indicators...")
    result = process_financial_indicators(prices, funds, cutoff_date="2023-12-31")
    
    print("Result columns:", result.columns)
    print("Result head:")
    print(result.head())
    
    assert "alavancagem_score" in result.columns
    assert "alavancagem_real" in result.columns
    
    hglg_data = result[result['ticker'] == 'HGLG11'].iloc[0]
    knri_data = result[result['ticker'] == 'KNRI11'].iloc[0]
    
    print(f"HGLG Real Leverage: {hglg_data['alavancagem_real']}")
    print(f"KNRI Real Leverage: {knri_data['alavancagem_real']}")
    
    # Check absolute values approx
    assert abs(hglg_data['alavancagem_real'] - 0.10) < 0.001
    assert abs(knri_data['alavancagem_real'] - 0.25) < 0.001
    
    print(f"HGLG Leverage Score: {hglg_data['alavancagem_score']}")
    print(f"KNRI Leverage Score: {knri_data['alavancagem_score']}")
    
    # HGLG (10%) is better/safer than KNRI (25%), so score should be higher
    assert hglg_data['alavancagem_score'] >= knri_data['alavancagem_score']
    
    print("Test passed!")

if __name__ == "__main__":
    test_integration()
