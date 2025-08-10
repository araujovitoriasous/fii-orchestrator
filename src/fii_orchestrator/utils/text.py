import re
from unidecode import unidecode

# detecta tickers estilo ABCD11 em títulos/descrições
TICKER_RE = re.compile(r"\b([A-Z]{4}\d{2})\b")

def detect_tickers(text: str) -> list[str]:
    if not text:
        return []
    txt = unidecode(text.upper())
    return sorted(set(TICKER_RE.findall(txt)))
