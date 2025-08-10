from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
BRONZE = DATA_DIR / "bronze"
META = DATA_DIR / "meta"

# fontes RSS (pode usar .env para customizar)
NEWS_RSS_SOURCES = [
    s.strip() for s in os.getenv("NEWS_RSS_SOURCES", "").split(",") if s.strip()
] or [
    "https://www.infomoney.com.br/feed/",
    "https://valor.globo.com/rss/",
    "https://www.suno.com.br/feed/",
]
