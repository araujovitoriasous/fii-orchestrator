from pathlib import Path
import json
from datetime import datetime, timezone
import polars as pl

def write_parquet_partition(df: pl.DataFrame, base: Path, partitions: dict[str, str], fname: str):
    path = base
    for key, val in partitions.items():
        path = path / f"{key}={val}"
    path.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path / fname)

def write_metadata(base: Path, kind: str, payload: dict):
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with open(base / f"{kind}_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
