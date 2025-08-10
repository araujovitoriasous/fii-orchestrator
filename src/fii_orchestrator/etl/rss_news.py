import feedparser
from datetime import datetime, timezone
from loguru import logger
import polars as pl

from fii_orchestrator.config import BRONZE, META, NEWS_RSS_SOURCES
from fii_orchestrator.etl.schemas import NewsItem
from fii_orchestrator.utils.io import write_parquet_partition, write_metadata
from fii_orchestrator.utils.text import detect_tickers

def fetch_feed(url: str) -> list[NewsItem]:
    feed = feedparser.parse(url)
    items: list[NewsItem] = []
    for e in feed.entries:
        # published_parsed pode faltar; caia para now()
        if getattr(e, "published_parsed", None):
            dt = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
        else:
            dt = datetime.now(timezone.utc)
        title = getattr(e, "title", "") or ""
        summary = getattr(e, "summary", "") or ""
        tickers = sorted(set(detect_tickers(title) + detect_tickers(summary)))
        items.append(NewsItem(
            id=getattr(e, "id", getattr(e, "link", title)),
            title=title,
            link=getattr(e, "link", ""),
            published_at=dt,
            source=feed.feed.get("title", url),
            summary=summary,
            raw_tickers=tickers,
        ))
    return items

def run():
    logger.info(f"RSS sources: {len(NEWS_RSS_SOURCES)}")
    all_rows = []
    sources_meta = []

    for url in NEWS_RSS_SOURCES:
        try:
            rows = fetch_feed(url)
            logger.info(f"{url} -> {len(rows)} items")
            all_rows.extend(rows)
            sources_meta.append({"url": url, "items": len(rows)})
        except Exception as e:
            logger.exception(f"Erro ao coletar {url}: {e}")

    if not all_rows:
        logger.warning("Nenhuma notícia coletada.")
        return

    df = pl.DataFrame([r.model_dump() for r in all_rows]).with_columns(
        pl.col("published_at").dt.convert_time_zone("America/Sao_Paulo")
    ).with_columns(
        year=pl.col("published_at").dt.year().cast(pl.Utf8),
        month=pl.col("published_at").dt.month().cast(pl.Utf8).str.zfill(2),
        day=pl.col("published_at").dt.day().cast(pl.Utf8).str.zfill(2),
    )

    # particiona por ano/mês; salvar 1 arquivo por run
    write_parquet_partition(
        df=df.drop(["year", "month", "day"]),
        base=BRONZE / "news",
        partitions={"year": df["year"][0], "month": df["month"][0]},
        fname=f"news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
    )

    write_metadata(META / "news", "news_run", {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources_meta,
        "rows": len(all_rows),
    })
    logger.info("RSS salvo em bronze/news com metadados em meta/news")

if __name__ == "__main__":
    run()
