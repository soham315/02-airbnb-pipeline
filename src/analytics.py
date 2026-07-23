"""Analytical summary tables built from enriched Airbnb review data."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

SENTIMENT_SCORE_MAP: dict[str, int] = {"positive": 1, "neutral": 0, "negative": -1}


def _top_themes(themes: Iterable[str]) -> str:
    """Return the most common themes as a comma-separated string."""
    filtered_themes: list[str] = [theme for theme in themes if theme]
    if not filtered_themes:
        return ""

    theme_counter: Counter[str] = Counter(filtered_themes)
    return ", ".join(theme for theme, _ in theme_counter.most_common(3))


def build_listing_summary(listings_df: pd.DataFrame, enriched_reviews_df: pd.DataFrame) -> pd.DataFrame:
    """Build a listing-level analytical layer for BI and stakeholder reporting.

    This table is useful because it moves us from raw review text to metrics that a
    dashboard, analyst, or product manager can consume quickly. Instead of reading
    hundreds of comments, they get a stable row per listing with recommendation rate,
    sentiment score, review volume, and the themes guests talk about most.
    """
    reviews_df: pd.DataFrame = enriched_reviews_df.copy()

    if reviews_df.empty:
        listing_summary_df: pd.DataFrame = listings_df.copy()
        listing_summary_df["avg_sentiment_score"] = 0.0
        listing_summary_df["top_themes"] = ""
        listing_summary_df["recommendation_rate"] = 0.0
        listing_summary_df["review_count"] = 0
        return listing_summary_df

    reviews_df["sentiment_score"] = reviews_df["sentiment"].map(SENTIMENT_SCORE_MAP).fillna(0)
    reviews_df["themes"] = reviews_df["themes"].apply(lambda value: value if isinstance(value, list) else [])
    reviews_df["would_recommend"] = reviews_df["would_recommend"].fillna(False).astype(bool)

    review_metrics_df: pd.DataFrame = (
        reviews_df.groupby("listing_id", as_index=False)
        .agg(
            avg_sentiment_score=("sentiment_score", "mean"),
            recommendation_rate=("would_recommend", lambda values: round(values.mean() * 100, 2)),
            review_count=("review_id", "count"),
        )
        .round({"avg_sentiment_score": 2})
    )

    theme_metrics_df: pd.DataFrame = (
        reviews_df.explode("themes")
        .dropna(subset=["themes"])
        .groupby("listing_id", as_index=False)
        .agg(top_themes=("themes", _top_themes))
    )

    listing_summary_df = listings_df.merge(review_metrics_df, how="left", on="listing_id")
    listing_summary_df = listing_summary_df.merge(theme_metrics_df, how="left", on="listing_id")

    listing_summary_df["avg_sentiment_score"] = listing_summary_df["avg_sentiment_score"].fillna(0.0)
    listing_summary_df["recommendation_rate"] = listing_summary_df["recommendation_rate"].fillna(0.0)
    listing_summary_df["review_count"] = listing_summary_df["review_count"].fillna(0).astype(int)
    listing_summary_df["top_themes"] = listing_summary_df["top_themes"].fillna("")

    return listing_summary_df.sort_values(["neighbourhood", "listing_id"]).reset_index(drop=True)


def build_neighbourhood_summary(listing_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate listing metrics to neighbourhood level.

    A neighbourhood summary is where the business value becomes obvious: market
    teams can spot strong areas, operations teams can identify weaker clusters,
    and leadership gets a compact table instead of listing-by-listing detail.
    """
    neighbourhood_summary_df: pd.DataFrame = (
        listing_summary_df.groupby("neighbourhood", as_index=False)
        .agg(
            listing_count=("listing_id", "nunique"),
            average_price=("price", "mean"),
            average_sentiment_score=("avg_sentiment_score", "mean"),
            average_recommendation_rate=("recommendation_rate", "mean"),
            total_reviews=("review_count", "sum"),
        )
        .round(
            {
                "average_price": 2,
                "average_sentiment_score": 2,
                "average_recommendation_rate": 2,
            }
        )
        .sort_values("average_sentiment_score", ascending=False)
        .reset_index(drop=True)
    )

    return neighbourhood_summary_df


def save_analytics_outputs(
    listing_summary_df: pd.DataFrame,
    neighbourhood_summary_df: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Persist analytical outputs as Parquet files for downstream tools."""
    output_path: Path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    listing_path: Path = output_path / "listing_summary.parquet"
    neighbourhood_path: Path = output_path / "neighbourhood_summary.parquet"

    listing_summary_df.to_parquet(listing_path, index=False)
    neighbourhood_summary_df.to_parquet(neighbourhood_path, index=False)
    return listing_path, neighbourhood_path
