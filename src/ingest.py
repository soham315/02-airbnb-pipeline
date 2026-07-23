"""Input loading and schema validation for the Airbnb learning pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOGGER = logging.getLogger(__name__)

LISTING_COLUMNS: list[str] = [
    "listing_id",
    "name",
    "neighbourhood",
    "room_type",
    "price",
    "minimum_nights",
    "number_of_reviews",
    "availability_365",
]
REVIEW_COLUMNS: list[str] = ["review_id", "listing_id", "reviewer_name", "date", "comments"]


def _validate_columns(df: pd.DataFrame, required_columns: Iterable[str], dataset_name: str) -> None:
    """Raise a clear error if a required input schema is missing columns."""
    missing_columns: list[str] = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{dataset_name} is missing required columns: {missing_columns}")


def load_listings(path: str | Path) -> pd.DataFrame:
    """Load the listings CSV, validate its shape, and log basic metadata."""
    listings_path: Path = Path(path)
    listings_df: pd.DataFrame = pd.read_csv(listings_path)
    _validate_columns(listings_df, LISTING_COLUMNS, "Listings CSV")

    LOGGER.info("Loaded listings with shape %s from %s", listings_df.shape, listings_path)
    return listings_df


def load_reviews(path: str | Path) -> pd.DataFrame:
    """Load the reviews CSV, validate its shape, and log basic metadata."""
    reviews_path: Path = Path(path)
    reviews_df: pd.DataFrame = pd.read_csv(reviews_path)
    _validate_columns(reviews_df, REVIEW_COLUMNS, "Reviews CSV")

    LOGGER.info("Loaded reviews with shape %s from %s", reviews_df.shape, reviews_path)
    return reviews_df


def merge_data(listings: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Attach reviews to listings with a left join on listing_id.

    We perform this merge close to ingestion because it is the first moment where
    we can validate that both sources line up on the same business key.
    Catching join issues early is useful in real pipelines: if listing IDs are out
    of sync, there is no point spending money on enrichment before we fix the join.

    A left join preserves listing coverage. That matters because a downstream
    analytics layer usually wants to keep every listing visible, even if it has zero reviews.
    """
    merged_df: pd.DataFrame = listings.merge(
        reviews,
        how="left",
        on="listing_id",
        validate="one_to_many",
    )

    LOGGER.info("Merged dataset shape: %s", merged_df.shape)
    return merged_df
