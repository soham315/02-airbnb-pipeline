"""Cleaning logic for listings and reviews before enrichment."""

from __future__ import annotations

import pandas as pd


def _strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim surrounding whitespace on every object column in a DataFrame."""
    cleaned_df: pd.DataFrame = df.copy()
    for column_name in cleaned_df.select_dtypes(include="object").columns:
        # Normalising whitespace early prevents subtle join mismatches, duplicate
        # categories, and messy prompts later when we send review text to the LLM.
        cleaned_df[column_name] = cleaned_df[column_name].astype("string").str.strip()
    return cleaned_df


def clean_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise listing fields so analytics run on stable numeric types.

    Every step here protects a later stage of the pipeline:
    - stripping whitespace stops category explosions like `Camden` vs ` Camden`
    - coercing numeric fields avoids silent string math bugs
    - filling missing availability prevents null-heavy aggregates
    - dropping duplicates preserves one row per business entity
    """
    cleaned_df: pd.DataFrame = _strip_string_columns(df)

    # Price often arrives as a string in real marketplaces, so we coerce instead
    # of assuming the source system always behaves perfectly.
    cleaned_df["price"] = pd.to_numeric(cleaned_df["price"], errors="coerce")
    cleaned_df["minimum_nights"] = pd.to_numeric(cleaned_df["minimum_nights"], errors="coerce")
    cleaned_df["number_of_reviews"] = pd.to_numeric(cleaned_df["number_of_reviews"], errors="coerce")
    cleaned_df["availability_365"] = pd.to_numeric(cleaned_df["availability_365"], errors="coerce")

    # Availability becomes 0 when the source is missing because that is usually a
    # more useful default for dashboards than leaving a null hole.
    cleaned_df["availability_365"] = cleaned_df["availability_365"].fillna(0)

    # Keeping a single row per listing avoids duplicate joins and double counting.
    cleaned_df = cleaned_df.drop_duplicates(subset=["listing_id"]).reset_index(drop=True)
    return cleaned_df


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare review text so the LLM receives clean, high-signal input.

    Cleaning matters a lot more before LLM calls than many engineers first expect:
    - extra whitespace wastes tokens and makes review text look lower quality
    - parsing dates upfront keeps time-based slicing reliable later
    - dropping empty comments avoids paying for calls that can never add insight
    """
    cleaned_df: pd.DataFrame = _strip_string_columns(df)

    # Parsing once here means every later notebook, filter, and chart can rely on
    # a real datetime type instead of re-implementing date logic ad hoc.
    cleaned_df["date"] = pd.to_datetime(cleaned_df["date"], errors="coerce")

    # Empty comments should be removed before enrichment because an LLM cannot
    # infer sentiment or themes from blank text, and each bad row still costs money.
    cleaned_df["comments"] = cleaned_df["comments"].fillna("")
    cleaned_df = cleaned_df.loc[cleaned_df["comments"].str.len() > 0].copy()

    cleaned_df = cleaned_df.drop_duplicates(subset=["review_id"]).reset_index(drop=True)
    return cleaned_df
