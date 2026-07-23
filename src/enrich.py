"""LLM-powered enrichment over Airbnb review text."""

from __future__ import annotations

from typing import Any

import pandas as pd
from tqdm import tqdm

from .llm import extract_review_insights


def enrich_reviews(reviews_df: pd.DataFrame, client: Any, sample: int | None = None) -> pd.DataFrame:
    """Enrich review comments with sentiment, themes, recommendation, and summary.

    We process reviews row by row in this learning project because it keeps the
    control flow easy to understand. In a larger production system you could move
    to async workers or the provider's batch API, but the pipeline shape would stay
    the same: clean text in, structured attributes out.

    The optional `sample` argument is intentionally practical. It lets learners test
    the whole pipeline on 10 or 20 reviews first, which is friendlier for both cost
    control and debugging.
    """
    if "comments" not in reviews_df.columns:
        raise ValueError("reviews_df must include a comments column for enrichment.")

    working_df: pd.DataFrame = reviews_df.copy()
    if sample is not None:
        working_df = working_df.head(sample).copy()

    if working_df.empty:
        empty_df: pd.DataFrame = working_df.copy()
        empty_df["sentiment"] = pd.Series(dtype="string")
        empty_df["themes"] = pd.Series(dtype="object")
        empty_df["would_recommend"] = pd.Series(dtype="bool")
        empty_df["summary"] = pd.Series(dtype="string")
        empty_df["llm_error"] = pd.Series(dtype="string")
        return empty_df

    enriched_rows: list[dict[str, Any]] = []

    for row in tqdm(
        list(working_df.itertuples(index=False, name="ReviewRow")),
        total=len(working_df),
        desc="🤖 Enriching reviews",
    ):
        row_dict: dict[str, Any] = row._asdict()
        llm_error: str | None = None

        try:
            insights: dict[str, Any] = extract_review_insights(str(row.comments), client)
        except Exception as exc:  # noqa: BLE001 - we want the pipeline to continue.
            # LLM failures should degrade gracefully instead of crashing the entire
            # pipeline. That mirrors a common production approach: log the issue,
            # write a safe default, and let downstream analysts inspect failures.
            insights = {
                "sentiment": "neutral",
                "themes": [],
                "would_recommend": False,
                "summary": "LLM extraction failed for this review.",
            }
            llm_error = str(exc)

        row_dict.update(insights)
        row_dict["llm_error"] = llm_error
        enriched_rows.append(row_dict)

    enriched_df: pd.DataFrame = pd.DataFrame(enriched_rows)
    return enriched_df
