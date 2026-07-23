"""Command-line entry point for the Airbnb pipeline learning project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Force UTF-8 output on Windows where supported
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

from .analytics import (
    build_listing_summary,
    build_neighbourhood_summary,
    save_analytics_outputs,
)
from .clean import clean_listings, clean_reviews
from .enrich import enrich_reviews
from .ingest import load_listings, load_reviews, merge_data
from .llm import get_openai_client

# Loading .env at startup keeps the CLI beginner-friendly.
load_dotenv()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the Airbnb + LLM enrichment pipeline."
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional number of reviews to enrich.",
    )

    parser.add_argument(
        "--listings-path",
        type=Path,
        default=Path("data/listings.csv"),
        help="Path to the listings CSV.",
    )

    parser.add_argument(
        "--reviews-path",
        type=Path,
        default=Path("data/reviews.csv"),
        help="Path to the reviews CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory where Parquet outputs will be saved.",
    )

    return parser.parse_args()


def run_pipeline(
    listings_path: Path,
    reviews_path: Path,
    output_dir: Path,
    sample: int | None = None,
) -> None:
    """Execute the full ingest -> clean -> enrich -> analytics workflow."""

    print("=" * 60)
    print("Starting Airbnb Pipeline with LLM Enrichment")
    print("=" * 60)

    print("\nLoading source data...")
    listings_raw = load_listings(listings_path)
    reviews_raw = load_reviews(reviews_path)

    print(f"Listings loaded: {len(listings_raw)}")
    print(f"Reviews loaded: {len(reviews_raw)}")

    print("\nCleaning data...")
    listings_clean = clean_listings(listings_raw)
    reviews_clean = clean_reviews(reviews_raw)

    print(f"Clean listings: {len(listings_clean)}")
    print(f"Clean reviews: {len(reviews_clean)}")

    merged_preview = merge_data(listings_clean, reviews_clean)
    print(f"\nMerged dataset shape: {merged_preview.shape}")

    print("\nPreparing LLM client...")

    client = get_openai_client()

    if client is None:
        print("No OpenAI API key found.")
        print("Using local fallback mode.")
    else:
        print("OpenAI client initialized successfully.")

    reviews_to_process = sample if sample is not None else len(reviews_clean)

    print(f"\nEnriching {reviews_to_process} review(s)...")

    enriched_reviews = enrich_reviews(
        reviews_clean,
        client=client,
        sample=sample,
    )

    print(f"Enriched reviews: {len(enriched_reviews)}")
    print(f"LLM errors: {enriched_reviews['llm_error'].notna().sum()}")

    print("\nBuilding analytical summaries...")

    listing_summary = build_listing_summary(
        listings_clean,
        enriched_reviews,
    )

    neighbourhood_summary = build_neighbourhood_summary(
        listing_summary,
    )

    print(f"Listing summary rows: {len(listing_summary)}")
    print(f"Neighbourhood summary rows: {len(neighbourhood_summary)}")

    print("\nSaving outputs...")

    output_dir.mkdir(parents=True, exist_ok=True)

    enriched_reviews_path = output_dir / "enriched_reviews.parquet"

    enriched_reviews.to_parquet(
        enriched_reviews_path,
        index=False,
    )

    listing_path, neighbourhood_path = save_analytics_outputs(
        listing_summary_df=listing_summary,
        neighbourhood_summary_df=neighbourhood_summary,
        output_dir=output_dir,
    )

    print("\nPipeline complete!")

    print(f"Enriched reviews saved to: {enriched_reviews_path}")
    print(f"Listing summary saved to: {listing_path}")
    print(f"Neighbourhood summary saved to: {neighbourhood_path}")

    print("\nTop 5 neighbourhoods by average sentiment:\n")
    print(neighbourhood_summary.head(5).to_string(index=False))


def main() -> None:
    """CLI wrapper."""

    args = parse_args()

    if args.sample is not None and args.sample <= 0:
        raise ValueError("--sample must be a positive integer.")

    run_pipeline(
        listings_path=args.listings_path,
        reviews_path=args.reviews_path,
        output_dir=args.output_dir,
        sample=args.sample,
    )


if __name__ == "__main__":
    main()