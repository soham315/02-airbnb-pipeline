from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "output"

SENTIMENT_ORDER = ["positive", "neutral", "negative"]
SENTIMENT_SCORE_MAP = {"positive": 1, "neutral": 0, "negative": -1}


st.set_page_config(page_title="Airbnb Pipeline Dashboard", page_icon="🏠", layout="wide")


@st.cache_data(show_spinner=False)
def load_outputs(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the generated Parquet outputs with Streamlit caching."""
    enriched_reviews = pd.read_parquet(output_dir / "enriched_reviews.parquet")
    listing_summary = pd.read_parquet(output_dir / "listing_summary.parquet")
    neighbourhood_summary = pd.read_parquet(output_dir / "neighbourhood_summary.parquet")
    return enriched_reviews, listing_summary, neighbourhood_summary


@st.cache_data(show_spinner=False)
def build_dashboard_frame(enriched_reviews: pd.DataFrame, listing_summary: pd.DataFrame) -> pd.DataFrame:
    """Join review and listing context so reviews can be filtered by neighbourhood."""
    review_context = enriched_reviews.merge(
        listing_summary[["listing_id", "name", "neighbourhood", "room_type", "price"]],
        on="listing_id",
        how="left",
    )
    return review_context


def apply_text_filter(dataframe: pd.DataFrame, search_text: str, searchable_columns: list[str]) -> pd.DataFrame:
    """Filter rows with a simple case-insensitive text search across selected columns."""
    if not search_text:
        return dataframe

    query = search_text.strip().lower()
    if not query:
        return dataframe

    mask = pd.Series(False, index=dataframe.index)
    for column in searchable_columns:
        if column in dataframe.columns:
            column_values = dataframe[column].fillna("").astype(str).str.lower()
            mask = mask | column_values.str.contains(query, regex=False)

    return dataframe.loc[mask]


def stringify_themes(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert a list/array-like 'themes' column into a display-friendly string.

    Parquet round-trips list columns as numpy arrays, which are unhashable and
    crash Streamlit's dataframe widget (it hashes column values internally for
    sorting/filtering). This normalizes the column to a plain comma-separated
    string before display. Safe to call even if the column is already a string
    or missing entirely.
    """
    if "themes" not in dataframe.columns:
        return dataframe

    display_df = dataframe.copy()

    def _to_display_string(value: object) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return ", ".join(str(item) for item in value)
        if hasattr(value, "tolist"):
            return ", ".join(str(item) for item in value.tolist())
        return ""

    display_df["themes"] = display_df["themes"].apply(_to_display_string)
    return display_df


def render_metric_card(title: str, value: str | float, delta: str | None = None) -> None:
    """Display a polished KPI card."""
    st.markdown(
        f"""
        <div style="padding: 1.1rem 1.2rem; border-radius: 14px; background: linear-gradient(135deg, #0f172a, #1d4ed8); color: white; box-shadow: 0 10px 30px rgba(15,23,42,0.15); margin-bottom: 0.6rem;">
            <div style="font-size: 0.85rem; opacity: 0.88;">{title}</div>
            <div style="font-size: 1.8rem; font-weight: 700; margin-top: 0.25rem;">{value}</div>
            {f'<div style="font-size: 0.8rem; margin-top: 0.35rem; opacity: 0.9;">{delta}</div>' if delta else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_pipeline_command() -> None:
    """Run the pipeline process and refresh cached data after completion."""
    with st.spinner("Running the pipeline ..."):
        try:
            subprocess.run(
                [sys.executable, "-m", "src.pipeline", "--sample", "20"],
                cwd=ROOT_DIR,
                check=True,
                text=True,
                capture_output=True,
            )
            st.cache_data.clear()
            st.success("Pipeline completed successfully. The dashboard data has been refreshed.")
        except subprocess.CalledProcessError as exc:
            error_output = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            st.error(f"Pipeline failed: {error_output}")
        except Exception as exc:
            st.error(f"Unexpected error while running the pipeline: {exc}")


def build_sentiment_chart(reviews_df: pd.DataFrame) -> px.bar:
    sentiment_breakdown = (
        reviews_df["sentiment"].dropna().astype(str).str.lower().value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    )
    return px.bar(
        x=sentiment_breakdown.index,
        y=sentiment_breakdown.values,
        color=sentiment_breakdown.index,
        title="Sentiment Distribution",
        labels={"x": "Sentiment", "y": "Review Count"},
        color_discrete_map={"positive": "#22c55e", "neutral": "#f59e0b", "negative": "#ef4444"},
    )

def build_theme_chart(reviews_df: pd.DataFrame) -> px.bar:
    def _to_list(value: object) -> list:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if hasattr(value, "tolist"):
            return value.tolist()
        return []

    theme_rows = reviews_df["themes"].dropna().apply(_to_list)
    theme_frame = theme_rows.explode().dropna()
    theme_counts = theme_frame.value_counts().head(10).reset_index()
    theme_counts.columns = ["theme", "count"]
    return px.bar(
        theme_counts,
        x="count",
        y="theme",
        orientation="h",
        color="theme",
        title="Top Review Themes",
        labels={"count": "Review Count", "theme": "Theme"},
    )


def main() -> None:
    with st.sidebar:
        st.title("Filters")
        st.caption("Slice the dashboard by neighbourhood and review sentiment.")

        neighbourhood_options = []
        if OUTPUT_DIR.exists():
            try:
                _, listing_summary, _ = load_outputs(OUTPUT_DIR)
                neighbourhood_options = sorted(listing_summary["neighbourhood"].dropna().unique().tolist())
            except Exception:
                neighbourhood_options = []

        selected_neighbourhoods = st.multiselect(
            "Neighbourhood",
            options=neighbourhood_options,
            default=neighbourhood_options,
        )

        sentiment_options = ["positive", "neutral", "negative"]
        selected_sentiments = st.multiselect(
            "Sentiment",
            options=sentiment_options,
            default=sentiment_options,
        )

        st.markdown("---")
        if st.button("Run Pipeline", use_container_width=True):
            run_pipeline_command()

    if not OUTPUT_DIR.exists() or not all((OUTPUT_DIR / name).exists() for name in ["enriched_reviews.parquet", "listing_summary.parquet", "neighbourhood_summary.parquet"]):
        st.warning("The parquet outputs are not available yet. Use the sidebar action to run the pipeline and generate them.")
        return

    enriched_reviews, listing_summary, neighbourhood_summary = load_outputs(OUTPUT_DIR)
    reviews_with_context = build_dashboard_frame(enriched_reviews, listing_summary)

    if selected_neighbourhoods:
        reviews_with_context = reviews_with_context[reviews_with_context["neighbourhood"].isin(selected_neighbourhoods)]
        listing_summary = listing_summary[listing_summary["neighbourhood"].isin(selected_neighbourhoods)]
        neighbourhood_summary = neighbourhood_summary[neighbourhood_summary["neighbourhood"].isin(selected_neighbourhoods)]

    if selected_sentiments:
        selected_sentiments_normalized = {value.lower() for value in selected_sentiments}
        reviews_with_context = reviews_with_context[
            reviews_with_context["sentiment"].astype(str).str.lower().isin(selected_sentiments_normalized)
        ]

    sentiment_scores = reviews_with_context["sentiment"].map(SENTIMENT_SCORE_MAP).fillna(0)
    recommendation_rate = (reviews_with_context["would_recommend"].fillna(False).mean() * 100) if not reviews_with_context.empty else 0.0
    average_sentiment = sentiment_scores.mean() if not sentiment_scores.empty else 0.0

    st.title("🏠 Airbnb Pipeline Dashboard")
    st.caption("Interactive, cache-backed analytics for the generated listing and review outputs.")

    metric_columns = st.columns(4)
    with metric_columns[0]:
        render_metric_card("Total Listings", f"{int(listing_summary['listing_id'].nunique())}")
    with metric_columns[1]:
        render_metric_card("Total Reviews", f"{int(len(reviews_with_context))}")
    with metric_columns[2]:
        render_metric_card("Average Sentiment", f"{average_sentiment:.2f}")
    with metric_columns[3]:
        render_metric_card("Recommendation Rate", f"{recommendation_rate:.2f}%")

    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.plotly_chart(build_sentiment_chart(reviews_with_context), width="stretch")
    with chart_columns[1]:
        st.plotly_chart(build_theme_chart(reviews_with_context), width="stretch")

    st.plotly_chart(
        px.bar(
            neighbourhood_summary,
            x="neighbourhood",
            y="average_recommendation_rate",
            color="average_recommendation_rate",
            title="Recommendation Rate by Neighbourhood",
            labels={"average_recommendation_rate": "Recommendation Rate (%)", "neighbourhood": "Neighbourhood"},
            color_continuous_scale="Viridis",
        ),
        width="stretch",
    )

    st.markdown("### Listings")
    listings_search = st.text_input("Search listings", placeholder="Type a listing name, neighbourhood, or room type")
    filtered_listings = apply_text_filter(
        listing_summary,
        listings_search,
        ["name", "neighbourhood", "room_type", "top_themes"],
    )
    st.dataframe(
        stringify_themes(filtered_listings).sort_values("listing_id").reset_index(drop=True),
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Reviews")
    reviews_search = st.text_input("Search reviews", placeholder="Search comments, themes, or reviewer names")
    filtered_reviews = apply_text_filter(
        reviews_with_context,
        reviews_search,
        ["reviewer_name", "comments", "summary", "themes", "neighbourhood", "name"],
    )
    st.dataframe(
        stringify_themes(filtered_reviews).sort_values("review_id").reset_index(drop=True),
        width="stretch",
        hide_index=True,
    )


if __name__ == "__main__":
    main()
