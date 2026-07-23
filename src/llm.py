"""Helpers for talking to an LLM in a predictable, low-friction way."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

# Loading environment variables at module import keeps the learning project simple.
# In a larger application you might centralise configuration, but here we want a
# beginner to be able to run one command and have the client pick up their .env file.
load_dotenv()

# The OpenAI SDK uses httpx under the hood, and its INFO logs are distracting in a
# teaching project. We silence them so the pipeline output stays focused on the data flow.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

SUPPORTED_THEMES: list[str] = ["cleanliness", "location", "host", "value", "amenities"]


def get_openai_client() -> OpenAI | None:
    """Return an OpenAI client when credentials exist, otherwise None.

    Returning None lets the rest of the pipeline switch to a tiny rule-based
    fallback during local demos. That means learners can still run the full
    pipeline shape without burning tokens or blocking on API setup.
    """
    api_key: str | None = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    return OpenAI(api_key=api_key)


def _normalise_themes(raw_themes: Any) -> list[str]:
    """Keep only supported themes and remove duplicates while preserving order."""
    if not isinstance(raw_themes, list):
        return []

    cleaned_themes: list[str] = []
    for theme in raw_themes:
        normalised_theme: str = str(theme).strip().lower()
        if normalised_theme in SUPPORTED_THEMES and normalised_theme not in cleaned_themes:
            cleaned_themes.append(normalised_theme)

    return cleaned_themes


def _coerce_bool(raw_value: Any) -> bool:
    """Convert slightly messy model output into a real Python bool."""
    if isinstance(raw_value, bool):
        return raw_value

    if isinstance(raw_value, str):
        return raw_value.strip().lower() in {"true", "yes", "y", "1"}

    return bool(raw_value)


def _fallback_review_insights(review_text: str) -> dict[str, Any]:
    """Produce lightweight local insights when no API key is configured.

    This is intentionally simple and deterministic. It is *not* meant to replace
    the LLM path in production, but it keeps the teaching project fully runnable
    in offline demos and CI environments.
    """
    text: str = review_text.lower()

    positive_keywords: list[str] = [
        "great",
        "clean",
        "excellent",
        "friendly",
        "comfortable",
        "perfect",
        "amazing",
        "lovely",
        "recommend",
        "helpful",
        "easy",
        "quiet",
    ]
    negative_keywords: list[str] = [
        "dirty",
        "noisy",
        "late",
        "small",
        "broken",
        "cold",
        "expensive",
        "uncomfortable",
        "disappointing",
        "bad",
        "issue",
        "problem",
    ]

    theme_keywords: dict[str, list[str]] = {
        "cleanliness": ["clean", "dirty", "tidy", "dusty", "bathroom"],
        "location": ["location", "station", "walk", "tube", "neighbourhood", "area"],
        "host": ["host", "check-in", "communication", "friendly", "helpful"],
        "value": ["value", "price", "expensive", "worth", "affordable"],
        "amenities": ["wifi", "kitchen", "bed", "shower", "heating", "amenities"],
    }

    positive_hits: int = sum(keyword in text for keyword in positive_keywords)
    negative_hits: int = sum(keyword in text for keyword in negative_keywords)

    if positive_hits > negative_hits:
        sentiment: str = "positive"
    elif negative_hits > positive_hits:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    themes: list[str] = [
        theme for theme, keywords in theme_keywords.items() if any(keyword in text for keyword in keywords)
    ]

    summary_source: str = review_text.strip().replace("\n", " ")
    summary: str = summary_source.split(".")[0].strip() or "The guest shared a mixed stay experience."
    if not summary.endswith("."):
        summary = f"{summary}."

    would_recommend: bool = sentiment == "positive" or (sentiment == "neutral" and "recommend" in text)

    return {
        "sentiment": sentiment,
        "themes": themes[:3],
        "would_recommend": would_recommend,
        "summary": summary[:180],
    }


def extract_review_insights(review_text: str, client: OpenAI | None) -> dict[str, Any]:
    """Extract structured review insights with JSON mode.

    Why JSON mode matters:
    - Free-form text is frustrating for pipelines because every downstream step has
      to guess how to parse it.
    - Asking the model for JSON gives us stable keys that slot neatly into a
      DataFrame and later analytics tables.

    Why cost matters:
    - This function is called once per review in the learning project.
    - Small prompt changes get multiplied by the number of rows, so we keep the
      instructions short and the output schema tiny.
    - The default model can be swapped with OPENAI_MODEL if you want a cheaper or
      stronger model.
    """
    cleaned_review_text: str = review_text.strip()
    if not cleaned_review_text:
        return {
            "sentiment": "neutral",
            "themes": [],
            "would_recommend": False,
            "summary": "No review text was provided.",
        }

    if client is None:
        return _fallback_review_insights(cleaned_review_text)

    model_name: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    response = client.chat.completions.create(
        model=model_name,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract structured review insights for an Airbnb analytics pipeline. "
                    "Return valid JSON with keys sentiment, themes, would_recommend, and summary. "
                    "sentiment must be one of positive, negative, neutral. "
                    "themes must be a list containing only cleanliness, location, host, value, amenities. "
                    "summary must be one short sentence."
                ),
            },
            {
                "role": "user",
                "content": f"Review: {cleaned_review_text}",
            },
        ],
    )

    raw_message: str = response.choices[0].message.content or "{}"
    parsed_response: dict[str, Any] = json.loads(raw_message)

    sentiment: str = str(parsed_response.get("sentiment", "neutral")).strip().lower()
    if sentiment not in {"positive", "negative", "neutral"}:
        sentiment = "neutral"

    summary: str = str(parsed_response.get("summary", "No summary returned.")).strip()
    if not summary:
        summary = "No summary returned."

    return {
        "sentiment": sentiment,
        "themes": _normalise_themes(parsed_response.get("themes", [])),
        "would_recommend": _coerce_bool(parsed_response.get("would_recommend", False)),
        "summary": summary,
    }
