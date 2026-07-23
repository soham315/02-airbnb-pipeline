# Airbnb Pipeline with LLM Enrichment

An end-to-end learning project showing how AI fits naturally into a modern data pipeline.

```text
listings.csv ──┐
               ├──[Ingest]──[Clean]──[Enrich w/ LLM]──[Analytics]──▶ output/
reviews.csv  ──┘                           ↑
                                     OpenAI GPT-4o
                                  (sentiment, themes,
                                   recommendation)
```

## What this project does

This project ingests synthetic Airbnb listing and review data, cleans both sources, enriches guest comments with structured LLM outputs, and builds analytics-ready summary tables at both listing and neighbourhood level.

## What you'll learn

- How to organise an AI-enabled data pipeline into clear, reusable stages
- How to validate and clean tabular data before sending text to an LLM
- How to turn free-text reviews into structured features like sentiment and themes
- How to keep LLM usage swappable behind a simple wrapper
- How to build an analytics layer that is ready for BI tools or dashboards
- How to control cost with sampling and graceful fallbacks

## Why this matters for your resume

Most AI demos stop at a notebook cell or one-off script. Employers usually need engineers who can place AI inside dependable workflows: ingestion, cleaning, enrichment, storage, and reporting. This project gives you a portfolio piece that shows you understand the whole pipeline, not just the prompt.

## Prerequisites

- Python 3.10+
- Basic pandas familiarity
- An OpenAI API key for live enrichment (optional for local fallback mode)
- Comfort running terminal commands

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy the environment template:
   ```bash
   cp .env.example .env
   ```
4. Add your OpenAI key to `.env`:
   ```bash
   OPENAI_API_KEY=your_key_here
   OPENAI_MODEL=gpt-4o-mini
   ```
5. (Optional) Launch Jupyter if you want to follow the notebook walkthrough:
   ```bash
   jupyter notebook notebooks/walkthrough.ipynb
   ```

## How to run

Run the full pipeline on every cleaned review:

```bash
python -m src.pipeline
```

Run a cheaper sample first:

```bash
python -m src.pipeline --sample 20
```

Generated outputs:

- `output/enriched_reviews.parquet`
- `output/listing_summary.parquet`
- `output/neighbourhood_summary.parquet`

## Project structure

```text
02-airbnb-pipeline/
├── data/
│   ├── listings.csv              # Synthetic listing metadata for one city
│   └── reviews.csv               # Synthetic guest reviews tied to listings
├── output/
│   └── .gitkeep                  # Keeps the output folder in version control
├── src/
│   ├── __init__.py               # Package marker
│   ├── llm.py                    # OpenAI wrapper + structured JSON extraction
│   ├── ingest.py                 # CSV loading, schema checks, and merge logic
│   ├── clean.py                  # Cleaning logic before enrichment
│   ├── enrich.py                 # Review-by-review LLM enrichment with tqdm
│   ├── analytics.py              # Listing and neighbourhood analytics tables
│   └── pipeline.py               # CLI orchestration for the full workflow
├── notebooks/
│   └── walkthrough.ipynb         # 7-cell guided exploration notebook
├── .env.example                  # Environment variable template
├── requirements.txt              # Python dependencies
└── README.md                     # Project guide
```

## Sample output

### Enriched reviews

| review_id | listing_id | sentiment | themes | would_recommend | summary |
|---|---:|---|---|---|---|
| 1001 | 1 | positive | cleanliness, location, host | true | Clean flat near the Tube with a responsive host and an easy stay. |
| 1014 | 4 | negative | value, amenities | false | The apartment was stylish but felt expensive for the size and weak Wi-Fi. |
| 1042 | 13 | neutral | location | true | Great location for exploring London, though the room itself was fairly basic. |

### Listing summary

| listing_id | name | neighbourhood | avg_sentiment_score | top_themes | recommendation_rate | review_count |
|---:|---|---|---:|---|---:|---:|
| 1 | Camden Market Studio | Camden | 0.67 | location, host, cleanliness | 83.33 | 6 |
| 7 | Shoreditch Loft Retreat | Shoreditch | 0.50 | amenities, location | 75.00 | 4 |
| 19 | Greenwich Family Flat | Greenwich | 1.00 | cleanliness, value, host | 100.00 | 3 |

## Estimated cost

Using `gpt-4o-mini`, this project is intentionally inexpensive.

- `--sample 20`: usually well under $0.01 to a few cents depending on review length
- Full 100-review run: typically a few cents
- Costs rise if you switch to `gpt-4o` or add longer prompts/output schemas

Tip: start with `--sample 20`, verify the shape of your results, then scale up.

## Adapting to Databricks

If you want to move from pandas to Spark, keep the pipeline stages the same and swap the DataFrame implementation:

```python
from pyspark.sql import functions as F

listings_df = spark.read.csv("/dbfs/FileStore/airbnb/listings.csv", header=True, inferSchema=True)
reviews_df = spark.read.csv("/dbfs/FileStore/airbnb/reviews.csv", header=True, inferSchema=True)

clean_reviews_df = reviews_df.filter(F.length(F.trim(F.col("comments"))) > 0)
merged_df = listings_df.join(clean_reviews_df, on="listing_id", how="left")
```

A common Databricks pattern is to batch review text out to an enrichment job, write the structured results back to Delta, and then build Gold-layer aggregates from the enriched review table.

## Adapting the LLM

The pipeline is designed so `src/llm.py` is the only file that knows how to talk to the model provider.

You can adapt it by:

- swapping `OpenAI` for another vendor SDK
- keeping the same output schema (`sentiment`, `themes`, `would_recommend`, `summary`)
- preserving the `extract_review_insights()` function so the rest of the pipeline stays unchanged

That separation is the key lesson: keep LLM logic thin, and keep the rest of the pipeline boring and reliable.
