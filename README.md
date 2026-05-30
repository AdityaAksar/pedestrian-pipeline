# Melbourne Pedestrian Analytics Pipeline

An end-to-end data pipeline that ingests, transforms, and forecasts pedestrian traffic across 100+ sensors in Melbourne, Australia 
fully automated via GitHub Actions and powered by BigQuery, dbt, and XGBoost.

---

## Architecture

```
Melbourne Open Data API
        |
GitHub Actions (daily, 02:00 AEST)
        |
Ingestion (Python) ─────────────> BigQuery (raw)
                                        |
                                   dbt (pedestrian-dbt)
                                        |
                                   BigQuery (staging/mart)
                                        |
Forecasting (XGBoost + Open-Meteo) ────┘
                                        |
                                   BigQuery (mart.predictions)
                                        |
                                   Power BI Dashboard
```

---

## Repositories

| Repository | Description |
|---|---|
| **pedestrian-pipeline** (this repo) | Ingestion and forecasting pipeline + GitHub Actions workflows |
| **pedestrian-dbt** | dbt transformations (staging to mart) |
| **pedestrian-forecast** | Exploratory notebooks, EDA, and model comparison |

---

## Pipeline Components

### 1. Daily Ingestion
- Fetches new data from the [Melbourne Open Data API](https://data.melbourne.vic.gov.au/)
- Loads raw data into BigQuery table `raw.pedestrian_counts`
- Triggered daily at 02:00 AEST via GitHub Actions

### 2. dbt Transformation
- Transforms data from raw to staging to mart layers
- Enriches data with weather features (temperature, precipitation, windspeed, humidity)
- Produces `mart_pedestrian_hourly` as the base table for forecasting

### 3. Daily Forecasting
- Fetches 7-day hourly weather forecast from [Open-Meteo API](https://open-meteo.com/) (free, no API key required)
- Loops through 101 sensors automatically
- Trains an XGBoost model per sensor using features: hour, day of week, weather, lag values, and rolling means
- Generates 168-hour (7-day) predictions per sensor
- Saves results to BigQuery `mart.predictions`
- Triggered daily after ingestion completes via GitHub Actions

---

## Model Performance

XGBoost was selected after comparison with Prophet:

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Prophet | 445.8 | 593.8 | 186.3% |
| **XGBoost** | **92.3** | **145.9** | **26.5%** |

XGBoost performs **4.8x better** than Prophet on Melbourne pedestrian data.

Average performance across 101 sensors:

| Metric | Value |
|---|---|
| MAE | 48.92 |
| RMSE | 86.74 |
| MAPE | 40.61% |

---

## Repository Structure

```
pedestrian-pipeline/
├── ingestion/
│   ├── run_ingestion.py
│   └── requirements.txt
├── forecasting/
│   ├── run_forecast.py
│   └── requirements_forecast.txt
└── .github/workflows/
    ├── daily_ingestion.yml
    └── daily_forecast.yml
```

---

## Setup

### Prerequisites
- Python 3.11+
- Google Cloud project with BigQuery enabled
- Service account with BigQuery Editor role

### GitHub Secrets

Add the following secret in repository settings:

| Secret | Description |
|---|---|
| `GCP_SA_KEY` | Service account key in raw JSON format |

### Manual Run

```bash
# Install dependencies
pip install -r forecasting/requirements_forecast.txt

# Set environment variable
export GCP_SA_KEY='<your-service-account-json>'

# Run forecasting pipeline
python forecasting/run_forecast.py
```

---

## Dashboard

The Power BI dashboard consists of 4 pages:

| Page | Content |
|---|---|
| **Overview** | Key metrics, total pedestrian count, active sensors |
| **Trend & Pattern** | Daily and weekly trends, hourly and day-of-week patterns |
| **Weather Impact** | Correlation between weather conditions and pedestrian volume |
| **Forecasting** | 7-day predictions per sensor with model performance metrics |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Orchestration | GitHub Actions |
| Ingestion | Python, Requests |
| Storage | Google BigQuery |
| Transformation | dbt |
| Forecasting | XGBoost, Scikit-learn, Open-Meteo API |
| Visualization | Power BI |

---

## Notes

- Sensors with historical data older than 30 days from the current date are automatically skipped
- Forecasting uses real weather forecast data from Open-Meteo rather than historical medians
- GitHub Actions usage: approximately 450 minutes per month (ingestion + forecasting) out of the 3,000-minute limit on GitHub Pro