import requests
import pandas as pd
from datetime import datetime, timedelta


BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
LAT = -37.8136
LON = 144.9631
HOURLY_VARS = [
    "temperature_2m",
    "precipitation",
    "windspeed_10m",
    "relativehumidity_2m",
]


def fetch_weather(days_back: int = 5) -> pd.DataFrame:
    end = (datetime.today() - timedelta(days=2)).strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    print(f"[weather] Fetching weather from {start} to {end} via Open-Meteo...")

    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Australia/Melbourne",
    }

    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    if not hourly:
        print("[weather] No data returned.")
        return pd.DataFrame()

    df = pd.DataFrame(hourly)
    df = df.rename(columns={
        "time":                 "datetime",
        "temperature_2m":       "temperature_c",
        "precipitation":        "precipitation_mm",
        "windspeed_10m":        "windspeed_ms",
        "relativehumidity_2m":  "humidity_pct",
    })

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.strftime("%Y-%m-%d")
    df["hour"] = df["datetime"].dt.hour
    df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    df["ingested_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    print(f"[weather] Done. Total rows: {len(df)}")
    return df