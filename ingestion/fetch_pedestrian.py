import requests
import pandas as pd
from datetime import datetime, timedelta


BASE_URL = (
    "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/"
    "pedestrian-counting-system-monthly-counts-per-hour/records"
)


def fetch_pedestrian(days_back: int = 3) -> pd.DataFrame:
    date_from = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    print(f"[pedestrian] Fetching data from {date_from}...")

    all_records = []
    offset = 0
    limit = 100

    while True:
        params = {
            "where": f"sensing_date >= '{date_from}'",
            "order_by": "sensing_date desc",
            "limit": limit,
            "offset": offset,
        }
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        records = data.get("results", [])
        if not records:
            break

        all_records.extend(records)
        print(f"[pedestrian] Fetched {len(all_records)} records so far...")

        if len(records) < limit:
            break
        offset += limit

    if not all_records:
        print("[pedestrian] No records found.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    df["lon"] = df["location"].apply(lambda x: x.get("lon") if isinstance(x, dict) else None)
    df["lat"] = df["location"].apply(lambda x: x.get("lat") if isinstance(x, dict) else None)
    df = df.drop(columns=["location"])
    df["sensing_date"] = pd.to_datetime(df["sensing_date"]).dt.strftime("%Y-%m-%d")
    df["sensing_datetime"] = (
        pd.to_datetime(df["sensing_date"]) + pd.to_timedelta(df["hourday"], unit="h")
    ).dt.strftime("%Y-%m-%dT%H:%M:%S")

    df["ingested_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    print(f"[pedestrian] Done. Total records: {len(df)}")
    return df


def fetch_sensor_locations() -> pd.DataFrame:
    url = (
        "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/"
        "pedestrian-counting-system-sensor-locations/records"
    )
    print("[sensor_locations] Fetching sensor locations...")

    all_records = []
    offset = 0
    limit = 100

    while True:
        params = {"limit": limit, "offset": offset}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        records = data.get("results", [])
        if not records:
            break

        all_records.extend(records)
        if len(records) < limit:
            break
        offset += limit

    if not all_records:
        print("[sensor_locations] No records found.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["ingested_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    print(f"[sensor_locations] Done. Total sensors: {len(df)}")
    return df