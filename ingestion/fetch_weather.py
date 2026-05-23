import requests
import pandas as pd
from datetime import datetime, timedelta


BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
LAT = -37.8136
LON = 144.9631


def fetch_weather(days_back: int = 20) -> pd.DataFrame:
    today = datetime.today()
    end_dt   = today - timedelta(days=15)
    start_dt = today - timedelta(days=days_back)

    if start_dt > end_dt:
        start_dt = end_dt - timedelta(days=5)

    start = start_dt.strftime("%Y%m%d")
    end   = end_dt.strftime("%Y%m%d")

    print(f"[weather] Fetching weather from {start} to {end} via NASA POWER...")

    params = {
        "parameters": "T2M,PRECTOTCORR,WS10M,RH2M",
        "community": "RE",
        "longitude": LON,
        "latitude": LAT,
        "start": start,
        "end": end,
        "format": "JSON",
        "time-standard": "LST",
    }

    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hourly_data = data.get("properties", {}).get("parameter", {})
    if not hourly_data:
        print("[weather] No data returned from NASA POWER.")
        return pd.DataFrame()

    t2m  = hourly_data.get("T2M", {})
    prec = hourly_data.get("PRECTOTCORR", {})
    wind = hourly_data.get("WS10M", {})
    rhum = hourly_data.get("RH2M", {})

    records = []
    for key in t2m:
        try:
            dt = datetime.strptime(key, "%Y%m%d%H")
            records.append({
                "datetime":             dt.strftime("%Y-%m-%dT%H:%M:%S"),
                "date":                 dt.strftime("%Y-%m-%d"),
                "hour":                 dt.hour,
                "temperature_2m":       None if t2m.get(key)  == -999.0 else float(t2m.get(key)),
                "precipitation":        None if prec.get(key) == -999.0 else float(prec.get(key)),
                "windspeed_10m":        None if wind.get(key) == -999.0 else float(wind.get(key)),
                "relativehumidity_2m":  None if rhum.get(key) == -999.0 else float(rhum.get(key)),
                "ingested_at":          datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
            })
        except ValueError:
            continue

    df = pd.DataFrame(records)
    df = df.dropna(subset=["temperature_2m", "precipitation", "windspeed_10m", "relativehumidity_2m"], how="all")

    if df.empty:
        print("[weather] All values null — data not yet available.")
        return pd.DataFrame()

    print(f"[weather] Done. {len(df)} rows, {df['date'].min()} to {df['date'].max()}")
    return df