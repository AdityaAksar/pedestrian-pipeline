import requests
import pandas as pd
from datetime import datetime


BASE_URL = "https://date.nager.at/api/v3/PublicHolidays"
COUNTRY_CODE = "AU"
VICTORIA_COUNTY = "AU-VIC"


def fetch_holidays(years: list = None) -> pd.DataFrame:
    if years is None:
        current_year = datetime.today().year
        years = [current_year, current_year + 1]

    print(f"[holidays] Fetching holidays for years: {years}...")

    all_records = []
    for year in years:
        url = f"{BASE_URL}/{year}/{COUNTRY_CODE}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        holidays = resp.json()

        for h in holidays:
            counties = h.get("counties") or []
            if not counties or VICTORIA_COUNTY in counties:
                all_records.append({
                    "date":        h["date"],
                    "name":        h["localName"],
                    "is_national": not bool(counties),
                    "year":        year,
                    "ingested_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
                })

    if not all_records:
        print("[holidays] No holidays found.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    print(f"[holidays] Done. Total holidays: {len(df)}")
    return df