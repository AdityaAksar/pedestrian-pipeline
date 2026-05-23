import sys
from ingestion.fetch_pedestrian import fetch_pedestrian, fetch_sensor_locations
from ingestion.fetch_weather import fetch_weather
from ingestion.fetch_holiday import fetch_holidays
from ingestion.load_bigquery import load_to_bigquery, deduplicate_table
from datetime import datetime


def run_ingestion():
    print("=" * 60)
    print(f"Starting ingestion run at {datetime.utcnow()} UTC")
    print("=" * 60)

    errors = []

    # 1. Pedestrian counts (incremental, last 3 days)
    try:
        df_pedestrian = fetch_pedestrian(days_back=3)
        load_to_bigquery(df_pedestrian, "pedestrian_counts", write_disposition="WRITE_APPEND")
        deduplicate_table("pedestrian_counts", unique_key="id")
    except Exception as e:
        print(f"[ERROR] pedestrian_counts: {e}")
        errors.append(f"pedestrian_counts: {e}")

    # 2. Sensor locations (static, overwrite setiap run)
    try:
        df_sensors = fetch_sensor_locations()
        load_to_bigquery(df_sensors, "sensor_locations", write_disposition="WRITE_TRUNCATE")
    except Exception as e:
        print(f"[ERROR] sensor_locations: {e}")
        errors.append(f"sensor_locations: {e}")

    # 3. Weather (incremental, last 5 days)
    try:
        df_weather = fetch_weather(days_back=5)
        if not df_weather.empty:
            load_to_bigquery(df_weather, "weather_hourly", write_disposition="WRITE_APPEND")
            deduplicate_table("weather_hourly", unique_key="datetime")
        else:
            print("[weather] Skipping load and dedup — no data fetched.")
    except Exception as e:
        print(f"[ERROR] weather_hourly: {e}")
        errors.append(f"weather_hourly: {e}")

    # 4. Public holidays (static, overwrite setiap run)
    try:
        current_year = datetime.today().year
        df_holidays = fetch_holidays(years=[current_year, current_year + 1])
        load_to_bigquery(df_holidays, "public_holidays", write_disposition="WRITE_TRUNCATE")
    except Exception as e:
        print(f"[ERROR] public_holidays: {e}")
        errors.append(f"public_holidays: {e}")

    # Summary
    print("=" * 60)
    if errors:
        print(f"Ingestion completed WITH ERRORS:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("Ingestion completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_ingestion()