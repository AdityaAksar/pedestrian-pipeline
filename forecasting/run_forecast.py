'''
run_forecast.py
---------------
Flow:
  1. Fetch Data from BigQuery
  2. Loop for each sensor:
     a. Fetch data from mart_pedestrian_hourly
     b. Create Feature (hour, day, weather, lag, rolling mean)
     c. Train XGBoost
     d. Generate Forecasting Data
     e. Append to mart.predictions
'''

import os
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.oauth2 import service_account
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

MELBOURNE_LAT = -37.8136
MELBOURNE_LON = 144.9631
PROJECT_ID    = 'melbourne-pedestrian-pipeline'

# ── Auth ──────────────────────────────────────────────────────────────────────
def get_bq_client():
    gcp_key = os.environ.get('GCP_SA_KEY')
    if not gcp_key:
        raise ValueError('GCP_SA_KEY environment variable not set')
    key_dict = json.loads(gcp_key)
    creds = service_account.Credentials.from_service_account_info(
        key_dict,
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    return bigquery.Client(
        project='melbourne-pedestrian-pipeline',
        credentials=creds
    )

# ── Open-Meteo: Fetch Forecasting ───────────────────────────────────
def fetch_weather_forecast() -> pd.DataFrame:
    print('Fetching weather forecast from Open-Meteo...')
    url = 'https://api.open-meteo.com/v1/forecast'
    params = {
        'latitude':  MELBOURNE_LAT,
        'longitude': MELBOURNE_LON,
        'hourly':    'temperature_2m,precipitation,windspeed_10m,relativehumidity_2m',
        'forecast_days': 7,
        'timezone':  'Australia/Melbourne'
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
 
    df_weather = pd.DataFrame({
        'sensing_datetime': pd.to_datetime(data['hourly']['time']),
        'temperature_c':    data['hourly']['temperature_2m'],
        'precipitation_mm': data['hourly']['precipitation'],
        'windspeed_ms':     data['hourly']['windspeed_10m'],
        'humidity_pct':     data['hourly']['relativehumidity_2m'],
    })
    print(f"Weather forecast: {len(df_weather)} jam ({df_weather['sensing_datetime'].min().date()} s/d {df_weather['sensing_datetime'].max().date()})")
    return df_weather

# ── Features ──────────────────────────────────────────────────────────────────
FEATURES = [
    'hour', 'day_of_week', 'month', 'quarter',
    'is_weekend', 'is_public_holiday',
    'temperature_c', 'precipitation_mm', 'windspeed_ms', 'humidity_pct',
    'lag_1h', 'lag_24h', 'lag_168h',
    'rolling_mean_24h', 'rolling_mean_168h'
]

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values('sensing_datetime').reset_index(drop=True)
    df['hour']              = df['sensing_datetime'].dt.hour
    df['day_of_week']       = df['sensing_datetime'].dt.dayofweek
    df['month']             = df['sensing_datetime'].dt.month
    df['quarter']           = df['sensing_datetime'].dt.quarter
    df['is_weekend']        = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_public_holiday'] = df['is_public_holiday'].astype(int)
    for col in ['temperature_c', 'precipitation_mm', 'windspeed_ms', 'humidity_pct']:
        df[col] = df[col].fillna(df[col].median())
    df['lag_1h']            = df['pedestrian_count'].shift(1)
    df['lag_24h']           = df['pedestrian_count'].shift(24)
    df['lag_168h']          = df['pedestrian_count'].shift(168)
    df['rolling_mean_24h']  = df['pedestrian_count'].shift(1).rolling(24).mean()
    df['rolling_mean_168h'] = df['pedestrian_count'].shift(1).rolling(168).mean()
    return df

# ── Forecast future 7 days ────────────────────────────────────────────────────
def generate_future_features(df_hist: pd.DataFrame, hours: int = 168) -> pd.DataFrame:
    last_dt = df_hist['sensing_datetime'].max()
    future_dts = [last_dt + timedelta(hours=i+1) for i in range(hours)]

    df_hist['hour'] = df_hist['sensing_datetime'].dt.hour
    weather_by_hour = df_hist.groupby('hour')[
        ['temperature_c', 'precipitation_mm', 'windspeed_ms', 'humidity_pct']
    ].median().reset_index()

    rows = []
    buffer = list(df_hist['pedestrian_count'].tail(168).values)

    for dt in future_dts:
        hour = dt.hour
        dow  = dt.weekday()
        weather = weather_by_hour[weather_by_hour['hour'] == hour]
        temp   = float(weather['temperature_c'].values[0])   if len(weather) else 15.0
        precip = float(weather['precipitation_mm'].values[0]) if len(weather) else 0.0
        wind   = float(weather['windspeed_ms'].values[0])    if len(weather) else 3.0
        humid  = float(weather['humidity_pct'].values[0])    if len(weather) else 60.0

        lag_1h   = buffer[-1]   if len(buffer) >= 1   else 0
        lag_24h  = buffer[-24]  if len(buffer) >= 24  else 0
        lag_168h = buffer[-168] if len(buffer) >= 168 else 0
        roll_24  = np.mean(buffer[-24:])  if len(buffer) >= 24  else lag_1h
        roll_168 = np.mean(buffer[-168:]) if len(buffer) >= 168 else roll_24

        rows.append({
            'sensing_datetime':  dt,
            'hour':              hour,
            'day_of_week':       dow,
            'month':             dt.month,
            'quarter':           (dt.month - 1) // 3 + 1,
            'is_weekend':        int(dow in [5, 6]),
            'is_public_holiday': 0,
            'temperature_c':     temp,
            'precipitation_mm':  precip,
            'windspeed_ms':      wind,
            'humidity_pct':      humid,
            'lag_1h':            lag_1h,
            'lag_24h':           lag_24h,
            'lag_168h':          lag_168h,
            'rolling_mean_24h':  roll_24,
            'rolling_mean_168h': roll_168,
        })
        buffer.append(0)

    return pd.DataFrame(rows)

# ── Build future feature rows ──────────────────────────────────────────────────
def build_future_rows(df_hist: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    last_dt = df_hist['sensing_datetime'].max()
    future_dts = [last_dt + timedelta(hours=i+1) for i in range(168)]
 
    df_weather = df_weather.copy()
    df_weather['sensing_datetime'] = df_weather['sensing_datetime'].dt.tz_localize(None)
 
    rows = []
    buffer = list(df_hist['pedestrian_count'].tail(168).values)
 
    for dt in future_dts:
        w = df_weather[df_weather['sensing_datetime'] == dt]
        if len(w) > 0:
            temp   = float(w['temperature_c'].values[0])
            precip = float(w['precipitation_mm'].values[0])
            wind   = float(w['windspeed_ms'].values[0])
            humid  = float(w['humidity_pct'].values[0])
        else:
            hour_mask = df_hist['sensing_datetime'].dt.hour == dt.hour
            temp   = float(df_hist.loc[hour_mask, 'temperature_c'].median())
            precip = float(df_hist.loc[hour_mask, 'precipitation_mm'].median())
            wind   = float(df_hist.loc[hour_mask, 'windspeed_ms'].median())
            humid  = float(df_hist.loc[hour_mask, 'humidity_pct'].median())
 
        lag_1h   = buffer[-1]   if len(buffer) >= 1   else 0
        lag_24h  = buffer[-24]  if len(buffer) >= 24  else 0
        lag_168h = buffer[-168] if len(buffer) >= 168 else 0
        roll_24  = np.mean(buffer[-24:])  if len(buffer) >= 24  else lag_1h
        roll_168 = np.mean(buffer[-168:]) if len(buffer) >= 168 else roll_24
 
        rows.append({
            'sensing_datetime':  dt,
            'hour':              dt.hour,
            'day_of_week':       dt.weekday(),
            'month':             dt.month,
            'quarter':           (dt.month - 1) // 3 + 1,
            'is_weekend':        int(dt.weekday() in [5, 6]),
            'is_public_holiday': 0,
            'temperature_c':     temp,
            'precipitation_mm':  precip,
            'windspeed_ms':      wind,
            'humidity_pct':      humid,
            'lag_1h':            lag_1h,
            'lag_24h':           lag_24h,
            'lag_168h':          lag_168h,
            'rolling_mean_24h':  roll_24,
            'rolling_mean_168h': roll_168,
        })
        buffer.append(0)
        buffer = buffer[-168:]
 
    return pd.DataFrame(rows)
 

# ── Per-sensor pipeline ───────────────────────────────────────────────────────
def run_sensor(client: bigquery.Client, sensor_name: str, location_id: int, df_weather: pd.DataFrame) -> pd.DataFrame | None:
    print(f"  → {sensor_name} (location_id={location_id})")

    # Fetch Data
    query = f"""
        SELECT
            sensing_datetime, pedestrian_count,
            is_public_holiday,
            temperature_c, precipitation_mm, windspeed_ms, humidity_pct
        FROM `{PROJECT_ID}.staging.mart_pedestrian_hourly`
        WHERE sensor_name = '{sensor_name}'
          AND location_id = {location_id}
        ORDER BY sensing_datetime
    """
    try:
        df = client.query(query).to_dataframe()
    except Exception as e:
        print(f"[SKIP] Query Failed: {e}")
        return None

    if len(df) < 300:
        print(f"[SKIP] Insufficient Data: {len(df)} rows")
        return None
    df['sensing_datetime'] = pd.to_datetime(df['sensing_datetime']).dt.tz_localize(None)

    # Feature engineering
    df_feat = create_features(df)
    df_feat = df_feat.dropna(subset=FEATURES + ['pedestrian_count']).reset_index(drop=True)

    if len(df_feat) < 200:
        print(f"[SKIP] Insufficient data after dropna: {len(df_feat)} rows")
        return None

    # Train/test split
    split_idx = int(len(df_feat) * 0.8)
    X_train = df_feat.iloc[:split_idx][FEATURES]
    y_train = df_feat.iloc[:split_idx]['pedestrian_count']
    X_test  = df_feat.iloc[split_idx:][FEATURES]
    y_test  = df_feat.iloc[split_idx:]['pedestrian_count']

    # Train XGBoost
    model = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0
    )
    model.fit(X_train, y_train)

    # Calculate Metric
    y_pred_test = np.maximum(model.predict(X_test), 0)
    mae  = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mape = np.mean(np.abs((y_test.values - y_pred_test) / (y_test.values + 1))) * 100
    print(f"    MAE={mae:.1f}  RMSE={rmse:.1f}  MAPE={mape:.1f}%")

    # Build future rows with Open-Meteo weather
    df_future = build_future_rows(df, df_weather)

    # Iterative prediction (update lag from previous predictions)
    buffer = list(df['pedestrian_count'].tail(168).values)
    preds  = []
    for i, row in df_future.iterrows():
        row = row.copy()
        row['lag_1h']           = buffer[-1]   if len(buffer) >= 1   else 0
        row['lag_24h']          = buffer[-24]  if len(buffer) >= 24  else 0
        row['lag_168h']         = buffer[-168] if len(buffer) >= 168 else 0
        row['rolling_mean_24h'] = np.mean(buffer[-24:])  if len(buffer) >= 24  else row['lag_1h']
        row['rolling_mean_168h']= np.mean(buffer[-168:]) if len(buffer) >= 168 else row['rolling_mean_24h']
 
        pred = float(np.maximum(model.predict(row[FEATURES].values.reshape(1, -1)), 0)[0])
        preds.append(pred)
        buffer.append(pred)
        buffer = buffer[-168:]
 
    df_future['predicted_count'] = preds
    df_future['actual_count']    = np.nan
    df_future['sensor_name']     = sensor_name
    df_future['location_id']     = location_id
    df_future['model']           = 'XGBoost'
    df_future['mae']             = mae
    df_future['rmse']            = rmse
    df_future['mape']            = mape
    df_future['predicted_at']    = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
 
    return df_future[[
        'sensor_name', 'location_id', 'sensing_datetime',
        'actual_count', 'predicted_count',
        'model', 'mae', 'rmse', 'mape', 'predicted_at'
    ]]

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== Melbourne Pedestrian Forecasting Pipeline ===")
    print(f"Run time: {datetime.now().isoformat()}")

    df_weather = fetch_weather_forecast()

    client = get_bq_client()

    sensors_query = '''
        SELECT DISTINCT sensor_name, location_id
        FROM `melbourne-pedestrian-pipeline.staging.mart_pedestrian_hourly`
        ORDER BY sensor_name
    '''
    df_sensors = client.query(sensors_query).to_dataframe()
    print(f"\nTotal sensor: {len(df_sensors)}")

    client.query('''
        DELETE FROM `melbourne-pedestrian-pipeline.mart.predictions`
        WHERE predicted_at IS NOT NULL
    ''').result()
    print("Prediction table deleted.\n")

    all_results = []
    success = 0
    skipped = 0

    for _, row in df_sensors.iterrows():
        result = run_sensor(client, row['sensor_name'], int(row['location_id']), df_weather)
        if result is not None:
            all_results.append(result)
            success += 1
        else:
            skipped += 1

    if not all_results:
        print('\n[ERROR] No sensor was successfully processed.')
        return

    # Load BigQuery
    df_all = pd.concat(all_results, ignore_index=True)
    df_all['sensing_datetime'] = df_all['sensing_datetime'].astype(str)

    job = client.load_table_from_dataframe(
        df_all,
        'melbourne-pedestrian-pipeline.mart.predictions',
        job_config=bigquery.LoadJobConfig(
            write_disposition='WRITE_TRUNCATE',
            autodetect=True
        )
    )
    job.result()

    print(f"\n{'='*50}")
    print(f"Done! {success} sensor successfully processed, {skipped} skipped.")
    print(f"Total rows saved: {len(df_all):,}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()