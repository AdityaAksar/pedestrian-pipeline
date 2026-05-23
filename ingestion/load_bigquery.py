import os
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import json


PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "melbourne-pedestrian-pipeline")
DATASET_RAW = "raw"


def get_bq_client() -> bigquery.Client:
    sa_key = os.environ.get("GCP_SA_KEY")
    if sa_key:
        key_dict = json.loads(sa_key)
        credentials = service_account.Credentials.from_service_account_info(
            key_dict,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return bigquery.Client(project=PROJECT_ID, credentials=credentials)
    return bigquery.Client(project=PROJECT_ID)


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    import datetime

    for col in df.columns:
        sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None

        if isinstance(sample, datetime.date) and not isinstance(sample, datetime.datetime):
            df[col] = df[col].apply(lambda x: x.isoformat() if pd.notna(x) and x is not None else None)

        elif isinstance(sample, datetime.datetime):
            df[col] = pd.to_datetime(df[col], errors="coerce")

        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype("Int64")

        elif df[col].dtype == object:
            non_str = df[col].dropna().apply(lambda x: not isinstance(x, str))
            if non_str.any():
                df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else None)

    return df


def load_to_bigquery(
    df: pd.DataFrame,
    table_name: str,
    write_disposition: str = "WRITE_APPEND",
) -> None:
    if df.empty:
        print(f"[bigquery] Skipping {table_name} — empty DataFrame.")
        return

    client = get_bq_client()
    table_id = f"{PROJECT_ID}.{DATASET_RAW}.{table_name}"

    df = sanitize_dataframe(df.copy())

    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
        autodetect=True,
    )

    print(f"[bigquery] Loading {len(df)} rows to {table_id}...")
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()

    table = client.get_table(table_id)
    print(f"[bigquery] Done. Table now has {table.num_rows} rows.")


def deduplicate_table(table_name: str, unique_key: str) -> None:
    client = get_bq_client()
    table_id = f"{PROJECT_ID}.{DATASET_RAW}.{table_name}"

    query = f"""
        CREATE OR REPLACE TABLE `{table_id}` AS
        SELECT * EXCEPT(row_num)
        FROM (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY {unique_key}
                    ORDER BY ingested_at DESC
                ) AS row_num
            FROM `{table_id}`
        )
        WHERE row_num = 1
    """

    print(f"[bigquery] Deduplicating {table_id} on key: {unique_key}...")
    client.query(query).result()
    print(f"[bigquery] Deduplication done.")