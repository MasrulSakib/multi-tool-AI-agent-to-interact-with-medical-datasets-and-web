import sqlite3
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"

# Each entry: (csv filename, database filename, table name)
DATASETS = [
    ("heart_disease.csv", "heart_disease.db", "heart_disease_records"),
    ("cancer.csv", "cancer.db", "cancer_records"),
    ("diabetes.csv", "diabetes.db", "diabetes_records"),
]


def convert_csv_to_sqlite(csv_filename: str, db_filename: str, table_name: str):
    csv_path = RAW_DIR / csv_filename
    db_path = DATA_DIR / db_filename

    if not csv_path.exists():
        print(f"  Skipping {csv_filename} -- not found in data/raw/. "
              f"Download it from Kaggle and place it there first.")
        return

    # Step 1: read the CSV into a pandas DataFrame. pandas automatically
    # figures out sensible column types (int, float, text) for us.
    df = pd.read_csv(csv_path)
    print(f"  Read {len(df)} rows, {len(df.columns)} columns from {csv_filename}")
    print(f"  Columns: {list(df.columns)}")

    # Step 2: write the DataFrame straight into a SQLite table.
    # if_exists="replace" means re-running this script rebuilds the DB
    # from scratch each time, which is what we want during development.
    connection = sqlite3.connect(db_path)
    df.to_sql(table_name, connection, if_exists="replace", index=False)
    connection.close()

    print(f"  Saved to {db_filename} as table '{table_name}'\n")


def main():
    print("Converting CSVs to SQLite databases...\n")
    for csv_filename, db_filename, table_name in DATASETS:
        print(f"Processing {csv_filename} -> {db_filename}")
        convert_csv_to_sqlite(csv_filename, db_filename, table_name)
    print("Done.")


if __name__ == "__main__":
    main()
