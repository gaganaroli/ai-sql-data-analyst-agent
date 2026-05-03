import pandas as pd
import sqlite3

def load_csv(csv_file):
    df = pd.read_csv(csv_file)

    conn = sqlite3.connect("data.db")

    df.to_sql(
        "sales_data",
        conn,
        if_exists="replace",
        index=False
    )

    return conn, df