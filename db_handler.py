import sqlite3
import pandas as pd
import os

os.makedirs("data", exist_ok=True)
DB_FILE = os.path.join("data", "stock_data.db")

def create_connection(db_file=DB_FILE):
    return sqlite3.connect(db_file)

def create_stock_data_table(conn):
    sql = """
    CREATE TABLE IF NOT EXISTS stock_data (
        date TEXT PRIMARY KEY,
        Open REAL,
        High REAL,
        Low REAL,
        Close REAL,
        Adj_Close REAL,
        Volume INTEGER,
        SMA REAL,
        EMA REAL
    );
    """
    conn.cursor().execute(sql)
    conn.commit()

def save_stock_data(df, db_file=DB_FILE):
    conn = create_connection(db_file)
    create_stock_data_table(conn)
    df.copy().reset_index().to_sql('stock_data', conn, if_exists='replace', index=False)
    conn.close()
    print("Stock data saved to database.")

def create_predictions_table(conn):
    sql = """
    CREATE TABLE IF NOT EXISTS predictions (
        date TEXT PRIMARY KEY,
        predicted REAL,
        actual REAL
    );
    """
    conn.cursor().execute(sql)
    conn.commit()

def save_predictions(dates, predictions, actuals, db_file=DB_FILE):
    conn = create_connection(db_file)
    create_predictions_table(conn)
    data = {"date": [str(d) for d in dates],
            "predicted": [float(p) for p in predictions],
            "actual": [float(a) for a in actuals]}
    pd.DataFrame(data).to_sql('predictions', conn, if_exists='replace', index=False)
    conn.close()
    print("Predictions saved to database.")