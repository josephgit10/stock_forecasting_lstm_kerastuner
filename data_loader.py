import requests
import pandas as pd
import io
import numpy as np
from config import TICKER, START_DATE, END_DATE, ALPHA_VANTAGE_API_KEY, DATA_SOURCE
import yfinance as yf

def load_stock_data_alpha_vantage():
    url = (
        f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
        f"&symbol={TICKER}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}&datatype=csv"
    )
    response = requests.get(url)
    if response.status_code == 200:
        data = pd.read_csv(io.StringIO(response.text))
        if data.empty or "timestamp" not in data.columns:
            print("No data or unexpected format. Using synthetic data.")
            return generate_synthetic_data()
        data.rename(
            columns={
                "timestamp": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            },
            inplace=True
        )
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data.dropna(subset=["Date"], inplace=True)
        data.set_index("Date", inplace=True)
        data.sort_index(inplace=True)
        data = data[(data.index >= START_DATE) & (data.index <= END_DATE)]
        return data
    else:
        print("Alpha Vantage API error. Using synthetic data.")
        return generate_synthetic_data()

def load_stock_data_yfinance():
    data = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=True)
    if data.empty:
        print("YFinance download failed. Using synthetic data.")
        return generate_synthetic_data()
    data = data.reset_index()
    data["Date"] = pd.to_datetime(data["Date"])
    data.set_index("Date", inplace=True)
    return data

def generate_synthetic_data():
    date_rng = pd.date_range(start=START_DATE, end=END_DATE, freq='B')
    data = pd.DataFrame(date_rng, columns=['Date'])
    np.random.seed(42)
    data['Open'] = 100 + np.random.randn(len(date_rng)).cumsum()
    data['High'] = data['Open'] + np.random.rand(len(date_rng)) * 2
    data['Low'] = data['Open'] - np.random.rand(len(date_rng)) * 2
    data['Close'] = data['Open'] + np.random.randn(len(date_rng))
    data['Volume'] = np.random.randint(1000000, 5000000, size=len(date_rng))
    data.set_index('Date', inplace=True)
    return data

def load_stock_data():
    return load_stock_data_alpha_vantage() if DATA_SOURCE.upper() == "ALPHA_VANTAGE" else load_stock_data_yfinance()