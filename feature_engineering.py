def add_technical_indicators(df, sma_period=10, ema_period=10):
    df = df.copy()
    df['SMA'] = df['Close'].rolling(window=sma_period, min_periods=1).mean()
    df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    return df