import numpy as np
from sklearn.preprocessing import StandardScaler

def extract_sequences(data, window_size, output_window=1):
    X, y = [], []
    for i in range(window_size, len(data) - output_window + 1):
        X.append(data[i - window_size:i, :])
        y.append(data[i + output_window - 1, 0])
    return np.array(X), np.array(y)

def preprocess_data(df, window_size, train_fraction=0.8):
    features = ['Close', 'SMA', 'EMA']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features].values)
    train_size = int(len(scaled_data) * train_fraction)
    scaled_train = scaled_data[:train_size]
    scaled_test = scaled_data[train_size - window_size:]
    return scaler, scaled_train, scaled_test, train_size

def prepare_train_test_sequences(scaled_train, scaled_test, window_size):
    X_train, y_train = extract_sequences(scaled_train, window_size, output_window=1)
    X_test, _ = extract_sequences(scaled_test, window_size, output_window=1)
    return X_train, y_train, X_test