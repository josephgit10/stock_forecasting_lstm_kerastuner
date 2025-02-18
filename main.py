import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from config import WINDOW_SIZE, LAYER_UNITS, EPOCHS, BATCH_SIZE, TRAIN_FRACTION, TICKER, MAX_TRIALS, EXECUTIONS_PER_TRIAL
from data_loader import load_stock_data
from feature_engineering import add_technical_indicators
from preprocessing import preprocess_data, prepare_train_test_sequences
from tuner_model import run_tuner
from evaluation import calculate_rmse, calculate_mape, plot_stock_trend, plot_training_history
import db_handler
import tensorflow as tf

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

def main():
    print(f"Downloading stock data for {TICKER}...")
    data = load_stock_data()
    print("Data loaded. Shape:", data.shape)
    
    data = add_technical_indicators(data, sma_period=10, ema_period=10)
    db_handler.save_stock_data(data)
    
    scaler, scaled_train, scaled_test, train_size = preprocess_data(data, WINDOW_SIZE, TRAIN_FRACTION)
    X_train, y_train, X_test = prepare_train_test_sequences(scaled_train, scaled_test, WINDOW_SIZE)
    print("Training sequences:", X_train.shape, y_train.shape)
    print("Test sequences:", X_test.shape)
    
    print("Starting hyperparameter tuning...")
    best_model, tuner = run_tuner(X_train, y_train, MAX_TRIALS, EXECUTIONS_PER_TRIAL, EPOCHS, BATCH_SIZE)
    print("Best model:")
    best_model.summary()
    
    print("Fine-tuning the best model...")
    history = best_model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=1)
    plot_training_history(history, output_path="results/training_history.png")
    
    print("Predicting on test data...")
    predictions_scaled = best_model.predict(X_test)
    predictions = scaler.inverse_transform(
        np.concatenate([predictions_scaled, np.zeros((predictions_scaled.shape[0], 2))], axis=1)
    )[:, 0]
    
    test_data = data.iloc[train_size:].copy().iloc[:len(predictions)]
    rmse = calculate_rmse(test_data["Close"].values, predictions)
    mape = calculate_mape(test_data["Close"].values, predictions)
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    plot_stock_trend(data.iloc[:train_size], test_data, predictions, output_path="results/stock_predictions.png")
    
    model_path = os.path.join("models", "lstm_stock_model.keras")
    best_model.save(model_path)
    print(f"Model saved as {model_path}")
    
    db_handler.save_predictions(test_data.index, predictions, test_data["Close"].values)

if __name__ == "__main__":
    main()