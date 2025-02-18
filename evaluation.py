import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import math

def calculate_rmse(actual, predicted):
    return math.sqrt(mean_squared_error(actual, predicted))

def calculate_mape(actual, predicted):
    return mean_absolute_percentage_error(actual, predicted) * 100

def plot_stock_trend(train, test, predictions, output_path="results/stock_predictions.png"):
    plt.figure(figsize=(20,10))
    plt.plot(train.index, train["Close"], label="Train Closing Price")
    plt.plot(test.index, test["Close"], label="Test Closing Price")
    plt.plot(test.index, predictions, label="Predicted Closing Price")
    plt.title("LSTM Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Stock Price ($)")
    plt.legend(loc="upper left")
    plt.savefig(output_path)
    plt.show()

def plot_training_history(history, output_path="results/training_history.png"):
    plt.figure(figsize=(10,6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_path)
    plt.show()