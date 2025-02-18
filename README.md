# Stock Forecasting with LSTM and Keras Tuner

This project is a robust, end-to-end stock forecasting pipeline. It leverages historical stock data, technical indicators, and hyperparameter tuning to build a predictive LSTM model. The project includes modules for data retrieval, feature engineering, data preprocessing, model tuning, training, evaluation, and persistence.

## Project Overview

The goal of this project is to forecast stock prices (using AAPL as an example) by:

- **Downloading Stock Data:**  
  Retrieves historical stock data from Alpha Vantage using the free `TIME_SERIES_DAILY` endpoint (with a fallback to synthetic data) or from yfinance if desired.
  
- **Feature Engineering:**  
  Adds technical indicators such as Simple Moving Average (SMA) and Exponential Moving Average (EMA) to enrich the dataset.
  
- **Data Preprocessing:**  
  Scales multiple features and creates sequential inputs required for LSTM-based forecasting.
  
- **Hyperparameter Tuning:**  
  Uses Keras Tuner to search over a range of LSTM configurations (e.g., number of layers, units, dropout) to identify an optimal model.
  
- **Model Training & Evaluation:**  
  Fine-tunes the best model and evaluates its performance using RMSE and MAPE metrics. Predictions and raw data are stored in a local SQLite database.
  
- **Persistence:**  
  The final model is saved in the native Keras format (`.keras`), and predictions are stored in SQL for further analysis.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/stock_forecasting_lstm_kerastuner.git
   cd stock_forecasting_lstm_kerastuner

2. **Create and Activate a Conda Environment:**

    ```bash
    conda create --name stock_forecasting_lstm_kerastuner python=3.8
    conda activate stock_forecasting_lstm_kerastuner

3. **Install Dependencies:**

   ```bash
    pip install -r requirements.txt

## Configuration

Open config.py and set the following:

1. **Ticker and Date Range:**
    TICKER (default is AAPL)
    START_DATE and END_DATE to define the historical range.

2. **Data Source:**
    DATA_SOURCE (choose "ALPHA_VANTAGE" or "YFINANCE")
    Replace ALPHA_VANTAGE_API_KEY with your valid Alpha Vantage API key.

3. **Model & Tuner Settings:**
    Adjust the model hyperparameters and tuner settings if needed.

## Usage
To run the complete pipeline, execute:
    python main.py

The script will:
1. Download and preprocess stock data.
2. Enrich the data with technical indicators.
3. Split the data into training and test sets.
4. Use Keras Tuner to find and fine-tune the best LSTM model.
5. Evaluate the model using RMSE and MAPE.
6. Save the final model as lstm_stock_model.keras and store predictions in a SQLite database.
7. Generate plots for both the training history and the prediction trends.
