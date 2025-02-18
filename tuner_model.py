import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
import keras_tuner as kt
from config import WINDOW_SIZE

def build_model(hp):
    input_shape = (WINDOW_SIZE, 3)
    inputs = Input(shape=input_shape)
    x = inputs
    num_layers = hp.Int("num_layers", 1, 3)
    for i in range(num_layers):
        units = hp.Int(f"units_{i}", min_value=32, max_value=128, step=32)
        return_seq = True if i < num_layers - 1 else False
        x = LSTM(units=units, return_sequences=return_seq)(x)
        if hp.Boolean(f"dropout_{i}"):
            dropout_rate = hp.Float(f"dropout_rate_{i}", min_value=0.1, max_value=0.5, step=0.1)
            x = Dropout(rate=dropout_rate)(x)
    outputs = Dense(1, activation="linear")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def run_tuner(X_train, y_train, max_trials, executions_per_trial, epochs, batch_size):
    tuner = kt.RandomSearch(
        build_model,
        objective="val_loss",
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory="tuner_dir",
        project_name="stock_forecasting"
    )
    tuner.search(X_train, y_train, epochs=epochs, validation_split=0.1, batch_size=batch_size)
    return tuner.get_best_models(num_models=1)[0], tuner