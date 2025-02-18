from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

def build_lstm_model(input_shape, layer_units):
    """
    Builds an LSTM model with two hidden LSTM layers and a Dense output layer.
    """
    inp = Input(shape=input_shape)
    x = LSTM(units=layer_units, return_sequences=True)(inp)
    x = LSTM(units=layer_units)(x)
    out = Dense(1, activation="linear")(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model

def train_model(model, X_train, y_train, epochs, batch_size):
    """
    Trains the LSTM model with the given training data and hyperparameters.
    Returns the training history.
    """
    history = model.fit(
        X_train, 
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        shuffle=True,
        verbose=1
    )
    return history