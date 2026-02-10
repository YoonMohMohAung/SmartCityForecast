import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

from src.models.model_utils import evaluate


def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def run_lstm(series, window_size=288):
    """
    window_size = 288 (1 day of 5-min data)
    """

    values = series.values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    train_size = int(len(scaled) * 0.8)
    train, test = scaled[:train_size], scaled[train_size - window_size:]

    X_train, y_train = create_sequences(train, window_size)
    X_test, y_test = create_sequences(test, window_size)

    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(window_size, 1)),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=3)],
        verbose=1
    )

    preds = model.predict(X_test)
    preds = scaler.inverse_transform(preds)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    evaluate("LSTM", y_test.flatten(), preds.flatten())
    return model