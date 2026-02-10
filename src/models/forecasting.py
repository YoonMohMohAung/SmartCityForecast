import numpy as np


def recursive_forecast(model, last_window, steps):
    preds = []
    window = last_window.copy()

    for _ in range(steps):
        next_pred = model.predict(window.reshape(1, -1))[0]
        preds.append(next_pred)
        window = np.roll(window, -1)
        window[-1] = next_pred

    return preds
