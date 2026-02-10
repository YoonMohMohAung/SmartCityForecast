from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.models.model_utils import evaluate


def run_sarimax(train, test):
    model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 288),  # daily seasonality
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted = model.fit(disp=False)
    preds = fitted.forecast(len(test))

    evaluate("SARIMAX", test.values, preds.values)
