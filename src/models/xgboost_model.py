from xgboost import XGBRegressor
from src.models.model_utils import time_series_split, evaluate


def run_xgboost(df):
    X_train, X_test, y_train, y_test = time_series_split(
        df, target="metropolitan_demand"
    )

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    evaluate("XGBoost", y_test, preds)
    return model