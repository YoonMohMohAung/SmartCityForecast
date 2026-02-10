from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


def tune_xgboost(X, y):
    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [4, 6],
        "learning_rate": [0.03, 0.05],
        "subsample": [0.8],
    }

    model = XGBRegressor(random_state=42)
    tscv = TimeSeriesSplit(n_splits=3)

    grid = GridSearchCV(
        model,
        param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    grid.fit(X, y)
    return grid.best_estimator_
