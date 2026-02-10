from sklearn.model_selection import train_test_split
from src.utils.metrics import rmse, mape


def time_series_split(df, target, test_size=0.2):
    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(X, y, test_size=test_size, shuffle=False)


def evaluate(name, y_true, y_pred):
    print(f"\n{name}")
    print("-" * len(name))
    print("RMSE:", rmse(y_true, y_pred))
    print("MAPE:", mape(y_true, y_pred))
