from src.data.load_data import load_energy_data
from src.data.clean_data import clean_energy_data
from src.features.time_features import add_time_features

from src.models.baseline import run_baseline
from src.models.sarimax import run_sarimax
from src.models.lstm import run_lstm
from src.models.xgboost_model import run_xgboost


def main():
    df = load_energy_data("data/raw/energy.csv")
    df = clean_energy_data(df)
    df_feat = add_time_features(df)

    train_size = int(len(df_feat) * 0.8)
    train = df_feat.iloc[:train_size]["metropolitan_demand"]
    test = df_feat.iloc[train_size:]["metropolitan_demand"]

    run_baseline(train, test)
    run_sarimax(train, test)
    run_lstm(df["metropolitan_demand"])
    run_xgboost(df_feat)


if __name__ == "__main__":
    main()
