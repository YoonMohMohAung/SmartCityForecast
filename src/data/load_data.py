import pandas as pd


def load_energy_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True)
    df = df.sort_values("datetime")
    df = df.set_index("datetime")

    return df
