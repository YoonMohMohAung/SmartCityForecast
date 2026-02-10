import pandas as pd


def clean_energy_data(df):
    df = df.copy()

    # keep only numeric columns
    df = df.apply(pd.to_numeric, errors="coerce")

    # focus only on energy (no imports/exports for now)
    target = "metropolitan_demand"

    df = df[[target]]
    df = df.dropna()

    return df
