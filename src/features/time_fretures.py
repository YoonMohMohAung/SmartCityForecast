def add_time_features(df):
    df = df.copy()

    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # lag features (5-min steps)
    df["lag_1"] = df["metropolitan_demand"].shift(1)
    df["lag_12"] = df["metropolitan_demand"].shift(12)    # 1 hour
    df["lag_288"] = df["metropolitan_demand"].shift(288)  # 1 day

    # rolling mean
    df["rolling_1h"] = df["metropolitan_demand"].rolling(12).mean()
    df["rolling_1d"] = df["metropolitan_demand"].rolling(288).mean()

    df = df.dropna()
    return df
