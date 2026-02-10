from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="Energy Forecast API")

model = joblib.load("models/xgboost.pkl")


@app.post("/predict")
def predict(last_288_values: list[float]):
    arr = np.array(last_288_values).reshape(1, -1)
    prediction = model.predict(arr)
    return {"prediction": float(prediction[0])}
