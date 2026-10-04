from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("heart_disease_model.pkl")

@app.get("/")
def home():
    return {"message": "Heart Disease API Running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    return {"prediction": int(prediction)}
