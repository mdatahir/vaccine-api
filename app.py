from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load model & encoders
clf = joblib.load("vaccine_model.pkl")
le_sex = joblib.load("le_sex.pkl")
le_marital = joblib.load("le_marital.pkl")
le_place = joblib.load("le_place.pkl")

class UserInput(BaseModel):
    age: int
    sex: str
    marital_status: str
    place: str
    edu: int
    qua: int
    job: int
    jobst: int
    child: int

@app.get("/")
def read_root():
    return {"message": "Vaccine Hesitancy Prediction API running ✅"}

@app.post("/predict")
def predict(data: UserInput):
    row = [[
        data.age,
        le_sex.transform([data.sex])[0],
        le_marital.transform([data.marital_status])[0],
        le_place.transform([data.place])[0],
        data.edu,
        data.qua,
        data.job,
        data.jobst,
        data.child
    ]]
    pred = clf.predict(row)[0]
    result = "Vaccine Hesitant" if pred == 1 else "Vaccine Confident"
    return {"prediction": result}
