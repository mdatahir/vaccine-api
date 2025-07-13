from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="Vaccine Hesitancy Predictor by Tahir",
    description="A FastAPI app to predict vaccine hesitancy based on demographic inputs.",
    version="1.0"
)

# Define input schema
class PredictionInput(BaseModel):
    age: int
    sex: str
    marital_status: str
    place: str
    edu: int
    qua: int
    job: int
    jobst: int
    child: int

# Load model and encoders
clf = joblib.load("vaccine_model.pkl")
le_sex = joblib.load("le_sex.pkl")
le_marital = joblib.load("le_marital.pkl")
le_place = joblib.load("le_place.pkl")

@app.get("/")
def read_root():
    return {"message": "Vaccine Hesitancy Predictor API by Tahir is up and running."}

@app.post("/predict")
def predict(input: PredictionInput):
    try:
        # Prepare input data
        input_data = [[
            input.age,
            le_sex.transform([input.sex])[0],
            le_marital.transform([input.marital_status])[0],
            le_place.transform([input.place])[0],
            input.edu,
            input.qua,
            input.job,
            input.jobst,
            input.child
        ]]

        # Predict
        prediction = clf.predict(input_data)[0]
        label = "Vaccine Hesitant (Cluster 1)" if prediction == 1 else "Vaccine Confident (Cluster 0)"
        return {"prediction": label}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
