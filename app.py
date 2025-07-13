from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load trained model and encoders
clf = joblib.load("vaccine_model.pkl")
le_sex = joblib.load("le_sex.pkl")
le_marital = joblib.load("le_marital.pkl")
le_place = joblib.load("le_place.pkl")

# Create FastAPI app
app = FastAPI()

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

@app.post("/predict")
def predict(input: UserInput):
    # Encode categorical inputs
    sex_enc = le_sex.transform([input.sex.capitalize()])[0]
    marital_enc = le_marital.transform([input.marital_status.capitalize()])[0]
    place_enc = le_place.transform([input.place.capitalize()])[0]

    # Create feature vector
    X = [[
        input.age, sex_enc, marital_enc, place_enc,
        input.edu, input.qua, input.job, input.jobst, input.child
    ]]

    pred = clf.predict(X)[0]
    result = "Vaccine Hesitant" if pred == 1 else "Vaccine Confident"

    return {"prediction": result}
