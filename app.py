from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model & encoders
clf = joblib.load("vaccine_model.pkl")
le_sex = joblib.load("le_sex.pkl")
le_marital = joblib.load("le_marital.pkl")
le_place = joblib.load("le_place.pkl")

app = FastAPI(
    title="Vaccine Hesitancy Prediction API",
    description="""
    🌟 This API predicts whether an individual is **Vaccine Hesitant** or **Vaccine Confident**
    based on demographic and socioeconomic factors.

    🔷 Use the `/predict` endpoint to submit the following fields:
    - `age`: Age of the person
    - `sex`: Gender (e.g., Male, Female)
    - `marital_status`: Marital Status (e.g., Single, Married)
    - `place`: Place of residence (e.g., Urban, Rural)
    - `edu`: Education level
    - `qua`: Qualification level
    - `job`: Job level
    - `jobst`: Job status
    - `child`: Number of children

    ✅ Returns: Vaccine Hesitant or Vaccine Confident
    """,
    version="1.0.0",
    contact={
        "name": "Tahir",
        "email": "md.a.tahir@gmail.com",
    },
    docs_url="/",  # Swagger UI directly at root
)

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

@app.get("/health", tags=["Health Check"])
def health():
    """
    ✅ Check if the API is running.
    """
    return {"status": "✅ API is running", "version": "1.0.0"}

@app.post("/predict", tags=["Prediction"])
def predict(data: UserInput):
    """
    🔷 Predict vaccine hesitancy based on user inputs.
    """
    try:
        row = [[
            data.age,
            le_sex.transform([data.sex.capitalize()])[0],
            le_marital.transform([data.marital_status.capitalize()])[0],
            le_place.transform([data.place.capitalize()])[0],
            data.edu,
            data.qua,
            data.job,
            data.jobst,
            data.child
        ]]
        pred = clf.predict(row)[0]
        result = "Vaccine Hesitant" if pred == 1 else "Vaccine Confident"
        return {
            "prediction": result,
            "input": data.dict()
        }
    except Exception as e:
        return {"error": str(e)}
