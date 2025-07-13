from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib

app = FastAPI()
templates = Jinja2Templates(directory="templates")

clf = joblib.load("vaccine_model.pkl")
le_sex = joblib.load("le_sex.pkl")
le_marital = joblib.load("le_marital.pkl")
le_place = joblib.load("le_place.pkl")

@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request,
            age: int = Form(...),
            sex: str = Form(...),
            marital_status: str = Form(...),
            place: str = Form(...),
            edu: int = Form(...),
            job: int = Form(...),
            jobst: int = Form(...),
            child: int = Form(...)):
    try:
        row = [[age,
                le_sex.transform([sex])[0],
                le_marital.transform([marital_status])[0],
                le_place.transform([place])[0],
                edu, job, jobst, child]]
        pred = clf.predict(row)[0]
        if pred == 1:
            result = "Vaccine Hesitant"
            message = ("⚠️ Vaccine prevents life-threatening diseases. "
                       "Your hesitancy is dangerous for your loved ones. "
                       "Kindly vaccinate your child or visit the nearest healthcare center for more information.")
        else:
            result = "Vaccine Compliant"
            message = ("✅ Thanks for being compliant. You are helping society live a healthier life by protecting "
                       "your kids and loved ones.")
        return templates.TemplateResponse("index.html",
                                          {"request": request, "result": result, "message": message})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse("index.html",
                                          {"request": request, "result": "Error", "message": str(e)})
