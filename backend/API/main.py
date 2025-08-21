from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from importlib.resources import files
import joblib
from backend import model
from fastapi.middleware.cors import CORSMiddleware

m = files(model).joinpath("model.pkl")
v = files(model).joinpath("vectorizer.pkl")

with m.open("rb") as f:
    model = joblib.load(f)

with v.open("rb") as f_2:
    vectorizer = joblib.load(f_2)

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"], )


class SentenceRequest(BaseModel):
    sentence: str


@app.get("/")
def read_index():
    return FileResponse("frontend/index.html")


app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.post("/predict")
async def predict(req: SentenceRequest):
    sentence_vec = vectorizer.transform([req.sentence])
    prediction = "Real" if model.predict_proba(sentence_vec)[0][0] < 0.5 else "Fake"
    return {"prediction": prediction}
