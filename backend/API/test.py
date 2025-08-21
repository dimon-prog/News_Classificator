from importlib.resources import files
import joblib
from backend import model

m = files(model).joinpath("model.pkl")
v = files(model).joinpath("vectorizer.pkl")

with m.open("rb") as f:
    model = joblib.load(f)

with v.open("rb") as f_2:
    vectorizer = joblib.load(f_2)
