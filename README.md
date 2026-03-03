Fake News Detector
# News Classificator

A web application built with FastAPI that classifies news headlines as Real or Fake.  
The machine learning model is trained with scikit-learn.
A simple full-stack fake news classification project built with **FastAPI** and **scikit-learn**.

The app takes a news headline and predicts whether it is **Real** or **Fake** using a pre-trained ensemble model.

- Input a news headline and get a prediction
- Classification: Fake or Real
- REST API for integration with other applications
- Interactive API documentation via Swagger UI
## Features

- FastAPI backend with a JSON prediction endpoint
- Lightweight web UI (HTML/CSS/JavaScript)
- Pre-trained TF-IDF vectorizer and classification model included in the repository
- Swagger/OpenAPI documentation out of the box

## Project Structure

```text
News_Classificator/
├── backend/
│   ├── API/
│   │   └── main.py              # FastAPI app and /predict endpoint
│   └── model/
│       ├── model.py             # Training script
│       ├── model.pkl            # Trained classifier
│       ├── vectorizer.pkl       # Trained TF-IDF vectorizer
│       └── data/                # Source datasets
├── frontend/
│   ├── index.html               # Web page
│   ├── css/style.css            # Styles
│   └── js/script.js             # Browser-side request logic
└── requirements.txt
```

## Requirements

Tech Stack
- Python 3.10+
- [FastAPI](https://fastapi.tiangolo.com/)
- [scikit-learn](https://scikit-learn.org/)
- [uvicorn](https://www.uvicorn.org/) — ASGI server
- `pip`

## Installation

```bash
git clone <your-repo-url>
cd News_Classificator
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Application

From the repository root:

```bash
uvicorn backend.API.main:app --reload
```

Then open:

- Web UI: http://127.0.0.1:8000/
- API docs (Swagger): http://127.0.0.1:8000/docs

## API Usage

### `POST /predict`

Predict whether a headline is real or fake.

**Request body**

```json
{
  "sentence": "Government announces new economic reform package"
}
```

**Response**

```json
{
  "prediction": "Real"
}
```

## Model Training (Optional)

If you want to retrain the model with the provided datasets:

```bash
cd backend/model
python model.py
```

> Note: `model.py` currently prints metrics and saves `vectorizer.pkl`. If you want to overwrite the model artifact, ensure model saving is enabled in the script.

## Notes
Run: uvicorn backend.API.main:app --reload
- The frontend currently sends requests to `http://localhost:8000/predict`.
- CORS is enabled for all origins in the FastAPI app.
