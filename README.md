Fake News Detector

A web application built with FastAPI that classifies news headlines as Real or Fake.  
The machine learning model is trained with scikit-learn.


- Input a news headline and get a prediction
- Classification: Fake or Real
- REST API for integration with other applications
- Interactive API documentation via Swagger UI

Tech Stack
- Python 3.10+
- [FastAPI](https://fastapi.tiangolo.com/)
- [scikit-learn](https://scikit-learn.org/)
- [uvicorn](https://www.uvicorn.org/) — ASGI server

 
Run: uvicorn backend.API.main:app --reload
