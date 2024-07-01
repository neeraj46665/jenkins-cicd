from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from prediction_model.predict import predict_review
from typing import List
import numpy as np
from pydantic import BaseModel
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for predicting sentiment of text",
    version="1.0"
)

# CORS (Cross-Origin Resource Sharing) middleware configuration
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SentimentInput(BaseModel):
    review: str

@app.get("/")
def index():
    return {"message": "Welcome to Sentiment Analysis API"}

@app.post("/predict_sentiment")
def predict_sentiment(input_data: SentimentInput):
    review = input_data.review
    prediction = predict_review(review)
    return {"sentiment": str(prediction)}

@app.post("/predict_sentiment_gui")
def predict_sentiment_gui(review: str = Form(...)):
    prediction = predict_review(review)
    # Convert NumPy int64 to standard Python int
    if isinstance(prediction, np.int64):
        prediction = prediction.item()
    return {"sentiment": prediction}

if __name__ == '__main__':
    import uvicorn
    # uvicorn.run(app, host='127.0.0.1', port=8000)
    uvicorn.run(app, host="0.0.0.0",port=8005)
