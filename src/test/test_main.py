import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Sentiment Analysis API"}

def test_predict_sentiment():
    response = client.post(
        "/predict_sentiment",
        json={"review": "I love this product!"}
    )
    assert response.status_code == 200
    assert "sentiment" in response.json()

def test_predict_sentiment_gui():
    response = client.post(
        "/predict_sentiment_gui",
        data={"review": "I hate this product!"}
    )
    assert response.status_code == 200
    assert "sentiment" in response.json()
