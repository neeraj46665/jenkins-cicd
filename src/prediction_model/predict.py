import os
import pandas as pd
import joblib
from prediction_model.config.config import *
from prediction_model.processing.data_preprocessing import preprocess_data

def predict_review(review):
    # Preprocess the review
    df = pd.DataFrame({'review': [review]})
    df = preprocess_data(df)
    
    # Load the CountVectorizer used during training
    count_vectorizer = joblib.load(COUNT_VECTORIZER)
    
    # Transform the review using the loaded vectorizer
    X = count_vectorizer.transform(df['review'])
    
    # Load trained model (assuming MultinomialNB here)
    model = joblib.load(NAIVE_BAYES_CV_MODEL)  # Adjust to your trained model path
    
    # Make prediction
    y_pred = model.predict(X)
    
    return y_pred[0]

if __name__ == "__main__":
    review = "This is a great movie!"
    prediction = predict_review(review)
    print(f"Prediction: {prediction}")
