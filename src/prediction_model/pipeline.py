# pipeline.py
import os
import joblib
from config.config import *
from prediction_model.processing.data_loader import load_data
from prediction_model.processing.data_preprocessing import preprocess_data

def main():
    # Load dataset
    df = load_data(DATASET_PATH)
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Vectorize data using CountVectorizer
    count_vectorizer = joblib.load(COUNT_VECTORIZER)
    X = count_vectorizer.transform(df['review'])
    
    # Load trained model
    model = joblib.load(LOGISTIC_REGRESSION_CV_MODEL)
    
    # Make predictions
    y_pred = model.predict(X)
    print(y_pred)

if __name__ == "__main__":
    main()
