
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report


import mlflow

from prediction_model.config.config import *
from prediction_model.processing.data_loader import load_data
from prediction_model.processing.data_preprocessing import preprocess_data
from prediction_model.models.logistic_regression import train_logistic_regression
from prediction_model.models.naive_bayes import train_naive_bayes
from prediction_model.models.random_forest import train_random_forest
from prediction_model.ml.mlflow_utils import mlflow_logging


def vectorize_data(X_train, X_test, vectorizer):
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec



def save_model(model, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)


def main():
    # Load dataset
    df = load_data(DATASET_PATH)
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['class'], test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Vectorize data using CountVectorizer and TF-IDF
    count_vectorizer = CountVectorizer()
    X_train_cv, X_test_cv = vectorize_data(X_train, X_test, count_vectorizer)
    
    
    
    # Initialize MLflow run
    mlflow.set_experiment("sentiment prediction")
    # with mlflow.start_run():
    #     mlflow.log_param("test_size", TEST_SIZE)
    #     mlflow.log_param("random_state", RANDOM_STATE)
        
    # Logistic Regression
    print("Logistic Regression with CountVectorizer")
    model, _, _ = train_logistic_regression(X_train_cv, y_train, X_test_cv, y_test)
    mlflow_logging(model, X_test_cv, y_test, "LogisticRegression_CountVectorizer")
    save_model(model, LOGISTIC_REGRESSION_CV_MODEL)
    save_model(count_vectorizer, COUNT_VECTORIZER)
    

    
    # Naive Bayes
    print("Naive Bayes with CountVectorizer")
    model, _, _ = train_naive_bayes(X_train_cv, y_train, X_test_cv, y_test)
    mlflow_logging(model, X_test_cv, y_test, "NaiveBayes_CountVectorizer")
    save_model(model, NAIVE_BAYES_CV_MODEL)
    

    
    # Random Forest
    print("Random Forest with CountVectorizer")
    model, _, _ = train_random_forest(X_train_cv, y_train, X_test_cv, y_test)
    mlflow_logging(model, X_test_cv, y_test, "RandomForest_CountVectorizer")
    save_model(model, RANDOM_FOREST_CV_MODEL)
        


if __name__ == "__main__":
    main()

