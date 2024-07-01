# config/config.py

import os

# Paths
# config.py


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


DATASET_PATH = os.path.join(BASE_DIR, '..', 'datasets', 'TestReviews.csv')

TRAINED_MODELS_DIR = 'src\\prediction_model\\trained_models'

# Model names
# Model names
LOGISTIC_REGRESSION_CV_MODEL = os.path.join(TRAINED_MODELS_DIR, 'logistic_regression_count_vectorizer.pkl')

NAIVE_BAYES_CV_MODEL = os.path.join(TRAINED_MODELS_DIR, 'naive_bayes_count_vectorizer.pkl')

RANDOM_FOREST_CV_MODEL = os.path.join(TRAINED_MODELS_DIR, 'random_forest_count_vectorizer.pkl')


# Vectorizers
COUNT_VECTORIZER = os.path.join(TRAINED_MODELS_DIR, 'count_vectorizer.pkl')





# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
LOGISTIC_REGRESSION_MAX_ITER = 1000
RANDOM_FOREST_ESTIMATORS = 100
