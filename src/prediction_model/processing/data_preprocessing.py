# processing/data_preprocessing.py
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
from prediction_model.config.config import *

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    """
    Preprocesses text data by removing HTML tags, tokenizing,
    removing stopwords, and stemming.
    
    Parameters:
    - text (str): Input text to preprocess.
    
    Returns:
    - str: Processed text.
    """
    text = re.sub(r'<.*?>', '', text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word.lower() not in stop_words and word.isalnum()]
    return ' '.join(tokens)

def preprocess_data(df):
    """
    Apply preprocessing to all text data in the DataFrame.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing 'review' column.
    
    Returns:
    - pd.DataFrame: Processed DataFrame.
    """
    df['review'] = df['review'].apply(preprocess_text)
    return df
