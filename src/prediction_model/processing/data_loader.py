# processing/data_loader.py
import pandas as pd
from prediction_model.config.config import *

def load_data(file_path):
    """
    Load dataset from CSV file.
    
    Parameters:
    - file_path (str): Path to the CSV file.
    
    Returns:
    - pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(file_path)
    return df
