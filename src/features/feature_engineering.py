import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml
import logging
from logging.handlers import RotatingFileHandler
from sklearn.feature_extraction.text import CountVectorizer


# >>>>>>>  Set up logging <<<<<
def setup_logging(log_file='logs/feature_engineering.log'):
    """
    Configure logging with both console and file handlers.
    
    Args:
        log_file (str): Path to the log file
    """
    # Create logger
    logger = logging.getLogger('feature_engineering')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
    
    # Create console handler and set level to debug
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # Create rotating file handler (max 5MB per file, keep 3 backup files)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter and add it to handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# Initialize logger
logger = setup_logging()

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        logger.info(f"Loading parameters from {params_path}")
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f'Parameters loaded: {list(params.keys())}')
        return params
    except FileNotFoundError as e:
        logger.error(f'File not found: {params_path}', exc_info=True)
        raise
    except yaml.YAMLError as e:
        logger.error(f'YAML error while parsing {params_path}: {e}', exc_info=True)
        raise
    except Exception as e:
        logger.error(f'Unexpected error loading parameters: {e}', exc_info=True)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.debug(f"Data shape before NaN filling: {df.shape}")
        df.fillna('', inplace=True)
        logger.info(f'Data loaded successfully. Shape: {df.shape}, Columns: {df.columns.tolist()}')
        return df
    except pd.errors.ParserError as e:
        logger.error(f'Failed to parse the CSV file {file_path}: {e}', exc_info=True)
        raise
    except Exception as e:
        logger.error(f'Unexpected error occurred while loading data from {file_path}: {e}', exc_info=True)
        raise

def apply_bow(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """Apply Bag of Words to the data."""
    try:
        logger.info(f"Applying Bag of Words vectorization with max_features={max_features}")
        vectorizer = CountVectorizer(max_features=max_features)

        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values
        
        logger.debug(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        logger.debug(f"Bag of Words vectorization completed. Train matrix shape: {X_train_bow.shape}, Test matrix shape: {X_test_bow.shape}")

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        logger.info(f'Bag of Words transformation completed. Train DF shape: {train_df.shape}, Test DF shape: {test_df.shape}')
        return train_df, test_df
    except Exception as e:
        logger.error(f'Error during Bag of Words transformation: {e}', exc_info=True)
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        logger.info(f"Saving data to {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        logger.debug(f"Directory created/verified: {os.path.dirname(file_path)}")
        df.to_csv(file_path, index=False)
        logger.info(f'Data saved successfully to {file_path} (shape: {df.shape})')
    except Exception as e:
        logger.error(f'Unexpected error occurred while saving data to {file_path}: {e}', exc_info=True)
        raise

def main():
    try:
        logger.info("="*50)
        logger.info("Starting feature engineering pipeline")
        logger.info("="*50)
        
        logger.info("Loading parameters")
        params = load_params('params.yaml')
        max_features = params['feature_engineering']['max_features']
        logger.info(f"Max features parameter: {max_features}")

        logger.info("Loading training and test data")
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        logger.info("Applying Bag of Words vectorization")
        train_df, test_df = apply_bow(train_data, test_data, max_features)

        logger.info("Saving processed data")
        save_data(train_df, os.path.join("./data", "processed", "train_bow.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_bow.csv"))
        
        logger.info("="*50)
        logger.info("✅ Feature engineering pipeline completed successfully")
        logger.info("="*50)
        
    except Exception as e:
        logger.error("="*50)
        logger.error(f'❌ Feature engineering pipeline failed: {e}', exc_info=True)
        logger.error("="*50)
        raise

if __name__ == '__main__':
    main()