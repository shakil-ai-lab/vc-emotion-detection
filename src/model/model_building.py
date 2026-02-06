import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml
import logging
from logging.handlers import RotatingFileHandler


# >>>>>>>  Set up logging <<<<<
def setup_logging(log_file='logs/model_building.log'):
    """
    Configure logging with both console and file handlers.
    
    Args:
        log_file (str): Path to the log file
    """
    # Create logger
    logger = logging.getLogger('model_building')
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
        logger.info(f"Loading model parameters from {params_path}")
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
        logger.info(f'Data loaded successfully. Shape: {df.shape}')
        logger.debug(f'Columns: {df.columns.tolist()}')
        return df
    except pd.errors.ParserError as e:
        logger.error(f'Failed to parse the CSV file {file_path}: {e}', exc_info=True)
        raise
    except Exception as e:
        logger.error(f'Unexpected error occurred while loading data from {file_path}: {e}', exc_info=True)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> GradientBoostingClassifier:
    """Train the Gradient Boosting model."""
    try:
        logger.info(f"Training Gradient Boosting model with parameters: n_estimators={params['n_estimators']}, learning_rate={params['learning_rate']}")
        logger.debug(f"Training data shape: {X_train.shape}, Target shape: {y_train.shape}")
        
        clf = GradientBoostingClassifier(
            n_estimators=params['n_estimators'], 
            learning_rate=params['learning_rate'],
            random_state=42
        )
        clf.fit(X_train, y_train)
        
        # Calculate training accuracy
        train_accuracy = accuracy_score(y_train, clf.predict(X_train))
        logger.info(f'Model training completed. Training accuracy: {train_accuracy:.4f}')
        logger.debug(f'Model parameters: {clf.get_params()}')
        
        return clf
    except Exception as e:
        logger.error(f'Error during model training: {e}', exc_info=True)
        raise

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        logger.info(f"Saving model to {file_path}")
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        logger.debug(f"Directory created/verified: {os.path.dirname(file_path)}")
        
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.info(f'Model saved successfully to {file_path}')
    except Exception as e:
        logger.error(f'Error occurred while saving the model to {file_path}: {e}', exc_info=True)
        raise

def main():
    try:
        logger.info("="*50)
        logger.info("Starting model building pipeline")
        logger.info("="*50)
        
        logger.info("Loading model parameters")
        params = load_params('params.yaml')['model_building']
        logger.debug(f"Model parameters: {params}")

        logger.info("Loading training data")
        train_data = load_data('./data/processed/train_tfidf.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        logger.info(f"Training data extracted. Features: {X_train.shape[1]}, Samples: {X_train.shape[0]}")

        logger.info("Starting model training")
        clf = train_model(X_train, y_train, params)
        
        logger.info("Saving trained model")
        save_model(clf, 'models/model.pkl')
        
        logger.info("="*50)
        logger.info("✅ Model building pipeline completed successfully")
        logger.info("="*50)
        
    except Exception as e:
        logger.error("="*50)
        logger.error(f'❌ Model building pipeline failed: {e}', exc_info=True)
        logger.error("="*50)
        raise

if __name__ == '__main__':
    main()