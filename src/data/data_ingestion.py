import numpy as np
import pandas as pd
import os
import yaml
import logging
from logging.handlers import RotatingFileHandler

from sklearn.model_selection import train_test_split


# >>>>>>>  Set up logging <<<<<
def setup_logging(log_file='logs/data_ingestion.log'):
    """
    Configure logging with both console and file handlers.
    
    Args:
        log_file (str): Path to the log file
    """
    # Create logger
    logger = logging.getLogger("data_ingestion")
    logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs
    
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

# to connect params.yaml
def load_params(param_path='params.yaml'):
    try:
        logger.info(f"Loading parameters from {param_path}")
        with open(param_path, 'r') as file:
            params = yaml.safe_load(file)
        
        # Validate that params is not None or empty
        if params is None:
            logger.error(f"params.yaml is empty or contains no valid YAML content")
            raise ValueError(f"params.yaml is empty or contains no valid YAML content at {param_path}")
        
        logger.debug(f"Raw params loaded: {params}")
        test_size = params['data_ingestion']['test_size']
        logger.debug(f"Successfully loaded test_size: {test_size}")
        return test_size

    except FileNotFoundError:
        logger.error(f"params file not found at path: {param_path}", exc_info=True)
        raise FileNotFoundError(f"params file not found at path: {param_path}")

    except KeyError as e:
        logger.error(f"Missing required key in params.yaml: {e}", exc_info=True)
        raise KeyError(f"Missing required key in params.yaml: {e}")

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}", exc_info=True)
        raise ValueError(f"Error parsing YAML file: {e}")


# function to read data from url
def read_data(url):
    try:
        logger.info(f"Reading data from URL: {url}")
        df = pd.read_csv(url)
        logger.info(f"Successfully read data. Shape: {df.shape}")
        logger.debug(f"Columns: {df.columns.tolist()}")
        return df

    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file from the given URL: {e}", exc_info=True)
        raise ValueError("Error parsing CSV file from the given URL")

    except Exception as e:
        logger.error(f"Failed to read data from URL: {e}", exc_info=True)
        raise RuntimeError(f"Failed to read data from URL: {e}")


# function to process data
def process_data(df):
    try:
        logger.info("Starting data processing")
        logger.debug(f"Initial dataframe shape: {df.shape}")
        
        # drop column > tweet_id
        logger.debug("Dropping 'tweet_id' column")
        final_df = df.drop(columns=['tweet_id'])

        # filter required sentiments
        logger.debug("Filtering for 'sadness' and 'happiness' sentiments")
        final_df = final_df[final_df['sentiment'].isin(['sadness', 'happiness'])]
        logger.info(f"After sentiment filtering, shape: {final_df.shape}")

        # encode sentiment
        logger.debug("Encoding sentiment values (sadness=0, happiness=1)")
        final_df['sentiment'] = final_df['sentiment'].replace({
            "sadness": 0,
            "happiness": 1
        })

        # fill NaN values
        logger.debug("Filling NaN values with empty strings")
        final_df = final_df.fillna("")
        
        logger.info(f"Data processing completed. Final shape: {final_df.shape}")
        return final_df

    except KeyError as e:
        logger.error(f"Required column missing in dataframe: {e}", exc_info=True)
        raise KeyError(f"Required column missing in dataframe: {e}")

    except Exception as e:
        logger.error(f"Error during data processing: {e}", exc_info=True)
        raise RuntimeError(f"Error during data processing: {e}")


# save data to train and test files
def save_data(train_data, test_data):
    try:
        logger.info("Starting to save train and test data")
        data_path = os.path.join('data', 'raw')
        logger.debug(f"Creating directory: {data_path}")
        os.makedirs(data_path, exist_ok=True)

        train_file = os.path.join(data_path, 'train.csv')
        test_file = os.path.join(data_path, 'test.csv')
        
        logger.info(f"Saving training data to {train_file} (shape: {train_data.shape})")
        train_data.to_csv(train_file, index=False)
        
        logger.info(f"Saving test data to {test_file} (shape: {test_data.shape})")
        test_data.to_csv(test_file, index=False)
        
        logger.info("Train and test data saved successfully")

    except PermissionError as e:
        logger.error(f"Permission denied while saving data files: {e}", exc_info=True)
        raise PermissionError("Permission denied while saving data files")

    except Exception as e:
        logger.error(f"Failed to save train/test data: {e}", exc_info=True)
        raise RuntimeError(f"Failed to save train/test data: {e}")


def main():
    try:
        logger.info("="*50)
        logger.info("Starting data ingestion pipeline")
        logger.info("="*50)
        
        test_size = load_params()
        logger.info(f"Test size parameter: {test_size}")

        url = "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"
        logger.info("Fetching data from remote URL")
        df = read_data(url)

        logger.info("Processing data")
        df = process_data(df)

        logger.info(f"Splitting data with test_size={test_size}")
        train_data, test_data = train_test_split(
            df,
            test_size=test_size,
            random_state=42
        )
        logger.info(f"Train set size: {train_data.shape[0]}, Test set size: {test_data.shape[0]}")

        save_data(train_data, test_data)

        logger.info("="*50)
        logger.info("✅ Data ingestion pipeline completed successfully")
        logger.info("="*50)

    except Exception as e:
        logger.error("="*50)
        logger.error(f"❌ Data ingestion pipeline failed: {e}", exc_info=True)
        logger.error("="*50)
        raise  # re-raise for logging systems / CI-CD failure detection


if __name__ == "__main__":
    main()
