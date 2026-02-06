import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
from logging.handlers import RotatingFileHandler


# >>>>>>>  Set up logging <<<<<
def setup_logging(log_file='logs/data_preprocessing.log'):
    """
    Configure logging with both console and file handlers.
    
    Args:
        log_file (str): Path to the log file
    """
    # Create logger
    logger = logging.getLogger('data_preprocessing')
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

nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text):
    """Lemmatize the text."""
    try:
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        text = [lemmatizer.lemmatize(word) for word in text]
        logger.debug(f"Lemmatization applied to text with {len(text)} words")
        return " ".join(text)
    except Exception as e:
        logger.error(f"Error during lemmatization: {e}", exc_info=True)
        raise

def remove_stop_words(text):
    """Remove stop words from the text."""
    try:
        stop_words = set(stopwords.words("english"))
        original_words = len(str(text).split())
        text = [word for word in str(text).split() if word not in stop_words]
        removed_count = original_words - len(text)
        logger.debug(f"Stop words removed: {removed_count} words from {original_words}")
        return " ".join(text)
    except Exception as e:
        logger.error(f"Error removing stop words: {e}", exc_info=True)
        raise

def removing_numbers(text):
    """Remove numbers from the text."""
    try:
        original_length = len(text)
        text = ''.join([char for char in text if not char.isdigit()])
        removed_count = original_length - len(text)
        logger.debug(f"Numbers removed: {removed_count} characters removed")
        return text
    except Exception as e:
        logger.error(f"Error removing numbers: {e}", exc_info=True)
        raise

def lower_case(text):
    """Convert text to lower case."""
    try:
        text = text.split()
        text = [word.lower() for word in text]
        logger.debug(f"Text converted to lower case")
        return " ".join(text)
    except Exception as e:
        logger.error(f"Error converting to lower case: {e}", exc_info=True)
        raise

def removing_punctuations(text):
    """Remove punctuations from the text."""
    try:
        original_length = len(text)
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text).strip()
        logger.debug(f"Punctuations removed: {original_length - len(text)} characters removed")
        return text
    except Exception as e:
        logger.error(f"Error removing punctuations: {e}", exc_info=True)
        raise

def removing_urls(text):
    """Remove URLs from the text."""
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        urls_found = len(url_pattern.findall(text))
        result = url_pattern.sub(r'', text)
        logger.debug(f"URLs removed: {urls_found} URL(s) found and removed")
        return result
    except Exception as e:
        logger.error(f"Error removing URLs: {e}", exc_info=True)
        raise

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    try:
        removed_count = 0
        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:
                df.text.iloc[i] = np.nan
                removed_count += 1
        logger.debug(f"Small sentences removed: {removed_count} sentences with less than 3 words")
        return df
    except Exception as e:
        logger.error(f"Error removing small sentences: {e}", exc_info=True)
        raise

def normalize_text(df):
    """Normalize the text data."""
    try:
        logger.info("Starting text normalization")
        logger.debug(f"Input dataframe shape: {df.shape}")
        
        df['content'] = df['content'].apply(lower_case)
        logger.info('Text converted to lower case')
        
        df['content'] = df['content'].apply(remove_stop_words)
        logger.info('Stop words removed')
        
        df['content'] = df['content'].apply(removing_numbers)
        logger.info('Numbers removed')
        
        df['content'] = df['content'].apply(removing_punctuations)
        logger.info('Punctuations removed')
        
        df['content'] = df['content'].apply(removing_urls)
        logger.info('URLs removed')
        
        df['content'] = df['content'].apply(lemmatization)
        logger.info('Lemmatization performed')
        
        logger.info(f'Text normalization completed. Final shape: {df.shape}')
        return df
    except Exception as e:
        logger.error(f'Error during text normalization: {e}', exc_info=True)
        raise

def main():
    try:
        logger.info("="*50)
        logger.info("Starting data preprocessing pipeline")
        logger.info("="*50)
        
        # Fetch the data from data/raw
        logger.info("Loading training and test data")
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.info(f'Data loaded successfully. Train shape: {train_data.shape}, Test shape: {test_data.shape}')

        # Transform the data
        logger.info('Processing training data')
        train_processed_data = normalize_text(train_data)
        logger.info('Processing test data')
        test_processed_data = normalize_text(test_data)

        # Store the data inside data/processed
        logger.info('Saving processed data')
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        logger.debug(f'Created directory: {data_path}')
        
        train_file = os.path.join(data_path, "train_processed.csv")
        test_file = os.path.join(data_path, "test_processed.csv")
        
        train_processed_data.to_csv(train_file, index=False)
        logger.info(f'Training data saved to {train_file}')
        
        test_processed_data.to_csv(test_file, index=False)
        logger.info(f'Test data saved to {test_file}')
        
        logger.info("="*50)
        logger.info('✅ Data preprocessing pipeline completed successfully')
        logger.info("="*50)
        
    except Exception as e:
        logger.error("="*50)
        logger.error(f'❌ Data preprocessing pipeline failed: {e}', exc_info=True)
        logger.error("="*50)
        raise

if __name__ == '__main__':
    main()