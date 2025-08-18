import pandas as pd
import numpy as np
import sqlite3
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path, dataset_name):
    """Load a CSV file and return a DataFrame."""
    try:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"{dataset_name} not found at {file_path}")
        df = pd.read_csv(file_path)
        logging.info(f"Loaded {dataset_name} with {df.shape[0]} rows")
        return df
    except FileNotFoundError as e:
        logging.error(e)
        return None
    except Exception as e:
        logging.error(f"Error loading {dataset_name}: {e}")
        return None
    
def clean_data(df, dataset_name):
    """Clean the dataset by handling missing values and duplicates."""
    try:
        if df is None:
            raise ValueError(f"No data to clean for {dataset_name}")
        logging.info(f"Cleaning {dataset_name}, initial shape: {df.shape}")
        
        # Handle missing values: fill numbers with mean, text with most common value
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # Remove duplicates
        df = df.drop_duplicates()
        logging.info(f"Cleaned {dataset_name}, final shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error cleaning {dataset_name}: {e}")
        return None
    
def store_in_database(df1, df2, db_name=r"C:\Users\Admin\Desktop\GroupX_DataAnalysis\scripts\database.db"):
    """Store datasets in an SQLite database."""
    try:
        conn = sqlite3.connect(db_name)
        df1.to_sql('individuals_internet', conn, if_exists='replace', index=False)
        df2.to_sql('youth_unemployment', conn, if_exists='replace', index=False)
        logging.info("Datasets stored in SQLite database")
        conn.close()
    except Exception as e:
        logging.error(f"Error storing data in database: {e}")

def main():
    """Main function to run the data pipeline."""
    # Paths to datasets (update after downloading from World Bank)
    dataset1_path = r"C:\Users\Admin\Desktop\GroupX_DataAnalysis\data\raw\Individuals_using_the_internet.csv "
    dataset2_path = r"C:\Users\Admin\Desktop\GroupX_DataAnalysis\data\raw\Youth_unemployment.csv"
    
    # Load data
    df1 = load_data(dataset1_path, "Individuals_using_the_internet")
    df2 = load_data(dataset2_path, "Youth_unemployment")
    
    if df1 is None or df2 is None:
        logging.error("Failed to load datasets. Check paths and try again.")
        return
    
    # Clean data
    df1_clean = clean_data(df1, "Individuals_using_the_internet")
    df2_clean = clean_data(df2, "Youth_unemployment")
    
    if df1_clean is None or df2_clean is None:
        logging.error("Failed to clean datasets.")
        return
    
    # Store in database
    store_in_database(df1_clean, df2_clean)
    
    # Save cleaned data
    df1_clean.to_csv(r"C:\Users\Admin\Desktop\GroupX_DataAnalysis\data\processed\Individuals_using_the_internet_clean.csv", index=False)
    df2_clean.to_csv(r"C:\Users\Admin\Desktop\GroupX_DataAnalysis\data\processed\Youth_unemployment_clean.csv", index=False)
    logging.info("Cleaned datasets saved")

if __name__ == "__main__":
    main()      