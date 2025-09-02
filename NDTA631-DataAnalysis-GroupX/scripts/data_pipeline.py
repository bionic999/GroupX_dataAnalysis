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
        
       
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        
      
        df = df.drop_duplicates()
        logging.info(f"Cleaned {dataset_name}, final shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error cleaning {dataset_name}: {e}")
        return None
    
def filter_south_africa_data(df, country_column, dataset_name):
    """Filter the dataset to include only rows where the country is South Africa."""
    try:
        if df is None:
            raise ValueError(f"No data to filter for {dataset_name}")
        if country_column not in df.columns:
            raise ValueError(f"Column '{country_column}' not found in {dataset_name}")
        
        logging.info(f"Filtering {dataset_name} for South Africa data")
        df_filtered = df[df[country_column].str.strip().str.lower() == "south africa"]
        logging.info(f"Filtered {dataset_name}, resulting shape: {df_filtered.shape}")
        return df_filtered
    except Exception as e:
        logging.error(f"Error filtering {dataset_name}: {e}")
        return None        
    
    
def store_in_database(df1, df2, db_name=r"C:\Users\Admin\Desktop\GroupX_DataAnalysis\NDTA631-DataAnalysis-GroupX\scripts\database.db"):
    """Store datasets in an SQLite database."""
    try:
        conn = sqlite3.connect(db_name)
        df1.to_sql('employment', conn, if_exists='replace', index=False)
        df2.to_sql('poverty', conn, if_exists='replace', index=False)
        logging.info("Datasets stored in SQLite database")
        conn.close()
    except Exception as e:
        logging.error(f"Error storing data in database: {e}")

def main():
    """Main function to run the data pipeline."""
    
    dataset1_path = r"C:\Users\Admin\Desktop\GroupX_DataAnalysis\NDTA631-DataAnalysis-GroupX\data\raw\employment.csv "
    dataset2_path = r"C:\Users\Admin\Desktop\GroupX_DataAnalysis\NDTA631-DataAnalysis-GroupX\data\raw\poverty.csv"
    
    
    df1 = load_data(dataset1_path, "employment")
    df2 = load_data(dataset2_path, "poverty")
    
    if df1 is None or df2 is None:
        logging.error("Failed to load datasets. Check paths and try again.")
        return
    
    
    df1_clean = clean_data(df1, "employment")
    df2_clean = clean_data(df2, "poverty")
    
    if df1_clean is None or df2_clean is None:
        logging.error("Failed to clean datasets.")
        return
    
        # Filter for South African data
    df1_filtered = filter_south_africa_data(df1_clean, "REF_AREA_LABEL", "employment")
    df2_filtered = filter_south_africa_data(df2_clean, "REF_AREA_LABEL", "poverty")
    
    if df1_filtered is None or df2_filtered is None:
        logging.error("Failed to filter datasets for South Africa.")
        return
    
    
    
    store_in_database(df1_filtered, df2_filtered)
    
    
    df1_filtered.to_csv(r"C:\Users\Admin\Desktop\GroupX_DataAnalysis\NDTA631-DataAnalysis-GroupX\data\processed\employment_sa_clean.csv", index=False)
    df2_filtered.to_csv(r"C:\Users\Admin\Desktop\GroupX_DataAnalysis\NDTA631-DataAnalysis-GroupX\data\processed\poverty_sa_clean.csv", index=False)
    logging.info("Cleaned and filtered datasets saved")

if __name__ == "__main__":
    main()      