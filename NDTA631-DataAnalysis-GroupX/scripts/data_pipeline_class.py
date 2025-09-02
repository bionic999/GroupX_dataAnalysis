import pandas as pd
import numpy as np
import sqlite3
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SouthAfricaDataCleaner:
    def __init__(self, dataset1_path, dataset2_path, db_name, output_dir):
        self.dataset1_path = Path(dataset1_path.strip())
        self.dataset2_path = Path(dataset2_path.strip())
        self.db_name = Path(db_name.strip())
        self.output_dir = Path(output_dir.strip())

    def load_data(self, file_path, dataset_name):
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

    def clean_data(self, df, dataset_name):
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

    def filter_south_africa_data(self, df, country_column, dataset_name):
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

    def store_in_database(self, df1, df2):
        """Store datasets in an SQLite database."""
        try:
            conn = sqlite3.connect(self.db_name)
            df1.to_sql('employment', conn, if_exists='replace', index=False)
            df2.to_sql('poverty', conn, if_exists='replace', index=False)
            logging.info("Datasets stored in SQLite database")
            conn.close()
        except Exception as e:
            logging.error(f"Error storing data in database: {e}")

    def run(self):
        """Run the complete data pipeline."""
        df1 = self.load_data(self.dataset1_path, "employment")
        df2 = self.load_data(self.dataset2_path, "poverty")

        if df1 is None or df2 is None:
            logging.error("Failed to load datasets. Check paths and try again.")
            return

        df1_clean = self.clean_data(df1, "employment")
        df2_clean = self.clean_data(df2, "poverty")

        if df1_clean is None or df2_clean is None:
            logging.error("Failed to clean datasets.")
            return

        df1_filtered = self.filter_south_africa_data(df1_clean, "REF_AREA_LABEL", "employment")
        df2_filtered = self.filter_south_africa_data(df2_clean, "REF_AREA_LABEL", "poverty")

        if df1_filtered is None or df2_filtered is None:
            logging.error("Failed to filter datasets for South Africa.")
            return

        self.store_in_database(df1_filtered, df2_filtered)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        df1_filtered.to_csv(self.output_dir / "employment_sa_clean.csv", index=False)
        df2_filtered.to_csv(self.output_dir / "poverty_sa_clean.csv", index=False)
        logging.info("Cleaned and filtered datasets saved")


if __name__ == "__main__":
    cleaner = SouthAfricaDataCleaner(
        dataset1_path=r"C:\Users\Admin\Desktop\GroupX_DataAnalysis\NDTA631-DataAnalysis-GroupX\data\raw\employment.csv",
        dataset2_path=r"C:\Users\Admin\Desktop\GroupX_DataAnalysis\NDTA631-DataAnalysis-GroupX\data\raw\poverty.csv",
        db_name=r"C:\Users\Admin\Desktop\GroupX_DataAnalysis\NDTA631-DataAnalysis-GroupX\scripts\database.db",
        output_dir=r"C:\Users\Admin\Desktop\GroupX_DataAnalysis\NDTA631-DataAnalysis-GroupX\data\processed"
    )
    cleaner.run()
