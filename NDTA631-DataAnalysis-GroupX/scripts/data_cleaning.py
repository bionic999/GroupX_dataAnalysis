import pandas as pd
import numpy as np
import os
import sqlite3
from datetime import datetime

class SouthAfricaDataCleaner:
    """
    Here we handle data cleaning operations for South Africa employment and poverty data.
    """
    
    def __init__(self, raw_data_path="../data/raw/", processed_data_path="../data/processed/"):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.employment_data = None
        self.poverty_data = None
        
        # Create processed data directory if it doesn't exist
        os.makedirs(self.processed_data_path, exist_ok=True)
    
    def load_raw_data(self):
        try:
            print("Loading raw datasets...")
            
            # Load employment data
            employment_file = os.path.join(self.raw_data_path, "employment.csv")
            self.employment_raw = pd.read_csv(employment_file)
            print(f"Employment data loaded: {self.employment_raw.shape}")
            
            # Load poverty data
            poverty_file = os.path.join(self.raw_data_path, "poverty.csv")
            self.poverty_raw = pd.read_csv(poverty_file)
            print(f"Poverty data loaded: {self.poverty_raw.shape}")
            
        except FileNotFoundError as e:
            print(f"Error loading data files: {e}")
            return False
        return True
    
    def filter_south_africa_data(self):
        """Extract South Africa specific data from both datasets."""
        print("Filtering South Africa data...")
        
        # Filter employment data for South Africa
        sa_employment = self.employment_raw[
            self.employment_raw['REF_AREA_LABEL'] == 'South Africa'
        ].copy()
        
        # Filter poverty data for South Africa  
        sa_poverty = self.poverty_raw[
            self.poverty_raw['REF_AREA_LABEL'] == 'South Africa'
        ].copy()
        
        print(f"South Africa employment records: {len(sa_employment)}")
        print(f"South Africa poverty records: {len(sa_poverty)}")
        
        return sa_employment, sa_poverty
    
    def clean_employment_data(self, sa_employment):
        """Clean the data."""
        print("Cleaning employment data...")
        
        # Get year columns (1960-2023)
        year_cols = [col for col in sa_employment.columns if col.isdigit()]
        
        # Create a clean employment dataset
        employment_clean = []
        
        for _, row in sa_employment.iterrows():
            for year in year_cols:
                value = row[year]
                if pd.notna(value) and value != '':
                    try:
                        employment_clean.append({
                            'year': int(year),
                            'employment_rate': float(value),
                            'country': row['REF_AREA_LABEL'],
                            'indicator': row['INDICATOR_LABEL']
                        })
                    except (ValueError, TypeError):
                        continue
        
        self.employment_data = pd.DataFrame(employment_clean)
        print(f"Clean employment data: {len(self.employment_data)} records")
        
        return self.employment_data
    
    def clean_poverty_data(self, sa_poverty):
        """Clean poverty data."""
        print("Cleaning poverty data...")
        
        poverty_clean = []
        
        for _, row in sa_poverty.iterrows():
            year = row['TIME_PERIOD']
            value = row['OBS_VALUE']
            
            if pd.notna(value) and pd.notna(year):
                try:
                    poverty_clean.append({
                        'year': int(year),
                        'poverty_rate': float(value),
                        'country': row['REF_AREA_LABEL'],
                        'indicator': row['INDICATOR_LABEL']
                    })
                except (ValueError, TypeError):
                    continue
        
        self.poverty_data = pd.DataFrame(poverty_clean)
        print(f"Clean poverty data: {len(self.poverty_data)} records")
        
        return self.poverty_data
    
    def merge_datasets(self):
        """Merge employment and poverty datasets on year."""
        print("Merging datasets...")
        
        if self.employment_data is None or self.poverty_data is None:
            print("Error: Clean datasets not available for merging")
            return None
        
        # Merge on year
        merged_data = pd.merge(
            self.employment_data, 
            self.poverty_data, 
            on=['year', 'country'], 
            how='outer'
        )
        
        # Sort by year
        merged_data = merged_data.sort_values('year').reset_index(drop=True)
        
        print(f"Merged dataset: {len(merged_data)} records")
        print(f"Year range: {merged_data['year'].min()} - {merged_data['year'].max()}")
        
        return merged_data
    
    def handle_missing_values(self, data):
        """Handle missing values in the merged dataset."""
        print("Handling missing values...")
        
        # Display missing value statistics
        print("\nMissing value summary:")
        print(data.isnull().sum())
        
        # For time series data, we can use forward fill for small gaps
        data_filled = data.copy()
        data_filled['employment_rate'] = data_filled['employment_rate'].fillna(method='ffill')
        data_filled['poverty_rate'] = data_filled['poverty_rate'].fillna(method='ffill')
        
        # Drop rows where both key metrics are missing
        data_filled = data_filled.dropna(subset=['employment_rate', 'poverty_rate'], how='all')
        
        print(f"Data after handling missing values: {len(data_filled)} records")
        
        return data_filled
    
    def generate_descriptive_statistics(self, data):
        """Generate descriptive statistics for the clean data."""
        print("\n=== DESCRIPTIVE STATISTICS ===")
        
        # Basic statistics
        print("\nBasic Statistics:")
        print(data[['year', 'employment_rate', 'poverty_rate']].describe())
        
        # Data availability
        print(f"\nData Availability:")
        print(f"Employment data points: {data['employment_rate'].notna().sum()}")
        print(f"Poverty data points: {data['poverty_rate'].notna().sum()}")
        print(f"Complete records (both metrics): {data[['employment_rate', 'poverty_rate']].notna().all(axis=1).sum()}")
        
        # Year range analysis
        emp_years = data[data['employment_rate'].notna()]['year']
        pov_years = data[data['poverty_rate'].notna()]['year']
        
        print(f"\nTemporal Coverage:")
        print(f"Employment data: {emp_years.min()} - {emp_years.max()}")
        print(f"Poverty data: {pov_years.min()} - {pov_years.max()}")
        
        return data.describe()
    
    def save_processed_data(self, data, filename="south_africa_clean.csv"):
        """Save the processed data to CSV."""
        output_path = os.path.join(self.processed_data_path, filename)
        data.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")
        return output_path
    
    def run_cleaning_pipeline(self):
        """Run the complete data cleaning pipeline."""
        print("=== STARTING DATA CLEANING PIPELINE ===\n")
        
        # Step 1: Load raw data
        if not self.load_raw_data():
            return None
        
        # Step 2: Filter South Africa data
        sa_employment, sa_poverty = self.filter_south_africa_data()
        
        # Step 3: Clean individual datasets
        employment_clean = self.clean_employment_data(sa_employment)
        poverty_clean = self.clean_poverty_data(sa_poverty)
        
        # Step 4: Merge datasets
        merged_data = self.merge_datasets()
        if merged_data is None:
            return None
        
        # Step 5: Handle missing values
        final_data = self.handle_missing_values(merged_data)
        
        # Step 6: Generate descriptive statistics
        stats = self.generate_descriptive_statistics(final_data)
        
        # Step 7: Save processed data
        output_path = self.save_processed_data(final_data)
        
        print(f"\n=== DATA CLEANING COMPLETED ===")
        print(f"Final dataset shape: {final_data.shape}")
        print(f"Saved to: {output_path}")
        
        return final_data, stats

def main():
    """Main function to run the data cleaning thing"""
    # Initialize the cleaner
    cleaner = SouthAfricaDataCleaner()
    
    # Run the cleaning pipeline
    clean_data, statistics = cleaner.run_cleaning_pipeline()
    
    if clean_data is not None:
        print("\nData cleaning successful!")
        print(f"Clean data available with {len(clean_data)} records")
    else:
        print("Data cleaning failed!")

if __name__ == "__main__":
    main()
