#!/usr/bin/env python3
"""
AFL Player Statistics Data Preprocessing Script

This script preprocesses the raw AFL player statistics data downloaded from wheeloratings.com.
It cleans the data, merges files from different years, and prepares it for machine learning.
"""

import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime

# Define paths
BASE_DIR = '/home/ubuntu/afl_prediction_project'
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data/raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data/processed')

# Ensure processed data directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def load_player_stats_data(year):
    """
    Load player statistics data for a specific year
    """
    year_dir = os.path.join(RAW_DATA_DIR, str(year))
    csv_files = glob.glob(os.path.join(year_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found for year {year}")
        return None
    
    # Load the first CSV file found for the year
    df = pd.read_csv(csv_files[0])
    df['Year'] = year
    
    print(f"Loaded data for {year}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def clean_player_stats_data(df):
    """
    Clean and preprocess player statistics data
    """
    if df is None or df.empty:
        return None
    
    # Make a copy to avoid modifying the original dataframe
    df_clean = df.copy()
    
    # Convert column names to lowercase and replace spaces with underscores
    df_clean.columns = [col.lower().replace(' ', '_') for col in df_clean.columns]
    
    # Handle missing values
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
    
    # Convert percentage columns to proper decimal values
    pct_cols = [col for col in df_clean.columns if 'pct' in col or '%' in col]
    for col in pct_cols:
        if col in df_clean.columns:
            # Check if values are already in decimal form
            if df_clean[col].max() > 1:
                df_clean[col] = df_clean[col] / 100
    
    # Add a form metric (if possible)
    if 'player' in df_clean.columns and 'gm' in df_clean.columns:
        # Group by player and calculate rolling averages for key stats
        # This will be expanded in the weekly update script
        pass
    
    return df_clean

def merge_all_years_data():
    """
    Merge player statistics data from all years (2020-2025)
    """
    all_data = []
    
    for year in range(2020, 2026):
        df = load_player_stats_data(year)
        if df is not None:
            df_clean = clean_player_stats_data(df)
            if df_clean is not None:
                all_data.append(df_clean)
    
    if not all_data:
        print("No data to merge")
        return None
    
    # Concatenate all dataframes
    merged_df = pd.concat(all_data, ignore_index=True)
    print(f"Merged data: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
    
    return merged_df

def analyze_data(df):
    """
    Perform basic analysis on the merged data
    """
    if df is None or df.empty:
        print("No data to analyze")
        return
    
    print("\n=== Data Analysis ===")
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(f"Total number of records: {df.shape[0]}")
    print(f"Total number of features: {df.shape[1]}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing Values:")
    print(missing_values[missing_values > 0])
    
    # Distribution of players by year
    if 'year' in df.columns:
        year_counts = df['year'].value_counts().sort_index()
        print("\nRecords by Year:")
        print(year_counts)
    
    # Distribution of players by position (if available)
    if 'position' in df.columns:
        position_counts = df['position'].value_counts()
        print("\nPlayers by Position:")
        print(position_counts)
    
    # Key statistics for disposals and goals (our prediction targets)
    target_cols = ['dis.', 'disposals', 'goals', 'gls_avg']
    for col in target_cols:
        if col in df.columns:
            print(f"\nStatistics for {col}:")
            print(df[col].describe())
    
    # Correlation analysis for key metrics
    if 'dis.' in df.columns or 'disposals' in df.columns:
        disposal_col = 'dis.' if 'dis.' in df.columns else 'disposals'
        corr_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        if corr_cols:
            corr_matrix = df[corr_cols].corr()
            print(f"\nTop 10 features correlated with {disposal_col}:")
            print(corr_matrix[disposal_col].sort_values(ascending=False).head(10))
    
    # Save analysis results
    analysis_file = os.path.join(PROCESSED_DATA_DIR, 'data_analysis_summary.txt')
    with open(analysis_file, 'w') as f:
        f.write(f"AFL Player Statistics Data Analysis\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total number of records: {df.shape[0]}\n")
        f.write(f"Total number of features: {df.shape[1]}\n\n")
        
        if 'year' in df.columns:
            f.write("Records by Year:\n")
            for year, count in year_counts.items():
                f.write(f"{year}: {count}\n")
            f.write("\n")
        
        if 'position' in df.columns:
            f.write("Players by Position:\n")
            for position, count in position_counts.items():
                f.write(f"{position}: {count}\n")
            f.write("\n")
        
        for col in target_cols:
            if col in df.columns:
                f.write(f"Statistics for {col}:\n")
                f.write(str(df[col].describe()) + "\n\n")
    
    print(f"\nAnalysis summary saved to {analysis_file}")

def save_processed_data(df):
    """
    Save the processed data to CSV and pickle formats
    """
    if df is None or df.empty:
        print("No data to save")
        return
    
    # Save as CSV
    csv_path = os.path.join(PROCESSED_DATA_DIR, 'afl_player_stats_all_years.csv')
    df.to_csv(csv_path, index=False)
    print(f"Processed data saved to {csv_path}")
    
    # Save as pickle for faster loading in ML scripts
    pickle_path = os.path.join(PROCESSED_DATA_DIR, 'afl_player_stats_all_years.pkl')
    df.to_pickle(pickle_path)
    print(f"Processed data saved to {pickle_path}")

def main():
    """
    Main function to execute the preprocessing pipeline
    """
    print("Starting AFL player statistics data preprocessing...")
    
    # Merge data from all years
    merged_df = merge_all_years_data()
    
    if merged_df is not None:
        # Analyze the merged data
        analyze_data(merged_df)
        
        # Save the processed data
        save_processed_data(merged_df)
        
        print("Data preprocessing completed successfully!")
    else:
        print("Data preprocessing failed: No data to process")

if __name__ == "__main__":
    main()
