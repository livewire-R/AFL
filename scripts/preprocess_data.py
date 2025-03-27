#!/usr/bin/env python3
"""
AFL Player Statistics Data Preprocessing Script

This script preprocesses the raw AFL player statistics data downloaded from wheeloratings.com.
It cleans the data, merges files from different years, and prepares it for machine learning.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = r'C:\Users\ralph\OneDrive\Desktop\AFL'
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Ensure processed data directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def load_historical_data():
    """
    Load historical player statistics data (2020-2024)
    """
    historical_dir = os.path.join(RAW_DATA_DIR, 'historical')
    dfs = []
    
    for year in range(2020, 2026):
        filename = f'afl-player-stats-{year}.csv'
        if year == 2024:
            filename = 'afl-player-stats-2024 (1).csv'
            
        file_path = os.path.join(historical_dir, filename)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Year'] = year
            dfs.append(df)
            logger.info(f"Loaded data for year {year}")
        else:
            logger.warning(f"No data file found for year {year}")
    
    if not dfs:
        raise ValueError("No historical data found")
    
    return pd.concat(dfs, ignore_index=True)

def load_current_data():
    """
    Load current season player statistics
    """
    current_dir = os.path.join(RAW_DATA_DIR, 'current')
    current_file = os.path.join(current_dir, 'afl-player-stats-2025.csv')
    
    if os.path.exists(current_file):
        df = pd.read_csv(current_file)
        df['Year'] = 2025
        logger.info("Loaded current season data")
        return df
    else:
        logger.warning("No current season data found")
        return None

def clean_player_stats_data(df):
    """
    Clean and preprocess player statistics data
    """
    # Convert numeric columns
    numeric_cols = ['Games', 'Disposals', 'Goals', 'Behinds', 'Marks', 'Tackles']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate averages
    if 'Games' in df.columns and 'Disposals' in df.columns:
        df['Avg_Disposals'] = df['Disposals'] / df['Games']
    if 'Games' in df.columns and 'Goals' in df.columns:
        df['Avg_Goals'] = df['Goals'] / df['Games']
    
    # Fill missing values
    df = df.fillna(0)
    
    return df

def process_data():
    """
    Main function to process all data
    """
    logger.info("Starting AFL player statistics data preprocessing...")
    
    try:
        # Load and merge historical and current data
        historical_df = load_historical_data()
        current_df = load_current_data()
        
        if current_df is not None:
            df = pd.concat([historical_df, current_df], ignore_index=True)
        else:
            df = historical_df
        
        # Clean and process the data
        df = clean_player_stats_data(df)
        
        # Save processed data
        output_file = os.path.join(PROCESSED_DATA_DIR, 'processed_player_stats.csv')
        df.to_csv(output_file, index=False)
        logger.info(f"Processed data saved to {output_file}")
        
        # Save as pickle for faster loading
        pickle_file = os.path.join(PROCESSED_DATA_DIR, 'processed_player_stats.pkl')
        df.to_pickle(pickle_file)
        logger.info(f"Processed data saved to {pickle_file}")
        
        return df
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {str(e)}")
        raise

if __name__ == "__main__":
    process_data()
