#!/usr/bin/env python3
"""
AFL Prediction - Feature Engineering

This script performs feature engineering on the preprocessed AFL player statistics data
to prepare it for machine learning model training.
"""

import os
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('feature_engineering')

# Define paths
BASE_DIR = '/home/ubuntu/afl_prediction_project'
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data():
    """
    Load the preprocessed player statistics data
    """
    try:
        # Check if processed data exists
        processed_data_path = os.path.join(PROCESSED_DATA_DIR, 'afl_player_stats_all_years.csv')
        
        if not os.path.exists(processed_data_path):
            logger.error(f"Processed data file not found: {processed_data_path}")
            return None
        
        # Load processed data
        data = pd.read_csv(processed_data_path)
        logger.info(f"Loaded processed data: {len(data)} records")
        
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def identify_key_columns(data):
    """
    Identify key columns in the data
    """
    if data is None:
        return None, None, None, None
    
    try:
        # Identify player column
        player_col = None
        for col in ['player', 'player_name', 'name']:
            if col in data.columns:
                player_col = col
                break
        
        # Identify team column
        team_col = None
        for col in ['team', 'club', 'team_name']:
            if col in data.columns:
                team_col = col
                break
        
        # Identify disposal column
        disposal_col = None
        for col in ['dis.', 'disposals', 'disposal', 'total_disposals']:
            if col in data.columns:
                disposal_col = col
                break
        
        # Identify goal column
        goal_col = None
        for col in ['goals', 'gls', 'gls_avg', 'goals_avg', 'total_goals']:
            if col in data.columns:
                goal_col = col
                break
        
        return player_col, team_col, disposal_col, goal_col
    except Exception as e:
        logger.error(f"Error identifying key columns: {e}")
        return None, None, None, None

def engineer_features(data, player_col, team_col, disposal_col, goal_col):
    """
    Engineer features for the machine learning model
    """
    if data is None or player_col is None or team_col is None:
        return None
    
    try:
        logger.info("Engineering features...")
        
        # Create a copy of the data
        df = data.copy()
        
        # Standardize column names
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        player_col = player_col.lower().replace(' ', '_')
        team_col = team_col.lower().replace(' ', '_')
        if disposal_col:
            disposal_col = disposal_col.lower().replace(' ', '_')
        if goal_col:
            goal_col = goal_col.lower().replace(' ', '_')
        
        # Add player experience feature (number of games played)
        if player_col in df.columns:
            logger.info("Adding player experience feature...")
            
            # Sort by player and date/round
            date_col = None
            for col in ['date', 'match_date', 'game_date']:
                if col in df.columns:
                    date_col = col
                    break
            
            round_col = None
            for col in ['round', 'rnd', 'round_number']:
                if col in df.columns:
                    round_col = col
                    break
            
            sort_col = date_col if date_col else round_col
            
            if sort_col:
                df = df.sort_values([player_col, sort_col])
            
            # Calculate cumulative games played for each player
            df['player_experience'] = df.groupby(player_col).cumcount() + 1
            
            logger.info("Player experience feature added")
        
        # Add team form features
        if team_col in df.columns:
            logger.info("Adding team form features...")
            
            # Calculate team average statistics
            team_stats = df.groupby(team_col).agg({
                col: 'mean' for col in df.select_dtypes(include=['float64', 'int64']).columns
                if col != player_col and col != team_col
            }).reset_index()
            
            # Rename columns to indicate team average
            team_stats.columns = [f'team_avg_{col}' if col != team_col else col for col in team_stats.columns]
            
            # Merge team statistics back to the main dataframe
            df = pd.merge(df, team_stats, on=team_col, how='left')
            
            logger.info("Team form features added")
        
        # Add position-specific features
        position_col = None
        for col in ['position', 'pos', 'player_position']:
            if col in df.columns:
                position_col = col
                break
        
        if position_col:
            logger.info("Adding position-specific features...")
            
            # Calculate position average statistics
            position_stats = df.groupby(position_col).agg({
                col: 'mean' for col in df.select_dtypes(include=['float64', 'int64']).columns
                if col != player_col and col != team_col and col != position_col
            }).reset_index()
            
            # Rename columns to indicate position average
            position_stats.columns = [f'position_avg_{col}' if col != position_col else col for col in position_stats.columns]
            
            # Merge position statistics back to the main dataframe
            df = pd.merge(df, position_stats, on=position_col, how='left')
            
            logger.info("Position-specific features added")
        
        # Add opponent team features
        opponent_col = None
        for col in ['opponent', 'opposition', 'against']:
            if col in df.columns:
                opponent_col = col
                break
        
        if opponent_col:
            logger.info("Adding opponent team features...")
            
            # Calculate opponent team average statistics
            opponent_stats = df.groupby(opponent_col).agg({
                col: 'mean' for col in df.select_dtypes(include=['float64', 'int64']).columns
                if col != player_col and col != team_col and col != opponent_col
            }).reset_index()
            
            # Rename columns to indicate opponent average
            opponent_stats.columns = [f'opponent_avg_{col}' if col != opponent_col else col for col in opponent_stats.columns]
            
            # Merge opponent statistics back to the main dataframe
            df = pd.merge(df, opponent_stats, on=opponent_col, how='left')
            
            logger.info("Opponent team features added")
        
        # Add venue features
        venue_col = None
        for col in ['venue', 'ground', 'stadium']:
            if col in df.columns:
                venue_col = col
                break
        
        if venue_col:
            logger.info("Adding venue features...")
            
            # Calculate venue average statistics
            venue_stats = df.groupby(venue_col).agg({
                col: 'mean' for col in df.select_dtypes(include=['float64', 'int64']).columns
                if col != player_col and col != team_col and col != venue_col
            }).reset_index()
            
            # Rename columns to indicate venue average
            venue_stats.columns = [f'venue_avg_{col}' if col != venue_col else col for col in venue_stats.columns]
            
            # Merge venue statistics back to the main dataframe
            df = pd.merge(df, venue_stats, on=venue_col, how='left')
            
            logger.info("Venue features added")
        
        # Add home/away feature
        home_away_col = None
        for col in ['home_away', 'home', 'is_home']:
            if col in df.columns:
                home_away_col = col
                break
        
        if not home_away_col and opponent_col and team_col:
            logger.info("Creating home/away feature...")
            
            # Try to determine home/away based on venue and team
            if venue_col:
                # Calculate most common venue for each team
                team_venues = df.groupby(team_col)[venue_col].agg(lambda x: x.value_counts().index[0]).to_dict()
                
                # Create home/away feature
                df['is_home'] = df.apply(lambda row: 1 if team_venues.get(row[team_col]) == row[venue_col] else 0, axis=1)
                
                logger.info("Home/away feature created")
        
        # Add form metrics if not already present
        form_metrics = [col for col in df.columns if 'last_' in col or 'consistency' in col or 'trend' in col]
        
        if not form_metrics and disposal_col:
            logger.info("Adding form metrics...")
            
            # Sort by player and date/round
            if sort_col:
                df = df.sort_values([player_col, sort_col])
            
            # Calculate rolling averages for disposals
            df[f'{disposal_col}_last_3'] = df.groupby(player_col)[disposal_col].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
            
            df[f'{disposal_col}_last_5'] = df.groupby(player_col)[disposal_col].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )
            
            # Calculate consistency (standard deviation)
            df[f'{disposal_col}_consistency'] = df.groupby(player_col)[disposal_col].transform(
                lambda x: x.rolling(window=5, min_periods=3).std()
            )
            
            # Calculate trend (positive or negative)
            df[f'{disposal_col}_trend'] = df.groupby(player_col)[disposal_col].transform(
                lambda x: x.rolling(window=3, min_periods=2).apply(
                    lambda y: 1 if y.iloc[-1] > y.iloc[0] else (-1 if y.iloc[-1] < y.iloc[0] else 0)
                )
            )
            
            logger.info("Form metrics added for disposals")
        
        if not form_metrics and goal_col:
            # Calculate rolling averages for goals
            df[f'{goal_col}_last_3'] = df.groupby(player_col)[goal_col].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
            
            df[f'{goal_col}_last_5'] = df.groupby(player_col)[goal_col].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )
            
            # Calculate consistency (standard deviation)
            df[f'{goal_col}_consistency'] = df.groupby(player_col)[goal_col].transform(
                lambda x: x.rolling(window=5, min_periods=3).std()
            )
            
            # Calculate trend (positive or negative)
            df[f'{goal_col}_trend'] = df.groupby(player_col)[goal_col].transform(
                lambda x: x.rolling(window=3, min_periods=2).apply(
                    lambda y: 1 if y.iloc[-1] > y.iloc[0] else (-1 if y.iloc[-1] < y.iloc[0] else 0)
                )
            )
            
            logger.info("Form metrics added for goals")
        
        # Fill NaN values in form metrics
        form_cols = [col for col in df.columns if 'last_' in col or 'consistency' in col or 'trend' in col]
        df[form_cols] = df[form_cols].fillna(0)
        
        # Save engineered features
        engineered_data_path = os.path.join(PROCESSED_DATA_DIR, 'afl_player_stats_engineered.csv')
        df.to_csv(engineered_data_path, index=False)
        
        logger.info(f"Engineered features saved to {engineered_data_path}")
        return df
    except Exception as e:
        logger.error(f"Error engineering features: {e}")
        return None

def prepare_training_data(data, disposal_col, goal_col):
    """
    Prepare the data for training machine learning models
    """
    if data is None:
        return None, None, None, None, None, None
    
    try:
        logger.info("Preparing training data...")
        
        # Create a copy of the data
        df = data.copy()
        
        # Identify categorical and numerical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Remove target columns from features
        if disposal_col and disposal_col in numerical_cols:
            numerical_cols.remove(disposal_col)
        
        if goal_col and goal_col in numerical_cols:
            numerical_cols.remove(goal_col)
        
        # Remove player name and date columns from features
        exclude_cols = []
        for col in categorical_cols:
            if 'player' in col.lower() or 'name' in col.lower() or 'date' in col.lower():
                exclude_cols.append(col)
        
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        
        logger.info(f"Categorical features: {len(categorical_cols)}")
        logger.info(f"Numerical features: {len(numerical_cols)}")
        
        # Prepare feature sets
        X = df[categorical_cols + numerical_cols]
        
        # Prepare target variables
        y_disposal = None
        y_goal = None
        
        if disposal_col and disposal_col in df.columns:
            y_disposal = df[disposal_col]
        
        if goal_col and goal_col in df.columns:
            y_goal = df[goal_col]
        
        # Split data into training and testing sets
        X_train, X_test, y_disposal_train, y_disposal_test, y_goal_train, y_goal_test = None, None, None, None, None, None
        
        if y_disposal is not None:
            X_train, X_test, y_disposal_train, y_disposal_test = train_test_split(
                X, y_disposal, test_size=0.2, random_state=42
            )
            
            if y_goal is not None:
                _, _, y_goal_train, y_goal_test = train_test_split(
                    X, y_goal, test_size=0.2, random_state=42
                )
        elif y_goal is not None:
            X_train, X_test, y_goal_train, y_goal_test = train_test_split(
                X, y_goal, test_size=0.2, random_state=42
            )
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ]
        )
        
        # Fit preprocessor on training data
        preprocessor.fit(X_train)
        
        # Save preprocessor
        preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.pkl')
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(preprocessor, f)
        
        logger.info(f"Preprocessor saved to {preprocessor_path}")
        
        # Save training and testing data
        train_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_disposal_train': y_disposal_train,
            'y_disposal_test': y_disposal_test,
            'y_goal_train': y_goal_train,
            'y_goal_test': y_goal_test,
            'categorical_cols': categorical_cols,
            'numerical_cols': numerical_cols
        }
        
        train_data_path = os.path.join(PROCESSED_DATA_DIR, 'training_data.pkl')
        with open(train_data_path, 'wb') as f:
            pickle.dump(train_data, f)
        
        logger.info(f"Training data saved to {train_data_path}")
        
        return X_train, X_test, y_disposal_train, y_disposal_test, y_goal_train, y_goal_test
    except Exception as e:
        logger.error(f"Error preparing training data: {e}")
        return None, None, None, None, None, None

def main():
    """
    Main function to execute the feature engineering process
    """
    logger.info("Starting feature engineering...")
    
    # Load data
    data = load_data()
    
    if data is None:
        logger.error("Failed to load data")
        return
    
    # Identify key columns
    player_col, team_col, disposal_col, goal_col = identify_key_columns(data)
    
    if player_col is None or team_col is None:
        logger.error("Failed to identify key columns")
        return
    
    # Engineer features
    engineered_data = engineer_features(data, player_col, team_col, disposal_col, goal_col)
    
    if engineered_data is None:
        logger.error("Failed to engineer features")
        return
    
    # Prepare training data
    X_train, X_test, y_disposal_train, y_disposal_test, y_goal_train, y_goal_test = prepare_training_data(
        engineered_data, disposal_col, goal_col
    )
    
    if X_train is None:
        logger.error("Failed to prepare training data")
        return
    
    logger.info("Feature engineering completed successfully!")
    
    # Print training data shapes
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    
    if y_disposal_train is not None:
        logger.info(f"y_disposal_train shape: {y_disposal_train.shape}")
        logger.info(f"y_disposal_test shape: {y_disposal_test.shape}")
    
    if y_goal_train is not None:
        logger.info(f"y_goal_train shape: {y_goal_train.shape}")
        logger.info(f"y_goal_test shape: {y_goal_test.shape}")

if __name__ == "__main__":
    main()
