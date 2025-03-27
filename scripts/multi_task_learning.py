#!/usr/bin/env python3
"""
Multi-Task Learning Model for AFL Predictions

This script implements a multi-task learning model that can simultaneously predict
multiple related targets (disposals and goals) for AFL players.
"""

import os
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from data_field_mapping import get_multi_task_feature_columns, get_multi_task_target_columns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('multi_task_learning')

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_and_preprocess_data():
    """
    Load and preprocess the AFL player statistics data from raw data directories
    """
    logger.info("Loading and preprocessing data...")
    
    # Lists to store dataframes
    historical_dfs = []
    current_dfs = []
    
    # Load historical data
    historical_dir = os.path.join(RAW_DATA_DIR, 'historical')
    if os.path.exists(historical_dir):
        for file in os.listdir(historical_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(historical_dir, file)
                try:
                    df = pd.read_csv(file_path)
                    historical_dfs.append(df)
                    logger.info(f"Loaded historical data from {file}")
                except Exception as e:
                    logger.error(f"Error loading {file}: {e}")
    
    # Load current season data
    current_dir = os.path.join(RAW_DATA_DIR, 'current')
    if os.path.exists(current_dir):
        for file in os.listdir(current_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(current_dir, file)
                try:
                    df = pd.read_csv(file_path)
                    current_dfs.append(df)
                    logger.info(f"Loaded current season data from {file}")
                except Exception as e:
                    logger.error(f"Error loading {file}: {e}")
    
    # Combine all data
    if historical_dfs and current_dfs:
        all_data = pd.concat(historical_dfs + current_dfs, ignore_index=True)
    elif historical_dfs:
        all_data = pd.concat(historical_dfs, ignore_index=True)
    elif current_dfs:
        all_data = pd.concat(current_dfs, ignore_index=True)
    else:
        logger.error("No data files found in raw data directories")
        return None
    
    logger.info(f"Combined data shape: {all_data.shape}")
    
    # Identify target columns
    target_columns = identify_target_columns(all_data)
    if not target_columns:
        logger.error("Could not identify target columns")
        return None
    
    # Preprocess data
    processed_data = preprocess_data(all_data, target_columns)
    
    return processed_data

def identify_target_columns(df):
    """
    Identify the target columns for disposals and goals using the data field mapping
    """
    target_columns = {}
    
    # Get target columns from data field mapping
    target_mapping = get_multi_task_target_columns()
    
    # Check if the mapped columns exist in the dataframe
    for target_name, column_name in target_mapping.items():
        if column_name in df.columns:
            target_columns[target_name] = column_name
            logger.info(f"Identified {target_name} column: {column_name}")
        else:
            # Fall back to searching for alternative column names
            if target_name == 'disposals':
                disposal_candidates = ['Dis.', 'disposals', 'disposal', 'total_disposals', 'Disposals']
                for col in disposal_candidates:
                    if col in df.columns:
                        target_columns['disposals'] = col
                        logger.info(f"Identified disposal column: {col}")
                        break
            elif target_name == 'goals':
                goal_candidates = ['Goals', 'goals', 'gls', 'gls_avg', 'goals_avg', 'total_goals', 'Goals_Total']
                for col in goal_candidates:
                    if col in df.columns:
                        target_columns['goals'] = col
                        logger.info(f"Identified goal column: {col}")
                        break
    
    return target_columns

def preprocess_data(df, target_columns):
    """
    Preprocess the data for multi-task learning
    """
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Handle missing values
    data = data.fillna(0)
    
    # Convert date columns to datetime
    if 'match_date' in data.columns:
        data['match_date'] = pd.to_datetime(data['match_date'])
        # Extract features from date
        data['day_of_week'] = data['match_date'].dt.dayofweek
        data['month'] = data['match_date'].dt.month
    
    # Encode categorical variables
    categorical_cols = ['team', 'opposition', 'venue']
    for col in categorical_cols:
        if col in data.columns:
            # Create dummies and drop the original column
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
            data = pd.concat([data, dummies], axis=1)
            data.drop(col, axis=1, inplace=True)
    
    # Create feature matrix and target vectors
    # Select only numeric columns for features
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    
    # Exclude target columns and any other non-feature columns
    exclude_cols = list(target_columns.values()) + ['player_name', 'match_date', 'round', 'season']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Create feature matrix
    X = data[feature_cols].copy()
    
    # Create target vectors
    y_dict = {}
    for target_name, target_col in target_columns.items():
        if target_col in data.columns:
            y_dict[target_name] = data[target_col].copy()
    
    # Split data into train and test sets
    X_train, X_test, y_train_dict, y_test_dict = {}, {}, {}, {}
    
    # Use the same split for all targets
    train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
    
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    
    for target_name, y_values in y_dict.items():
        y_train_dict[target_name] = y_values.iloc[train_idx]
        y_test_dict[target_name] = y_values.iloc[test_idx]
    
    # Create a preprocessor
    preprocessor = StandardScaler()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Save preprocessor
    preprocessor_path = os.path.join(MODELS_DIR, 'multi_task_preprocessor.pkl')
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    logger.info(f"Saved preprocessor to {preprocessor_path}")
    
    # Save feature column names
    feature_cols_path = os.path.join(MODELS_DIR, 'feature_columns.pkl')
    with open(feature_cols_path, 'wb') as f:
        pickle.dump(feature_cols, f)
    logger.info(f"Saved feature column names to {feature_cols_path}")
    
    # Return processed data
    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train_dict': y_train_dict,
        'y_test_dict': y_test_dict,
        'feature_cols': feature_cols,
        'target_columns': target_columns,
        'preprocessor': preprocessor
    }
    
    return processed_data

def build_multi_task_model(input_dim, output_dims):
    """
    Build a multi-task learning model using Keras
    
    Args:
        input_dim: Dimension of input features
        output_dims: Dictionary of output dimensions for each task
    
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = Input(shape=(input_dim,), name='input')
    
    # Shared layers
    x = Dense(128, activation='relu', name='shared_dense_1')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', name='shared_dense_2')(x)
    x = Dropout(0.3)(x)
    
    # Task-specific layers and outputs
    outputs = {}
    for task_name in output_dims.keys():
        task_layer = Dense(32, activation='relu', name=f'{task_name}_dense')(x)
        outputs[task_name] = Dense(1, activation='linear', name=f'{task_name}_output')(task_layer)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Define loss and metrics for each output
    losses = {task_name: 'mse' for task_name in output_dims.keys()}
    metrics = {task_name: ['mae', 'mse'] for task_name in output_dims.keys()}
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=losses,
        metrics=metrics
    )
    
    return model

def train_multi_task_model(processed_data):
    """
    Train the multi-task learning model
    
    Args:
        processed_data: Dictionary containing processed data
    
    Returns:
        Trained model and training history
    """
    logger.info("Training multi-task learning model...")
    
    # Extract data
    X_train_scaled = processed_data['X_train_scaled']
    X_test_scaled = processed_data['X_test_scaled']
    y_train_dict = processed_data['y_train_dict']
    y_test_dict = processed_data['y_test_dict']
    
    # Get input dimension
    input_dim = X_train_scaled.shape[1]
    
    # Get output dimensions
    output_dims = {task_name: 1 for task_name in y_train_dict.keys()}
    
    # Build model
    model = build_multi_task_model(input_dim, output_dims)
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, 'multi_task_model.h5'),
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Train model
    history = model.fit(
        X_train_scaled,
        y_train_dict,
        validation_data=(X_test_scaled, y_test_dict),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    model.save(os.path.join(MODELS_DIR, 'multi_task_model.h5'))
    logger.info("Model saved to models/multi_task_model.h5")
    
    return model, history

def evaluate_multi_task_model(model, processed_data):
    """
    Evaluate the multi-task learning model
    
    Args:
        model: Trained multi-task model
        processed_data: Dictionary containing processed data
    
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating multi-task learning model...")
    
    # Extract data
    X_test_scaled = processed_data['X_test_scaled']
    y_test_dict = processed_data['y_test_dict']
    
    # Make predictions
    predictions = model.predict(X_test_scaled)
    
    # Calculate metrics for each task
    metrics = {}
    for task_name, y_true in y_test_dict.items():
        y_pred = predictions[task_name].flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics[task_name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        logger.info(f"{task_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, 'multi_task_model_metrics.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Create visualization of predictions vs actual
    plt.figure(figsize=(15, 5 * len(y_test_dict)))
    
    for i, (task_name, y_true) in enumerate(y_test_dict.items()):
        y_pred = predictions[task_name].flatten()
        
        plt.subplot(len(y_test_dict), 2, 2*i+1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel(f'Actual {task_name}')
        plt.ylabel(f'Predicted {task_name}')
        plt.title(f'{task_name} - Actual vs Predicted')
        
        plt.subplot(len(y_test_dict), 2, 2*i+2)
        plt.hist(y_true - y_pred, bins=50)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title(f'{task_name} - Error Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'multi_task_model_evaluation.png'))
    plt.close()
    
    return metrics

def create_prediction_pipeline(model, processed_data):
    """
    Create a prediction pipeline for easy inference
    
    Args:
        model: Trained multi-task model
        processed_data: Dictionary containing processed data
    
    Returns:
        Prediction pipeline
    """
    logger.info("Creating prediction pipeline...")
    
    # Extract preprocessor and feature columns
    preprocessor = processed_data['preprocessor']
    feature_cols = processed_data['feature_cols']
    target_columns = processed_data['target_columns']
    
    # Create prediction pipeline
    prediction_pipeline = {
        'model': model,
        'preprocessor': preprocessor,
        'feature_cols': feature_cols,
        'target_columns': target_columns
    }
    
    # Save prediction pipeline
    pipeline_path = os.path.join(MODELS_DIR, 'multi_task_prediction_pipeline.pkl')
    with open(pipeline_path, 'wb') as f:
        pickle.dump(prediction_pipeline, f)
    logger.info(f"Saved prediction pipeline to {pipeline_path}")
    
    return prediction_pipeline

def main():
    """
    Main function to run the multi-task learning pipeline
    """
    logger.info("Starting multi-task learning pipeline...")
    
    # Load and preprocess data
    processed_data = load_and_preprocess_data()
    if processed_data is None:
        logger.error("Failed to load and preprocess data")
        return
    
    # Train multi-task model
    model, history = train_multi_task_model(processed_data)
    
    # Evaluate model
    metrics = evaluate_multi_task_model(model, processed_data)
    
    # Create prediction pipeline
    prediction_pipeline = create_prediction_pipeline(model, processed_data)
    
    logger.info("Multi-task learning pipeline completed successfully")

if __name__ == "__main__":
    main()
