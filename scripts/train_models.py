#!/usr/bin/env python3
"""
AFL Prediction - Machine Learning Model Training

This script trains machine learning models to predict player disposals and goals
using the preprocessed and engineered features.
"""

import os
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_training')

# Define paths
BASE_DIR = '/home/ubuntu/afl_prediction_project'
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_training_data():
    """
    Load the prepared training data
    """
    try:
        # Check if training data exists
        train_data_path = os.path.join(PROCESSED_DATA_DIR, 'training_data.pkl')
        
        if not os.path.exists(train_data_path):
            logger.error(f"Training data file not found: {train_data_path}")
            return None
        
        # Load training data
        with open(train_data_path, 'rb') as f:
            train_data = pickle.load(f)
        
        logger.info("Loaded training data")
        return train_data
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return None

def load_preprocessor():
    """
    Load the preprocessor for feature transformation
    """
    try:
        # Check if preprocessor exists
        preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.pkl')
        
        if not os.path.exists(preprocessor_path):
            logger.error(f"Preprocessor file not found: {preprocessor_path}")
            return None
        
        # Load preprocessor
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        logger.info("Loaded preprocessor")
        return preprocessor
    except Exception as e:
        logger.error(f"Error loading preprocessor: {e}")
        return None

def train_disposal_model(train_data, preprocessor):
    """
    Train a model to predict player disposals
    """
    if train_data is None or preprocessor is None:
        return None
    
    try:
        logger.info("Training disposal prediction model...")
        
        # Extract training data
        X_train = train_data['X_train']
        X_test = train_data['X_test']
        y_train = train_data['y_disposal_train']
        y_test = train_data['y_disposal_test']
        
        if y_train is None or y_test is None:
            logger.error("Disposal target variables not found in training data")
            return None
        
        # Transform features
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Define models to try
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train_transformed, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_transformed)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            logger.info(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
        
        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['r2'])
        best_model = results[best_model_name]['model']
        best_metrics = results[best_model_name]
        
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Best model metrics - RMSE: {best_metrics['rmse']:.2f}, MAE: {best_metrics['mae']:.2f}, R²: {best_metrics['r2']:.2f}")
        
        # Save best model
        model_path = os.path.join(MODELS_DIR, 'disposals_prediction_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        logger.info(f"Best model saved to {model_path}")
        
        # Save model comparison results
        results_df = pd.DataFrame({
            'Model': list(results.keys()),
            'RMSE': [results[model]['rmse'] for model in results],
            'MAE': [results[model]['mae'] for model in results],
            'R²': [results[model]['r2'] for model in results]
        })
        
        results_path = os.path.join(RESULTS_DIR, 'disposal_model_comparison.csv')
        results_df.to_csv(results_path, index=False)
        
        logger.info(f"Model comparison results saved to {results_path}")
        
        # Create visualization of model comparison
        plt.figure(figsize=(12, 8))
        
        # Plot RMSE
        plt.subplot(2, 2, 1)
        sns.barplot(x='Model', y='RMSE', data=results_df)
        plt.title('RMSE by Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Plot MAE
        plt.subplot(2, 2, 2)
        sns.barplot(x='Model', y='MAE', data=results_df)
        plt.title('MAE by Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Plot R²
        plt.subplot(2, 2, 3)
        sns.barplot(x='Model', y='R²', data=results_df)
        plt.title('R² by Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(RESULTS_DIR, 'disposal_model_comparison.png')
        plt.savefig(viz_path)
        plt.close()
        
        logger.info(f"Model comparison visualization saved to {viz_path}")
        
        # Fine-tune best model
        if best_model_name == 'Random Forest':
            logger.info("Fine-tuning Random Forest model...")
            
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                RandomForestRegressor(random_state=42),
                param_grid=param_grid,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            grid_search.fit(X_train_transformed, y_train)
            
            best_params = grid_search.best_params_
            logger.info(f"Best parameters: {best_params}")
            
            # Train model with best parameters
            tuned_model = RandomForestRegressor(random_state=42, **best_params)
            tuned_model.fit(X_train_transformed, y_train)
            
            # Make predictions
            y_pred = tuned_model.predict(X_test_transformed)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"Tuned model - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
            
            # Save tuned model
            tuned_model_path = os.path.join(MODELS_DIR, 'disposals_prediction_model_tuned.pkl')
            with open(tuned_model_path, 'wb') as f:
                pickle.dump(tuned_model, f)
            
            logger.info(f"Tuned model saved to {tuned_model_path}")
            
            # Return tuned model
            return tuned_model
        
        elif best_model_name == 'Gradient Boosting':
            logger.info("Fine-tuning Gradient Boosting model...")
            
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                GradientBoostingRegressor(random_state=42),
                param_grid=param_grid,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            grid_search.fit(X_train_transformed, y_train)
            
            best_params = grid_search.best_params_
            logger.info(f"Best parameters: {best_params}")
            
            # Train model with best parameters
            tuned_model = GradientBoostingRegressor(random_state=42, **best_params)
            tuned_model.fit(X_train_transformed, y_train)
            
            # Make predictions
            y_pred = tuned_model.predict(X_test_transformed)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"Tuned model - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
            
            # Save tuned model
            tuned_model_path = os.path.join(MODELS_DIR, 'disposals_prediction_model_tuned.pkl')
            with open(tuned_model_path, 'wb') as f:
                pickle.dump(tuned_model, f)
            
            logger.info(f"Tuned model saved to {tuned_model_path}")
            
            # Return tuned model
            return tuned_model
        
        # Return best model
        return best_model
    except Exception as e:
        logger.error(f"Error training disposal model: {e}")
        return None

def train_goal_model(train_data, preprocessor):
    """
    Train a model to predict player goals
    """
    if train_data is None or preprocessor is None:
        return None
    
    try:
        logger.info("Training goal prediction model...")
        
        # Extract training data
        X_train = train_data['X_train']
        X_test = train_data['X_test']
        y_train = train_data['y_goal_train']
        y_test = train_data['y_goal_test']
        
        if y_train is None or y_test is None:
            logger.error("Goal target variables not found in training data")
            return None
        
        # Transform features
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Define models to try
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train_transformed, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_transformed)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            logger.info(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
        
        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['r2'])
        best_model = results[best_model_name]['model']
        best_metrics = results[best_model_name]
        
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Best model metrics - RMSE: {best_metrics['rmse']:.2f}, MAE: {best_metrics['mae']:.2f}, R²: {best_metrics['r2']:.2f}")
        
        # Save best model
        model_path = os.path.join(MODELS_DIR, 'goals_prediction_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        logger.info(f"Best model saved to {model_path}")
        
        # Save model comparison results
        results_df = pd.DataFrame({
            'Model': list(results.keys()),
            'RMSE': [results[model]['rmse'] for model in results],
            'MAE': [results[model]['mae'] for model in results],
            'R²': [results[model]['r2'] for model in results]
        })
        
        results_path = os.path.join(RESULTS_DIR, 'goal_model_comparison.csv')
        results_df.to_csv(results_path, index=False)
        
        logger.info(f"Model comparison results saved to {results_path}")
        
        # Create visualization of model comparison
        plt.figure(figsize=(12, 8))
        
        # Plot RMSE
        plt.subplot(2, 2, 1)
        sns.barplot(x='Model', y='RMSE', data=results_df)
        plt.title('RMSE by Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Plot MAE
        plt.subplot(2, 2, 2)
        sns.barplot(x='Model', y='MAE', data=results_df)
        plt.title('MAE by Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Plot R²
        plt.subplot(2, 2, 3)
        sns.barplot(x='Model', y='R²', data=results_df)
        plt.title('R² by Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(RESULTS_DIR, 'goal_model_comparison.png')
        plt.savefig(viz_path)
        plt.close()
        
        logger.info(f"Model comparison visualization saved to {viz_path}")
        
        # Fine-tune best model
        if best_model_name == 'Random Forest':
            logger.info("Fine-tuning Random Forest model...")
            
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                RandomForestRegressor(random_state=42),
                param_grid=param_grid,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            grid_search.fit(X_train_transformed, y_train)
            
            best_params = grid_search.best_params_
            logger.info(f"Best parameters: {best_params}")
            
            # Train model with best parameters
            tuned_model = RandomForestRegressor(random_state=42, **best_params)
            tuned_model.fit(X_train_transformed, y_train)
            
            # Make predictions
            y_pred = tuned_model.predict(X_test_transformed)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"Tuned model - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
            
            # Save tuned model
            tuned_model_path = os.path.join(MODELS_DIR, 'goals_prediction_model_tuned.pkl')
            with open(tuned_model_path, 'wb') as f:
                pickle.dump(tuned_model, f)
            
            logger.info(f"Tuned model saved to {tuned_model_path}")
            
            # Return tuned model
            return tuned_model
        
        elif best_model_name == 'Gradient Boosting':
            logger.info("Fine-tuning Gradient Boosting model...")
            
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                GradientBoostingRegressor(random_state=42),
                param_grid=param_grid,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            grid_search.fit(X_train_transformed, y_train)
            
            best_params = grid_search.best_params_
            logger.info(f"Best parameters: {best_params}")
            
            # Train model with best parameters
            tuned_model = GradientBoostingRegressor(random_state=42, **best_params)
            tuned_model.fit(X_train_transformed, y_train)
            
            # Make predictions
            y_pred = tuned_model.predict(X_test_transformed)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"Tuned model - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
            
            # Save tuned model
            tuned_model_path = os.path.join(MODELS_DIR, 'goals_prediction_model_tuned.pkl')
            with open(tuned_model_path, 'wb') as f:
                pickle.dump(tuned_model, f)
            
            logger.info(f"Tuned model saved to {tuned_model_path}")
            
            # Return tuned model
            return tuned_model
        
        # Return best model
        return best_model
    except Exception as e:
        logger.error(f"Error training goal model: {e}")
        return None

def analyze_feature_importance(disposal_model, goal_model, train_data, preprocessor):
    """
    Analyze feature importance for the trained models
    """
    if disposal_model is None and goal_model is None:
        return
    
    try:
        logger.info("Analyzing feature importance...")
        
        # Get feature names
        categorical_cols = train_data['categorical_cols']
        numerical_cols = train_data['numerical_cols']
        
        # Get one-hot encoded feature names
        categorical_features = []
        for col in categorical_cols:
            unique_values = train_data['X_train'][col].unique()
            for value in unique_values:
                categorical_features.append(f"{col}_{value}")
        
        # Combine feature names
        feature_names = numerical_cols + categorical_features
        
        # Analyze disposal model feature importance
        if disposal_model is not None and hasattr(disposal_model, 'feature_importances_'):
            logger.info("Analyzing disposal model feature importance...")
            
            # Get feature importances
            importances = disposal_model.feature_importances_
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names[:len(importances)],
                'Importance': importances
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Save to file
            importance_path = os.path.join(RESULTS_DIR, 'disposal_feature_importance.csv')
            importance_df.to_csv(importance_path, index=False)
            
            logger.info(f"Disposal feature importance saved to {importance_path}")
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
            plt.title('Top 20 Features for Disposal Prediction')
            plt.tight_layout()
            
            # Save visualization
            viz_path = os.path.join(RESULTS_DIR, 'disposal_feature_importance.png')
            plt.savefig(viz_path)
            plt.close()
            
            logger.info(f"Disposal feature importance visualization saved to {viz_path}")
        
        # Analyze goal model feature importance
        if goal_model is not None and hasattr(goal_model, 'feature_importances_'):
            logger.info("Analyzing goal model feature importance...")
            
            # Get feature importances
            importances = goal_model.feature_importances_
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names[:len(importances)],
                'Importance': importances
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Save to file
            importance_path = os.path.join(RESULTS_DIR, 'goal_feature_importance.csv')
            importance_df.to_csv(importance_path, index=False)
            
            logger.info(f"Goal feature importance saved to {importance_path}")
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
            plt.title('Top 20 Features for Goal Prediction')
            plt.tight_layout()
            
            # Save visualization
            viz_path = os.path.join(RESULTS_DIR, 'goal_feature_importance.png')
            plt.savefig(viz_path)
            plt.close()
            
            logger.info(f"Goal feature importance visualization saved to {viz_path}")
    except Exception as e:
        logger.error(f"Error analyzing feature importance: {e}")

def create_prediction_pipeline(disposal_model, goal_model, preprocessor):
    """
    Create a prediction pipeline for easy use in the web application
    """
    if disposal_model is None and goal_model is None:
        return
    
    try:
        logger.info("Creating prediction pipeline...")
        
        # Create pipeline dictionary
        pipeline = {
            'preprocessor': preprocessor,
            'disposal_model': disposal_model,
            'goal_model': goal_model
        }
        
        # Save pipeline
        pipeline_path = os.path.join(MODELS_DIR, 'prediction_pipeline.pkl')
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline, f)
        
        logger.info(f"Prediction pipeline saved to {pipeline_path}")
    except Exception as e:
        logger.error(f"Error creating prediction pipeline: {e}")

def main():
    """
    Main function to execute the model training process
    """
    logger.info("Starting model training...")
    
    # Load training data
    train_data = load_training_data()
    
    if train_data is None:
        logger.error("Failed to load training data")
        return
    
    # Load preprocessor
    preprocessor = load_preprocessor()
    
    if preprocessor is None:
        logger.error("Failed to load preprocessor")
        return
    
    # Train disposal model
    disposal_model = train_disposal_model(train_data, preprocessor)
    
    # Train goal model
    goal_model = train_goal_model(train_data, preprocessor)
    
    # Analyze feature importance
    analyze_feature_importance(disposal_model, goal_model, train_data, preprocessor)
    
    # Create prediction pipeline
    create_prediction_pipeline(disposal_model, goal_model, preprocessor)
    
    logger.info("Model training completed successfully!")

if __name__ == "__main__":
    main()
