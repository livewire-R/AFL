#!/usr/bin/env python3
"""
AFL Player Statistics Machine Learning Model Development

This script develops machine learning models to predict player disposals and goals
using the processed AFL player statistics data.
"""

import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# Define paths
BASE_DIR = '/home/ubuntu/afl_prediction_project'
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data/processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ANALYSIS_DIR = os.path.join(BASE_DIR, 'data/analysis')

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

def load_processed_data():
    """
    Load the processed player statistics data
    """
    pickle_path = os.path.join(PROCESSED_DATA_DIR, 'afl_player_stats_all_years.pkl')
    csv_path = os.path.join(PROCESSED_DATA_DIR, 'afl_player_stats_all_years.csv')
    
    if os.path.exists(pickle_path):
        df = pd.read_pickle(pickle_path)
        print(f"Loaded processed data from {pickle_path}")
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded processed data from {csv_path}")
    else:
        print("No processed data found. Please run preprocess_data.py first.")
        return None
    
    return df

def identify_target_columns(df):
    """
    Identify the target columns for disposals and goals
    """
    disposal_candidates = ['dis.', 'disposals', 'disposal', 'total_disposals']
    goal_candidates = ['goals', 'gls', 'gls_avg', 'goals_avg', 'total_goals']
    
    disposal_col = None
    for col in disposal_candidates:
        if col in df.columns:
            disposal_col = col
            break
    
    goal_col = None
    for col in goal_candidates:
        if col in df.columns:
            goal_col = col
            break
    
    if disposal_col is None:
        print("Warning: Could not identify disposal column")
    else:
        print(f"Identified disposal column: {disposal_col}")
    
    if goal_col is None:
        print("Warning: Could not identify goal column")
    else:
        print(f"Identified goal column: {goal_col}")
    
    return disposal_col, goal_col

def prepare_features_and_target(df, target_col, include_form=True):
    """
    Prepare features and target for model training
    """
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in data")
        return None, None
    
    # Select only numeric columns for features
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Exclude the target column and any other prediction targets
    exclude_cols = ['dis.', 'disposals', 'disposal', 'total_disposals', 
                    'goals', 'gls', 'gls_avg', 'goals_avg', 'total_goals']
    
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # If we have form metrics, include them
    form_cols = []
    if include_form:
        form_candidates = [col for col in df.columns if 'form' in col.lower() or 
                          'rolling' in col.lower() or 'avg_last' in col.lower()]
        form_cols = [col for col in form_candidates if col in df.columns]
        if form_cols:
            print(f"Including {len(form_cols)} form metrics in features")
        else:
            print("No form metrics found in data")
    
    # Combine regular features and form metrics
    all_feature_cols = list(set(feature_cols + form_cols))
    
    # Create feature matrix and target vector
    X = df[all_feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle missing values
    X = X.fillna(0)
    y = y.fillna(0)
    
    print(f"Prepared features ({X.shape[1]} columns) and target ({target_col})")
    return X, y

def train_and_evaluate_models(X, y, target_name):
    """
    Train and evaluate multiple regression models
    """
    if X is None or y is None:
        return None
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining models for {target_name} prediction...")
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # Define models to evaluate
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    # Dictionary to store results
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Create a pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Store results
        results[name] = {
            'model': pipeline,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        print(f"{name} - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        print(f"{name} - Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
        print(f"{name} - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    
    # Find the best model based on test RMSE
    best_model_name = min(results, key=lambda k: results[k]['test_rmse'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model for {target_name}: {best_model_name}")
    print(f"Test RMSE: {results[best_model_name]['test_rmse']:.4f}")
    print(f"Test MAE: {results[best_model_name]['test_mae']:.4f}")
    print(f"Test R²: {results[best_model_name]['test_r2']:.4f}")
    
    # Save the best model
    model_path = os.path.join(MODELS_DIR, f'{target_name.lower()}_prediction_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Best model saved to {model_path}")
    
    # Save model comparison results
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Train RMSE': [results[m]['train_rmse'] for m in results],
        'Test RMSE': [results[m]['test_rmse'] for m in results],
        'Train MAE': [results[m]['train_mae'] for m in results],
        'Test MAE': [results[m]['test_mae'] for m in results],
        'Train R²': [results[m]['train_r2'] for m in results],
        'Test R²': [results[m]['test_r2'] for m in results]
    })
    
    results_path = os.path.join(ANALYSIS_DIR, f'{target_name.lower()}_model_comparison.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Model comparison results saved to {results_path}")
    
    # Plot model comparison
    plt.figure(figsize=(12, 8))
    
    # Plot RMSE comparison
    plt.subplot(2, 1, 1)
    models_list = results_df['Model'].tolist()
    train_rmse = results_df['Train RMSE'].tolist()
    test_rmse = results_df['Test RMSE'].tolist()
    
    x = np.arange(len(models_list))
    width = 0.35
    
    plt.bar(x - width/2, train_rmse, width, label='Train RMSE')
    plt.bar(x + width/2, test_rmse, width, label='Test RMSE')
    
    plt.xlabel('Model')
    plt.ylabel('RMSE')
    plt.title(f'RMSE Comparison for {target_name} Prediction')
    plt.xticks(x, models_list, rotation=45, ha='right')
    plt.legend()
    
    # Plot R² comparison
    plt.subplot(2, 1, 2)
    train_r2 = results_df['Train R²'].tolist()
    test_r2 = results_df['Test R²'].tolist()
    
    plt.bar(x - width/2, train_r2, width, label='Train R²')
    plt.bar(x + width/2, test_r2, width, label='Test R²')
    
    plt.xlabel('Model')
    plt.ylabel('R²')
    plt.title(f'R² Comparison for {target_name} Prediction')
    plt.xticks(x, models_list, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    
    plot_path = os.path.join(ANALYSIS_DIR, f'{target_name.lower()}_model_comparison.png')
    plt.savefig(plot_path)
    print(f"Model comparison plot saved to {plot_path}")
    
    return best_model, results_df

def analyze_feature_importance(model, feature_names, target_name, top_n=20):
    """
    Analyze feature importance for the best model (if applicable)
    """
    # Check if the model has feature_importances_ attribute (tree-based models)
    if hasattr(model[-1], 'feature_importances_'):
        # Get feature importances
        importances = model[-1].feature_importances_
        
        # Create a DataFrame for feature importances
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Save feature importances
        importance_path = os.path.join(ANALYSIS_DIR, f'{target_name.lower()}_feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)
        print(f"Feature importance saved to {importance_path}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        
        # Take top N features
        top_features = importance_df.head(top_n)
        
        # Create horizontal bar plot
        sns.barplot(x='Importance', y='Feature', data=top_features)
        
        plt.title(f'Top {top_n} Features for {target_name} Prediction', fontsize=16)
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(ANALYSIS_DIR, f'{target_name.lower()}_feature_importance.png')
        plt.savefig(plot_path)
        print(f"Feature importance plot saved to {plot_path}")
        
        return importance_df
    else:
        print(f"Feature importance not available for this model type")
        return None

def tune_best_model(X, y, target_name, model_type):
    """
    Perform hyperparameter tuning for the best model type
    """
    print(f"\nPerforming hyperparameter tuning for {target_name} prediction using {model_type}...")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define hyperparameter grid based on model type
    if model_type == 'Random Forest':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'Ridge Regression':
        model = Ridge()
        param_grid = {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        }
    elif model_type == 'Lasso Regression':
        model = Lasso()
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
        }
    else:  # Linear Regression or other
        print("No hyperparameter tuning needed for this model type")
        return None
    
    # Create a pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # Set up grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        {'model__' + key: value for key, value in param_grid.items()},
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # Perform grid search
    grid_search.fit(X_train, y_train)
    
    # Get best parameters and model
    best_params = {k.replace('model__', ''): v for k, v in grid_search.best_params_.items()}
    best_model = grid_search.best_estimator_
    
    print(f"Best parameters: {best_params}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    print(f"Tuned model - Test RMSE: {test_rmse:.4f}")
    print(f"Tuned model - Test MAE: {test_mae:.4f}")
    print(f"Tuned model - Test R²: {test_r2:.4f}")
    
    # Save the tuned model
    model_path = os.path.join(MODELS_DIR, f'{target_name.lower()}_prediction_model_tuned.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Tuned model saved to {model_path}")
    
    return best_model

def main():
    """
    Main function to execute the model development pipeline
    """
    print("Starting AFL player statistics model development...")
    
    # Load processed data
    df = load_processed_data()
    
    if df is None:
        return
    
    # Identify target columns
    disposal_col, goal_col = identify_target_columns(df)
    
    # Train and evaluate models for disposals
    if disposal_col is not None:
        print("\n" + "="*50)
        print(f"Developing model for {disposal_col} prediction")
        print("="*50)
        
        # Prepare features and target
        X, y = prepare_features_and_target(df, disposal_col)
        
        if X is not None and y is not None:
            # Train and evaluate models
            best_model, results_df = train_and_evaluate_models(X, y, "Disposals")
            
            # Analyze feature importance
            if best_model is not None:
                analyze_feature_importance(best_model, X.columns, "Disposals")
                
                # Get the best model type
                best_model_type = results_df.loc[results_df['Test RMSE'].idxmin(), 'Model']
                
                # Tune the best model
                tune_best_model(X, y, "Disposals", best_model_type)
    
    # Train and evaluate models for goals
    if goal_col is not None:
        print("\n" + "="*50)
        print(f"Developing model for {goal_col} prediction")
        print("="*50)
        
        # Prepare features and target
        X, y = prepare_features_and_target(df, goal_col)
        
        if X is not None and y is not None:
            # Train and evaluate models
            best_model, results_df = train_and_evaluate_models(X, y, "Goals")
            
            # Analyze feature importance
            if best_model is not None:
                analyze_feature_importance(best_model, X.columns, "Goals")
                
                # Get the best model type
                best_model_type = results_df.loc[results_df['Test RMSE'].idxmin(), 'Model']
                
                # Tune the best model
                tune_best_model(X, y, "Goals", best_model_type)
    
    print("\nModel development completed successfully!")

if __name__ == "__main__":
    main()
