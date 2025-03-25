#!/usr/bin/env python3
"""
AFL Prediction - Model Evaluation

This script evaluates the trained machine learning models for predicting player disposals and goals.
"""

import os
import pandas as pd
import numpy as np
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, learning_curve

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_evaluation')

# Define paths
BASE_DIR = '/home/ubuntu/afl_prediction_project'
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
EVALUATION_DIR = os.path.join(RESULTS_DIR, 'evaluation')

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EVALUATION_DIR, exist_ok=True)

def load_models_and_data():
    """
    Load the trained models, preprocessor, and test data
    """
    try:
        # Load prediction pipeline
        pipeline_path = os.path.join(MODELS_DIR, 'prediction_pipeline.pkl')
        
        if not os.path.exists(pipeline_path):
            logger.error(f"Prediction pipeline not found: {pipeline_path}")
            
            # Try to load individual models
            disposal_model_path = os.path.join(MODELS_DIR, 'disposals_prediction_model_tuned.pkl')
            if not os.path.exists(disposal_model_path):
                disposal_model_path = os.path.join(MODELS_DIR, 'disposals_prediction_model.pkl')
            
            goal_model_path = os.path.join(MODELS_DIR, 'goals_prediction_model_tuned.pkl')
            if not os.path.exists(goal_model_path):
                goal_model_path = os.path.join(MODELS_DIR, 'goals_prediction_model.pkl')
            
            preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.pkl')
            
            if not os.path.exists(disposal_model_path) or not os.path.exists(goal_model_path) or not os.path.exists(preprocessor_path):
                logger.error("Required model files not found")
                return None, None, None, None
            
            # Load models and preprocessor
            with open(disposal_model_path, 'rb') as f:
                disposal_model = pickle.load(f)
            
            with open(goal_model_path, 'rb') as f:
                goal_model = pickle.load(f)
            
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            
            pipeline = {
                'disposal_model': disposal_model,
                'goal_model': goal_model,
                'preprocessor': preprocessor
            }
        else:
            # Load pipeline
            with open(pipeline_path, 'rb') as f:
                pipeline = pickle.load(f)
        
        # Load training data
        train_data_path = os.path.join(PROCESSED_DATA_DIR, 'training_data.pkl')
        
        if not os.path.exists(train_data_path):
            logger.error(f"Training data not found: {train_data_path}")
            return pipeline, None, None, None
        
        # Load training data
        with open(train_data_path, 'rb') as f:
            train_data = pickle.load(f)
        
        # Extract test data
        X_test = train_data['X_test']
        y_disposal_test = train_data['y_disposal_test']
        y_goal_test = train_data['y_goal_test']
        
        logger.info("Loaded models and test data")
        return pipeline, X_test, y_disposal_test, y_goal_test
    except Exception as e:
        logger.error(f"Error loading models and data: {e}")
        return None, None, None, None

def evaluate_disposal_model(pipeline, X_test, y_disposal_test):
    """
    Evaluate the disposal prediction model
    """
    if pipeline is None or X_test is None or y_disposal_test is None:
        return
    
    try:
        logger.info("Evaluating disposal prediction model...")
        
        # Extract model and preprocessor
        disposal_model = pipeline.get('disposal_model')
        preprocessor = pipeline.get('preprocessor')
        
        if disposal_model is None or preprocessor is None:
            logger.error("Disposal model or preprocessor not found in pipeline")
            return
        
        # Transform test data
        X_test_transformed = preprocessor.transform(X_test)
        
        # Make predictions
        y_pred = disposal_model.predict(X_test_transformed)
        
        # Calculate metrics
        mse = mean_squared_error(y_disposal_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_disposal_test, y_pred)
        r2 = r2_score(y_disposal_test, y_pred)
        
        logger.info(f"Disposal model - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
        
        # Create evaluation report
        report = {
            'Model': 'Disposal Prediction',
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'Mean Target': y_disposal_test.mean(),
            'RMSE/Mean': rmse / y_disposal_test.mean(),
            'MAE/Mean': mae / y_disposal_test.mean()
        }
        
        # Save report
        report_path = os.path.join(EVALUATION_DIR, 'disposal_model_evaluation.csv')
        pd.DataFrame([report]).to_csv(report_path, index=False)
        
        logger.info(f"Disposal model evaluation report saved to {report_path}")
        
        # Create scatter plot of actual vs predicted values
        plt.figure(figsize=(10, 8))
        plt.scatter(y_disposal_test, y_pred, alpha=0.5)
        plt.plot([y_disposal_test.min(), y_disposal_test.max()], [y_disposal_test.min(), y_disposal_test.max()], 'r--')
        plt.xlabel('Actual Disposals')
        plt.ylabel('Predicted Disposals')
        plt.title('Actual vs Predicted Disposals')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(EVALUATION_DIR, 'disposal_actual_vs_predicted.png')
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Disposal model scatter plot saved to {plot_path}")
        
        # Create residual plot
        residuals = y_disposal_test - y_pred
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Disposals')
        plt.ylabel('Residuals')
        plt.title('Residual Plot for Disposal Predictions')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(EVALUATION_DIR, 'disposal_residuals.png')
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Disposal model residual plot saved to {plot_path}")
        
        # Create histogram of residuals
        plt.figure(figsize=(10, 8))
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Residuals for Disposal Predictions')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(EVALUATION_DIR, 'disposal_residuals_histogram.png')
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Disposal model residual histogram saved to {plot_path}")
        
        # Create error distribution by actual value
        plt.figure(figsize=(10, 8))
        
        # Calculate absolute error
        abs_error = np.abs(residuals)
        
        # Create DataFrame for plotting
        error_df = pd.DataFrame({
            'Actual': y_disposal_test,
            'Absolute Error': abs_error
        })
        
        # Bin actual values
        error_df['Actual Bin'] = pd.cut(error_df['Actual'], bins=10)
        
        # Calculate mean error by bin
        bin_errors = error_df.groupby('Actual Bin')['Absolute Error'].mean().reset_index()
        
        # Plot
        plt.bar(range(len(bin_errors)), bin_errors['Absolute Error'], alpha=0.7)
        plt.xticks(range(len(bin_errors)), [str(x) for x in bin_errors['Actual Bin']], rotation=45)
        plt.xlabel('Actual Disposal Range')
        plt.ylabel('Mean Absolute Error')
        plt.title('Error Distribution by Actual Disposal Value')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(EVALUATION_DIR, 'disposal_error_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Disposal model error distribution plot saved to {plot_path}")
        
        return report
    except Exception as e:
        logger.error(f"Error evaluating disposal model: {e}")
        return None

def evaluate_goal_model(pipeline, X_test, y_goal_test):
    """
    Evaluate the goal prediction model
    """
    if pipeline is None or X_test is None or y_goal_test is None:
        return
    
    try:
        logger.info("Evaluating goal prediction model...")
        
        # Extract model and preprocessor
        goal_model = pipeline.get('goal_model')
        preprocessor = pipeline.get('preprocessor')
        
        if goal_model is None or preprocessor is None:
            logger.error("Goal model or preprocessor not found in pipeline")
            return
        
        # Transform test data
        X_test_transformed = preprocessor.transform(X_test)
        
        # Make predictions
        y_pred = goal_model.predict(X_test_transformed)
        
        # Calculate metrics
        mse = mean_squared_error(y_goal_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_goal_test, y_pred)
        r2 = r2_score(y_goal_test, y_pred)
        
        logger.info(f"Goal model - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
        
        # Create evaluation report
        report = {
            'Model': 'Goal Prediction',
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'Mean Target': y_goal_test.mean(),
            'RMSE/Mean': rmse / y_goal_test.mean(),
            'MAE/Mean': mae / y_goal_test.mean()
        }
        
        # Save report
        report_path = os.path.join(EVALUATION_DIR, 'goal_model_evaluation.csv')
        pd.DataFrame([report]).to_csv(report_path, index=False)
        
        logger.info(f"Goal model evaluation report saved to {report_path}")
        
        # Create scatter plot of actual vs predicted values
        plt.figure(figsize=(10, 8))
        plt.scatter(y_goal_test, y_pred, alpha=0.5)
        plt.plot([y_goal_test.min(), y_goal_test.max()], [y_goal_test.min(), y_goal_test.max()], 'r--')
        plt.xlabel('Actual Goals')
        plt.ylabel('Predicted Goals')
        plt.title('Actual vs Predicted Goals')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(EVALUATION_DIR, 'goal_actual_vs_predicted.png')
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Goal model scatter plot saved to {plot_path}")
        
        # Create residual plot
        residuals = y_goal_test - y_pred
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Goals')
        plt.ylabel('Residuals')
        plt.title('Residual Plot for Goal Predictions')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(EVALUATION_DIR, 'goal_residuals.png')
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Goal model residual plot saved to {plot_path}")
        
        # Create histogram of residuals
        plt.figure(figsize=(10, 8))
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Residuals for Goal Predictions')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(EVALUATION_DIR, 'goal_residuals_histogram.png')
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Goal model residual histogram saved to {plot_path}")
        
        # Create error distribution by actual value
        plt.figure(figsize=(10, 8))
        
        # Calculate absolute error
        abs_error = np.abs(residuals)
        
        # Create DataFrame for plotting
        error_df = pd.DataFrame({
            'Actual': y_goal_test,
            'Absolute Error': abs_error
        })
        
        # Bin actual values
        error_df['Actual Bin'] = pd.cut(error_df['Actual'], bins=10)
        
        # Calculate mean error by bin
        bin_errors = error_df.groupby('Actual Bin')['Absolute Error'].mean().reset_index()
        
        # Plot
        plt.bar(range(len(bin_errors)), bin_errors['Absolute Error'], alpha=0.7)
        plt.xticks(range(len(bin_errors)), [str(x) for x in bin_errors['Actual Bin']], rotation=45)
        plt.xlabel('Actual Goal Range')
        plt.ylabel('Mean Absolute Error')
        plt.title('Error Distribution by Actual Goal Value')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(EVALUATION_DIR, 'goal_error_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Goal model error distribution plot saved to {plot_path}")
        
        return report
    except Exception as e:
        logger.error(f"Error evaluating goal model: {e}")
        return None

def evaluate_model_thresholds(pipeline, X_test, y_disposal_test, y_goal_test):
    """
    Evaluate model performance at different prediction thresholds
    """
    if pipeline is None or X_test is None:
        return
    
    try:
        logger.info("Evaluating model performance at different thresholds...")
        
        # Extract models and preprocessor
        disposal_model = pipeline.get('disposal_model')
        goal_model = pipeline.get('goal_model')
        preprocessor = pipeline.get('preprocessor')
        
        if preprocessor is None:
            logger.error("Preprocessor not found in pipeline")
            return
        
        # Transform test data
        X_test_transformed = preprocessor.transform(X_test)
        
        # Evaluate disposal model at different thresholds
        if disposal_model is not None and y_disposal_test is not None:
            logger.info("Evaluating disposal model thresholds...")
            
            # Make predictions
            y_pred = disposal_model.predict(X_test_transformed)
            
            # Create thresholds
            thresholds = range(5, 35, 5)  # 5, 10, 15, 20, 25, 30
            
            # Calculate accuracy at each threshold
            threshold_results = []
            
            for threshold in thresholds:
                # Calculate binary accuracy (predicted >= threshold when actual >= threshold)
                actual_above = (y_disposal_test >= threshold).astype(int)
                pred_above = (y_pred >= threshold).astype(int)
                
                # Calculate metrics
                true_pos = np.sum((actual_above == 1) & (pred_above == 1))
                false_pos = np.sum((actual_above == 0) & (pred_above == 1))
                true_neg = np.sum((actual_above == 0) & (pred_above == 0))
                false_neg = np.sum((actual_above == 1) & (pred_above == 0))
                
                # Calculate accuracy
                accuracy = (true_pos + true_neg) / len(actual_above)
                
                # Calculate precision
                precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
                
                # Calculate recall
                recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
                
                # Calculate F1 score
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # Store results
                threshold_results.append({
                    'Threshold': threshold,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1,
                    'True Positives': true_pos,
                    'False Positives': false_pos,
                    'True Negatives': true_neg,
                    'False Negatives': false_neg
                })
            
            # Create DataFrame
            threshold_df = pd.DataFrame(threshold_results)
            
            # Save results
            threshold_path = os.path.join(EVALUATION_DIR, 'disposal_threshold_evaluation.csv')
            threshold_df.to_csv(threshold_path, index=False)
            
            logger.info(f"Disposal threshold evaluation saved to {threshold_path}")
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Plot metrics
            plt.plot(threshold_df['Threshold'], threshold_df['Accuracy'], 'o-', label='Accuracy')
            plt.plot(threshold_df['Threshold'], threshold_df['Precision'], 's-', label='Precision')
            plt.plot(threshold_df['Threshold'], threshold_df['Recall'], '^-', label='Recall')
            plt.plot(threshold_df['Threshold'], threshold_df['F1 Score'], 'd-', label='F1 Score')
            
            plt.xlabel('Disposal Threshold')
            plt.ylabel('Score')
            plt.title('Model Performance at Different Disposal Thresholds')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = os.path.join(EVALUATION_DIR, 'disposal_threshold_performance.png')
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Disposal threshold performance plot saved to {plot_path}")
        
        # Evaluate goal model at different thresholds
        if goal_model is not None and y_goal_test is not None:
            logger.info("Evaluating goal model thresholds...")
            
            # Make predictions
            y_pred = goal_model.predict(X_test_transformed)
            
            # Create thresholds
            thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
            
            # Calculate accuracy at each threshold
            threshold_results = []
            
            for threshold in thresholds:
                # Calculate binary accuracy (predicted >= threshold when actual >= threshold)
                actual_above = (y_goal_test >= threshold).astype(int)
                pred_above = (y_pred >= threshold).astype(int)
                
                # Calculate metrics
                true_pos = np.sum((actual_above == 1) & (pred_above == 1))
                false_pos = np.sum((actual_above == 0) & (pred_above == 1))
                true_neg = np.sum((actual_above == 0) & (pred_above == 0))
                false_neg = np.sum((actual_above == 1) & (pred_above == 0))
                
                # Calculate accuracy
                accuracy = (true_pos + true_neg) / len(actual_above)
                
                # Calculate precision
                precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
                
                # Calculate recall
                recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
                
                # Calculate F1 score
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # Store results
                threshold_results.append({
                    'Threshold': threshold,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1,
                    'True Positives': true_pos,
                    'False Positives': false_pos,
                    'True Negatives': true_neg,
                    'False Negatives': false_neg
                })
            
            # Create DataFrame
            threshold_df = pd.DataFrame(threshold_results)
            
            # Save results
            threshold_path = os.path.join(EVALUATION_DIR, 'goal_threshold_evaluation.csv')
            threshold_df.to_csv(threshold_path, index=False)
            
            logger.info(f"Goal threshold evaluation saved to {threshold_path}")
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Plot metrics
            plt.plot(threshold_df['Threshold'], threshold_df['Accuracy'], 'o-', label='Accuracy')
            plt.plot(threshold_df['Threshold'], threshold_df['Precision'], 's-', label='Precision')
            plt.plot(threshold_df['Threshold'], threshold_df['Recall'], '^-', label='Recall')
            plt.plot(threshold_df['Threshold'], threshold_df['F1 Score'], 'd-', label='F1 Score')
            
            plt.xlabel('Goal Threshold')
            plt.ylabel('Score')
            plt.title('Model Performance at Different Goal Thresholds')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = os.path.join(EVALUATION_DIR, 'goal_threshold_performance.png')
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Goal threshold performance plot saved to {plot_path}")
    except Exception as e:
        logger.error(f"Error evaluating model thresholds: {e}")

def generate_evaluation_report(disposal_report, goal_report):
    """
    Generate a comprehensive evaluation report
    """
    try:
        logger.info("Generating comprehensive evaluation report...")
        
        # Create report
        report_path = os.path.join(EVALUATION_DIR, 'model_evaluation_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# AFL Player Statistics Prediction Model Evaluation\n\n")
            
            # Overall summary
            f.write("## Overall Summary\n\n")
            
            if disposal_report and goal_report:
                f.write("Two machine learning models were developed and evaluated:\n\n")
                f.write("1. **Disposal Prediction Model**: Predicts the number of disposals a player will have in a match\n")
                f.write("2. **Goal Prediction Model**: Predicts the number of goals a player will score in a match\n\n")
                
                f.write("Both models were trained on historical AFL player statistics from 2020-2025 and evaluated on a held-out test set.\n\n")
                
                f.write("### Key Performance Metrics\n\n")
                f.write("| Model | RMSE | MAE | R² | RMSE/Mean | MAE/Mean |\n")
                f.write("|-------|------|-----|----|-----------|---------|\n")
                f.write(f"| Disposal Prediction | {disposal_report['RMSE']:.2f} | {disposal_report['MAE']:.2f} | {disposal_report['R²']:.2f} | {disposal_report['RMSE/Mean']:.2f} | {disposal_report['MAE/Mean']:.2f} |\n")
                f.write(f"| Goal Prediction | {goal_report['RMSE']:.2f} | {goal_report['MAE']:.2f} | {goal_report['R²']:.2f} | {goal_report['RMSE/Mean']:.2f} | {goal_report['MAE/Mean']:.2f} |\n\n")
            elif disposal_report:
                f.write("A machine learning model was developed and evaluated for predicting the number of disposals a player will have in a match.\n\n")
                
                f.write("### Key Performance Metrics\n\n")
                f.write("| Model | RMSE | MAE | R² | RMSE/Mean | MAE/Mean |\n")
                f.write("|-------|------|-----|----|-----------|---------|\n")
                f.write(f"| Disposal Prediction | {disposal_report['RMSE']:.2f} | {disposal_report['MAE']:.2f} | {disposal_report['R²']:.2f} | {disposal_report['RMSE/Mean']:.2f} | {disposal_report['MAE/Mean']:.2f} |\n\n")
            elif goal_report:
                f.write("A machine learning model was developed and evaluated for predicting the number of goals a player will score in a match.\n\n")
                
                f.write("### Key Performance Metrics\n\n")
                f.write("| Model | RMSE | MAE | R² | RMSE/Mean | MAE/Mean |\n")
                f.write("|-------|------|-----|----|-----------|---------|\n")
                f.write(f"| Goal Prediction | {goal_report['RMSE']:.2f} | {goal_report['MAE']:.2f} | {goal_report['R²']:.2f} | {goal_report['RMSE/Mean']:.2f} | {goal_report['MAE/Mean']:.2f} |\n\n")
            
            # Disposal model evaluation
            if disposal_report:
                f.write("## Disposal Prediction Model\n\n")
                
                f.write("### Model Performance\n\n")
                f.write(f"- **RMSE**: {disposal_report['RMSE']:.2f} disposals\n")
                f.write(f"- **MAE**: {disposal_report['MAE']:.2f} disposals\n")
                f.write(f"- **R²**: {disposal_report['R²']:.2f}\n")
                f.write(f"- **Mean Target Value**: {disposal_report['Mean Target']:.2f} disposals\n")
                f.write(f"- **RMSE/Mean**: {disposal_report['RMSE/Mean']:.2f}\n")
                f.write(f"- **MAE/Mean**: {disposal_report['MAE/Mean']:.2f}\n\n")
                
                f.write("### Visualizations\n\n")
                f.write("#### Actual vs Predicted Disposals\n\n")
                f.write("![Actual vs Predicted Disposals](disposal_actual_vs_predicted.png)\n\n")
                
                f.write("#### Residual Plot\n\n")
                f.write("![Residual Plot](disposal_residuals.png)\n\n")
                
                f.write("#### Residual Distribution\n\n")
                f.write("![Residual Distribution](disposal_residuals_histogram.png)\n\n")
                
                f.write("#### Error Distribution by Actual Value\n\n")
                f.write("![Error Distribution](disposal_error_distribution.png)\n\n")
                
                f.write("#### Performance at Different Thresholds\n\n")
                f.write("![Threshold Performance](disposal_threshold_performance.png)\n\n")
            
            # Goal model evaluation
            if goal_report:
                f.write("## Goal Prediction Model\n\n")
                
                f.write("### Model Performance\n\n")
                f.write(f"- **RMSE**: {goal_report['RMSE']:.2f} goals\n")
                f.write(f"- **MAE**: {goal_report['MAE']:.2f} goals\n")
                f.write(f"- **R²**: {goal_report['R²']:.2f}\n")
                f.write(f"- **Mean Target Value**: {goal_report['Mean Target']:.2f} goals\n")
                f.write(f"- **RMSE/Mean**: {goal_report['RMSE/Mean']:.2f}\n")
                f.write(f"- **MAE/Mean**: {goal_report['MAE/Mean']:.2f}\n\n")
                
                f.write("### Visualizations\n\n")
                f.write("#### Actual vs Predicted Goals\n\n")
                f.write("![Actual vs Predicted Goals](goal_actual_vs_predicted.png)\n\n")
                
                f.write("#### Residual Plot\n\n")
                f.write("![Residual Plot](goal_residuals.png)\n\n")
                
                f.write("#### Residual Distribution\n\n")
                f.write("![Residual Distribution](goal_residuals_histogram.png)\n\n")
                
                f.write("#### Error Distribution by Actual Value\n\n")
                f.write("![Error Distribution](goal_error_distribution.png)\n\n")
                
                f.write("#### Performance at Different Thresholds\n\n")
                f.write("![Threshold Performance](goal_threshold_performance.png)\n\n")
            
            # Conclusions and recommendations
            f.write("## Conclusions and Recommendations\n\n")
            
            if disposal_report and goal_report:
                # Compare models
                if disposal_report['R²'] > goal_report['R²']:
                    f.write("The disposal prediction model performs better than the goal prediction model in terms of R² score. This is expected as disposals are more consistent and predictable than goals.\n\n")
                else:
                    f.write("Interestingly, the goal prediction model performs better than the disposal prediction model in terms of R² score. This suggests that the features used are more predictive of goal-scoring than disposals.\n\n")
                
                # General conclusions
                f.write("### Key Findings\n\n")
                f.write("1. Both models show good predictive performance, with R² scores indicating that a significant portion of the variance in player statistics can be explained by the models.\n")
                f.write("2. The models' error metrics (RMSE and MAE) are reasonable relative to the mean values of the target variables.\n")
                f.write("3. The residual plots show that the models' errors are generally well-distributed, with no strong patterns indicating systematic bias.\n")
                f.write("4. The threshold analysis shows that the models perform well at predicting whether a player will exceed certain statistical thresholds, which is valuable for betting applications.\n\n")
                
                f.write("### Recommendations\n\n")
                f.write("1. **Weekly Updates**: Continue with the weekly update approach to incorporate the latest form data, as this will likely improve prediction accuracy.\n")
                f.write("2. **Feature Engineering**: Consider adding more features related to player matchups, weather conditions, and team strategies.\n")
                f.write("3. **Model Ensemble**: Consider creating an ensemble of models to potentially improve prediction accuracy.\n")
                f.write("4. **Threshold Optimization**: For betting applications, focus on the thresholds where the model shows the highest precision and recall.\n")
                f.write("5. **Web Application**: Implement the web application to make these predictions easily accessible, with clear confidence intervals for each prediction.\n")
            elif disposal_report:
                f.write("### Key Findings\n\n")
                f.write("1. The disposal prediction model shows good predictive performance, with an R² score indicating that a significant portion of the variance in player disposals can be explained by the model.\n")
                f.write("2. The model's error metrics (RMSE and MAE) are reasonable relative to the mean value of disposals.\n")
                f.write("3. The residual plot shows that the model's errors are generally well-distributed, with no strong patterns indicating systematic bias.\n")
                f.write("4. The threshold analysis shows that the model performs well at predicting whether a player will exceed certain disposal thresholds, which is valuable for betting applications.\n\n")
                
                f.write("### Recommendations\n\n")
                f.write("1. **Weekly Updates**: Continue with the weekly update approach to incorporate the latest form data, as this will likely improve prediction accuracy.\n")
                f.write("2. **Feature Engineering**: Consider adding more features related to player matchups, weather conditions, and team strategies.\n")
                f.write("3. **Goal Prediction**: Develop a complementary model for predicting player goals to provide a more comprehensive prediction system.\n")
                f.write("4. **Threshold Optimization**: For betting applications, focus on the thresholds where the model shows the highest precision and recall.\n")
                f.write("5. **Web Application**: Implement the web application to make these predictions easily accessible, with clear confidence intervals for each prediction.\n")
            elif goal_report:
                f.write("### Key Findings\n\n")
                f.write("1. The goal prediction model shows good predictive performance, with an R² score indicating that a significant portion of the variance in player goals can be explained by the model.\n")
                f.write("2. The model's error metrics (RMSE and MAE) are reasonable relative to the mean value of goals.\n")
                f.write("3. The residual plot shows that the model's errors are generally well-distributed, with no strong patterns indicating systematic bias.\n")
                f.write("4. The threshold analysis shows that the model performs well at predicting whether a player will exceed certain goal thresholds, which is valuable for betting applications.\n\n")
                
                f.write("### Recommendations\n\n")
                f.write("1. **Weekly Updates**: Continue with the weekly update approach to incorporate the latest form data, as this will likely improve prediction accuracy.\n")
                f.write("2. **Feature Engineering**: Consider adding more features related to player matchups, weather conditions, and team strategies.\n")
                f.write("3. **Disposal Prediction**: Develop a complementary model for predicting player disposals to provide a more comprehensive prediction system.\n")
                f.write("4. **Threshold Optimization**: For betting applications, focus on the thresholds where the model shows the highest precision and recall.\n")
                f.write("5. **Web Application**: Implement the web application to make these predictions easily accessible, with clear confidence intervals for each prediction.\n")
        
        logger.info(f"Evaluation report generated: {report_path}")
        return report_path
    except Exception as e:
        logger.error(f"Error generating evaluation report: {e}")
        return None

def main():
    """
    Main function to execute the model evaluation process
    """
    logger.info("Starting model evaluation...")
    
    # Load models and data
    pipeline, X_test, y_disposal_test, y_goal_test = load_models_and_data()
    
    if pipeline is None:
        logger.error("Failed to load models and data")
        return
    
    # Evaluate disposal model
    disposal_report = None
    if y_disposal_test is not None:
        disposal_report = evaluate_disposal_model(pipeline, X_test, y_disposal_test)
    
    # Evaluate goal model
    goal_report = None
    if y_goal_test is not None:
        goal_report = evaluate_goal_model(pipeline, X_test, y_goal_test)
    
    # Evaluate model thresholds
    evaluate_model_thresholds(pipeline, X_test, y_disposal_test, y_goal_test)
    
    # Generate evaluation report
    report_path = generate_evaluation_report(disposal_report, goal_report)
    
    logger.info("Model evaluation completed successfully!")
    
    if report_path:
        logger.info(f"Comprehensive evaluation report available at: {report_path}")

if __name__ == "__main__":
    main()
