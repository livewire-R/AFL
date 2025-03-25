#!/usr/bin/env python3
"""
AFL Player Statistics Feature Analysis Script

This script analyzes the processed AFL player statistics data to identify
the most important features for predicting disposals and goals.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

# Define paths
BASE_DIR = '/home/ubuntu/afl_prediction_project'
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data/processed')
ANALYSIS_DIR = os.path.join(BASE_DIR, 'data/analysis')

# Ensure analysis directory exists
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

def analyze_feature_importance(df, target_col, top_n=15):
    """
    Analyze feature importance for a target column
    """
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in data")
        return None
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_df = df[numeric_cols].copy()
    
    # Remove the target column from features
    features = numeric_df.drop(columns=[target_col], errors='ignore')
    
    # Handle missing values
    features = features.fillna(0)
    target = numeric_df[target_col].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Select top features
    selector = SelectKBest(score_func=f_regression, k=top_n)
    selector.fit(features_scaled, target)
    
    # Get feature scores
    feature_scores = pd.DataFrame({
        'Feature': features.columns,
        'Score': selector.scores_
    })
    
    # Sort by score in descending order
    feature_scores = feature_scores.sort_values('Score', ascending=False)
    
    return feature_scores

def plot_feature_importance(feature_scores, target_name, top_n=15):
    """
    Plot feature importance scores
    """
    plt.figure(figsize=(12, 8))
    
    # Take top N features
    top_features = feature_scores.head(top_n)
    
    # Create horizontal bar plot
    sns.barplot(x='Score', y='Feature', data=top_features)
    
    plt.title(f'Top {top_n} Features for Predicting {target_name}', fontsize=16)
    plt.xlabel('Importance Score', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(ANALYSIS_DIR, f'feature_importance_{target_name.lower().replace(" ", "_")}.png')
    plt.savefig(plot_path)
    print(f"Feature importance plot saved to {plot_path}")
    
    return plot_path

def analyze_correlations(df, target_col, top_n=15):
    """
    Analyze correlations between features and target
    """
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in data")
        return None
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_df = df[numeric_cols].copy()
    
    # Calculate correlations with target
    correlations = numeric_df.corr()[target_col].sort_values(ascending=False)
    
    # Create a DataFrame for the correlations
    corr_df = pd.DataFrame({
        'Feature': correlations.index,
        'Correlation': correlations.values
    })
    
    return corr_df

def plot_correlation_heatmap(df, target_cols, top_n=15):
    """
    Plot correlation heatmap for top features
    """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_df = df[numeric_cols].copy()
    
    # Get top correlated features for each target
    top_features = set()
    for target_col in target_cols:
        if target_col in numeric_df.columns:
            correlations = numeric_df.corr()[target_col].sort_values(ascending=False)
            top_features.update(correlations.head(top_n).index)
    
    # Add target columns to the set
    top_features.update(target_cols)
    
    # Convert to list and filter out None values
    top_features = [f for f in top_features if f is not None]
    
    # Create correlation matrix for top features
    corr_matrix = numeric_df[top_features].corr()
    
    # Plot heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    
    plt.title('Correlation Heatmap of Top Features', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(ANALYSIS_DIR, 'correlation_heatmap.png')
    plt.savefig(plot_path)
    print(f"Correlation heatmap saved to {plot_path}")
    
    return plot_path

def analyze_form_metrics(df, target_cols):
    """
    Analyze how form metrics could be incorporated
    """
    # Check if we have player and match/game columns
    player_col = None
    for col in ['player', 'player_name', 'name']:
        if col in df.columns:
            player_col = col
            break
    
    match_col = None
    for col in ['match', 'game', 'gm', 'match_id']:
        if col in df.columns:
            match_col = col
            break
    
    if player_col is None or match_col is None:
        print("Cannot analyze form metrics: missing player or match columns")
        return
    
    # Check if we have date or round columns
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
    
    time_col = date_col if date_col is not None else round_col
    
    if time_col is None:
        print("Cannot analyze form metrics: missing date or round columns")
        return
    
    # Create a report on form metrics
    form_report_path = os.path.join(ANALYSIS_DIR, 'form_metrics_analysis.txt')
    with open(form_report_path, 'w') as f:
        f.write("Form Metrics Analysis for AFL Prediction Model\n")
        f.write("=============================================\n\n")
        
        f.write("Identified columns for form calculation:\n")
        f.write(f"- Player column: {player_col}\n")
        f.write(f"- Match column: {match_col}\n")
        f.write(f"- Time column: {time_col}\n\n")
        
        f.write("Proposed form metrics to calculate:\n")
        f.write("1. Rolling average of disposals (last 3, 5, and 10 games)\n")
        f.write("2. Rolling average of goals (last 3, 5, and 10 games)\n")
        f.write("3. Consistency score (standard deviation of recent performances)\n")
        f.write("4. Form trend (positive or negative based on recent games)\n")
        f.write("5. Performance against upcoming opponent (historical average)\n\n")
        
        f.write("Implementation approach:\n")
        f.write("- Sort data by player and date/round\n")
        f.write("- Group by player\n")
        f.write("- Calculate rolling statistics for key metrics\n")
        f.write("- Join these form metrics back to the main dataset\n")
        f.write("- Include these metrics as features in the prediction model\n\n")
        
        f.write("Weekly update process:\n")
        f.write("1. Fetch latest match results\n")
        f.write("2. Update player statistics database\n")
        f.write("3. Recalculate form metrics for all players\n")
        f.write("4. Generate predictions for upcoming matches\n")
    
    print(f"Form metrics analysis saved to {form_report_path}")
    return form_report_path

def main():
    """
    Main function to execute the feature analysis
    """
    print("Starting AFL player statistics feature analysis...")
    
    # Load processed data
    df = load_processed_data()
    
    if df is None:
        return
    
    # Identify target columns
    disposal_col, goal_col = identify_target_columns(df)
    target_cols = [col for col in [disposal_col, goal_col] if col is not None]
    
    # Analyze feature importance for disposals
    if disposal_col is not None:
        print(f"\nAnalyzing feature importance for {disposal_col}...")
        disposal_importance = analyze_feature_importance(df, disposal_col)
        if disposal_importance is not None:
            print("\nTop features for predicting disposals:")
            print(disposal_importance.head(15))
            plot_feature_importance(disposal_importance, "Disposals")
    
    # Analyze feature importance for goals
    if goal_col is not None:
        print(f"\nAnalyzing feature importance for {goal_col}...")
        goal_importance = analyze_feature_importance(df, goal_col)
        if goal_importance is not None:
            print("\nTop features for predicting goals:")
            print(goal_importance.head(15))
            plot_feature_importance(goal_importance, "Goals")
    
    # Plot correlation heatmap
    if target_cols:
        print("\nCreating correlation heatmap...")
        plot_correlation_heatmap(df, target_cols)
    
    # Analyze form metrics
    print("\nAnalyzing potential form metrics...")
    analyze_form_metrics(df, target_cols)
    
    print("\nFeature analysis completed successfully!")

if __name__ == "__main__":
    main()
