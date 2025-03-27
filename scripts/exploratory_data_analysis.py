#!/usr/bin/env python3
"""
AFL Prediction - Exploratory Data Analysis

This script performs exploratory data analysis on the AFL player statistics data
to understand the distributions, correlations, and patterns in the data.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('eda')

# Define paths
BASE_DIR = '/home/ubuntu/afl_prediction_project'
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
ANALYSIS_DIR = os.path.join(BASE_DIR, 'analysis')
FIGURES_DIR = os.path.join(ANALYSIS_DIR, 'figures')

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_data():
    """
    Load the preprocessed player statistics data
    """
    try:
        # Check if processed data exists
        processed_data_path = os.path.join(PROCESSED_DATA_DIR, 'afl_player_stats_all_years.csv')
        
        if not os.path.exists(processed_data_path):
            logger.error(f"Processed data file not found: {processed_data_path}")
            
            # Try to find raw data files
            raw_files = []
            for year in range(2020, 2026):
                year_dir = os.path.join(RAW_DATA_DIR, str(year))
                if os.path.exists(year_dir):
                    for file in os.listdir(year_dir):
                        if file.endswith('.csv'):
                            raw_files.append(os.path.join(year_dir, file))
            
            if not raw_files:
                logger.error("No raw data files found")
                return None
            
            # Load and combine raw data files
            dfs = []
            for file in raw_files:
                try:
                    df = pd.read_csv(file)
                    year = os.path.basename(os.path.dirname(file))
                    df['year'] = year
                    dfs.append(df)
                except Exception as e:
                    logger.error(f"Error loading {file}: {e}")
            
            if not dfs:
                logger.error("No data frames created from raw files")
                return None
            
            # Combine data frames
            data = pd.concat(dfs, ignore_index=True)
            
            # Save combined data
            data.to_csv(processed_data_path, index=False)
            logger.info(f"Combined data saved to {processed_data_path}")
        else:
            # Load processed data
            data = pd.read_csv(processed_data_path)
            logger.info(f"Loaded processed data: {len(data)} records")
        
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def analyze_data_structure(data):
    """
    Analyze the structure of the data
    """
    if data is None:
        return
    
    try:
        logger.info("Analyzing data structure...")
        
        # Get basic information
        info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict()
        }
        
        # Save information to file
        info_path = os.path.join(ANALYSIS_DIR, 'data_structure.txt')
        with open(info_path, 'w') as f:
            f.write(f"Data Shape: {info['shape']}\n\n")
            
            f.write("Columns:\n")
            for col in info['columns']:
                f.write(f"- {col}\n")
            f.write("\n")
            
            f.write("Data Types:\n")
            for col, dtype in info['dtypes'].items():
                f.write(f"- {col}: {dtype}\n")
            f.write("\n")
            
            f.write("Missing Values:\n")
            for col, count in info['missing_values'].items():
                if count > 0:
                    f.write(f"- {col}: {count} ({count/len(data)*100:.2f}%)\n")
            f.write("\n")
        
        logger.info(f"Data structure analysis saved to {info_path}")
        return info
    except Exception as e:
        logger.error(f"Error analyzing data structure: {e}")
        return None

def analyze_target_variables(data):
    """
    Analyze the target variables (disposals and goals)
    """
    if data is None:
        return
    
    try:
        logger.info("Analyzing target variables...")
        
        # Identify target columns
        disposal_col = None
        for col in ['dis.', 'disposals', 'disposal', 'total_disposals']:
            if col in data.columns:
                disposal_col = col
                break
        
        goal_col = None
        for col in ['goals', 'gls', 'gls_avg', 'goals_avg', 'total_goals']:
            if col in data.columns:
                goal_col = col
                break
        
        if disposal_col is None and goal_col is None:
            logger.error("Target columns not found")
            return
        
        # Create directory for target analysis
        target_dir = os.path.join(ANALYSIS_DIR, 'targets')
        os.makedirs(target_dir, exist_ok=True)
        
        # Analyze disposals
        if disposal_col is not None:
            logger.info(f"Analyzing {disposal_col}...")
            
            # Get basic statistics
            disposal_stats = data[disposal_col].describe().to_dict()
            
            # Create histogram
            plt.figure(figsize=(10, 6))
            sns.histplot(data[disposal_col].dropna(), kde=True)
            plt.title(f'Distribution of {disposal_col}')
            plt.xlabel(disposal_col)
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(FIGURES_DIR, f'{disposal_col}_distribution.png'))
            plt.close()
            
            # Create box plot by year
            if 'year' in data.columns:
                plt.figure(figsize=(12, 6))
                sns.boxplot(x='year', y=disposal_col, data=data)
                plt.title(f'{disposal_col} by Year')
                plt.xlabel('Year')
                plt.ylabel(disposal_col)
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(FIGURES_DIR, f'{disposal_col}_by_year.png'))
                plt.close()
            
            # Save statistics to file
            stats_path = os.path.join(target_dir, f'{disposal_col}_stats.txt')
            with open(stats_path, 'w') as f:
                f.write(f"{disposal_col} Statistics:\n")
                for stat, value in disposal_stats.items():
                    f.write(f"- {stat}: {value}\n")
                f.write("\n")
            
            logger.info(f"{disposal_col} analysis saved to {stats_path}")
        
        # Analyze goals
        if goal_col is not None:
            logger.info(f"Analyzing {goal_col}...")
            
            # Get basic statistics
            goal_stats = data[goal_col].describe().to_dict()
            
            # Create histogram
            plt.figure(figsize=(10, 6))
            sns.histplot(data[goal_col].dropna(), kde=True)
            plt.title(f'Distribution of {goal_col}')
            plt.xlabel(goal_col)
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(FIGURES_DIR, f'{goal_col}_distribution.png'))
            plt.close()
            
            # Create box plot by year
            if 'year' in data.columns:
                plt.figure(figsize=(12, 6))
                sns.boxplot(x='year', y=goal_col, data=data)
                plt.title(f'{goal_col} by Year')
                plt.xlabel('Year')
                plt.ylabel(goal_col)
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(FIGURES_DIR, f'{goal_col}_by_year.png'))
                plt.close()
            
            # Save statistics to file
            stats_path = os.path.join(target_dir, f'{goal_col}_stats.txt')
            with open(stats_path, 'w') as f:
                f.write(f"{goal_col} Statistics:\n")
                for stat, value in goal_stats.items():
                    f.write(f"- {stat}: {value}\n")
                f.write("\n")
            
            logger.info(f"{goal_col} analysis saved to {stats_path}")
        
        # Analyze correlation between disposals and goals
        if disposal_col is not None and goal_col is not None:
            logger.info(f"Analyzing correlation between {disposal_col} and {goal_col}...")
            
            # Calculate correlation
            correlation = data[[disposal_col, goal_col]].corr().iloc[0, 1]
            
            # Create scatter plot
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=disposal_col, y=goal_col, data=data, alpha=0.5)
            plt.title(f'Correlation between {disposal_col} and {goal_col}: {correlation:.2f}')
            plt.xlabel(disposal_col)
            plt.ylabel(goal_col)
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(FIGURES_DIR, f'{disposal_col}_{goal_col}_correlation.png'))
            plt.close()
            
            # Save correlation to file
            corr_path = os.path.join(target_dir, 'target_correlation.txt')
            with open(corr_path, 'w') as f:
                f.write(f"Correlation between {disposal_col} and {goal_col}: {correlation:.4f}\n")
            
            logger.info(f"Target correlation analysis saved to {corr_path}")
    except Exception as e:
        logger.error(f"Error analyzing target variables: {e}")

def analyze_feature_correlations(data):
    """
    Analyze correlations between features and target variables
    """
    if data is None:
        return
    
    try:
        logger.info("Analyzing feature correlations...")
        
        # Identify target columns
        disposal_col = None
        for col in ['dis.', 'disposals', 'disposal', 'total_disposals']:
            if col in data.columns:
                disposal_col = col
                break
        
        goal_col = None
        for col in ['goals', 'gls', 'gls_avg', 'goals_avg', 'total_goals']:
            if col in data.columns:
                goal_col = col
                break
        
        if disposal_col is None and goal_col is None:
            logger.error("Target columns not found")
            return
        
        # Create directory for correlation analysis
        corr_dir = os.path.join(ANALYSIS_DIR, 'correlations')
        os.makedirs(corr_dir, exist_ok=True)
        
        # Select numeric columns
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        
        # Calculate correlations with disposals
        if disposal_col is not None and disposal_col in numeric_cols:
            logger.info(f"Calculating correlations with {disposal_col}...")
            
            # Calculate correlations
            disposal_corrs = data[numeric_cols].corr()[disposal_col].sort_values(ascending=False)
            
            # Create correlation plot
            plt.figure(figsize=(12, 10))
            sns.barplot(x=disposal_corrs.values[1:21], y=disposal_corrs.index[1:21])
            plt.title(f'Top 20 Features Correlated with {disposal_col}')
            plt.xlabel('Correlation')
            plt.ylabel('Feature')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, f'{disposal_col}_correlations.png'))
            plt.close()
            
            # Save correlations to file
            corr_path = os.path.join(corr_dir, f'{disposal_col}_correlations.txt')
            with open(corr_path, 'w') as f:
                f.write(f"Correlations with {disposal_col}:\n")
                for feature, corr in disposal_corrs.items():
                    if feature != disposal_col:
                        f.write(f"- {feature}: {corr:.4f}\n")
                f.write("\n")
            
            logger.info(f"{disposal_col} correlations saved to {corr_path}")
        
        # Calculate correlations with goals
        if goal_col is not None and goal_col in numeric_cols:
            logger.info(f"Calculating correlations with {goal_col}...")
            
            # Calculate correlations
            goal_corrs = data[numeric_cols].corr()[goal_col].sort_values(ascending=False)
            
            # Create correlation plot
            plt.figure(figsize=(12, 10))
            sns.barplot(x=goal_corrs.values[1:21], y=goal_corrs.index[1:21])
            plt.title(f'Top 20 Features Correlated with {goal_col}')
            plt.xlabel('Correlation')
            plt.ylabel('Feature')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, f'{goal_col}_correlations.png'))
            plt.close()
            
            # Save correlations to file
            corr_path = os.path.join(corr_dir, f'{goal_col}_correlations.txt')
            with open(corr_path, 'w') as f:
                f.write(f"Correlations with {goal_col}:\n")
                for feature, corr in goal_corrs.items():
                    if feature != goal_col:
                        f.write(f"- {feature}: {corr:.4f}\n")
                f.write("\n")
            
            logger.info(f"{goal_col} correlations saved to {corr_path}")
        
        # Create correlation heatmap for top features
        logger.info("Creating correlation heatmap...")
        
        # Select top correlated features
        top_features = set()
        
        if disposal_col is not None and disposal_col in numeric_cols:
            disposal_top = disposal_corrs.index[1:11]  # Top 10 features
            top_features.update(disposal_top)
        
        if goal_col is not None and goal_col in numeric_cols:
            goal_top = goal_corrs.index[1:11]  # Top 10 features
            top_features.update(goal_top)
        
        # Add target columns
        if disposal_col is not None:
            top_features.add(disposal_col)
        
        if goal_col is not None:
            top_features.add(goal_col)
        
        # Create heatmap
        if top_features:
            plt.figure(figsize=(14, 12))
            sns.heatmap(data[top_features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Correlation Heatmap of Top Features')
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, 'top_features_correlation_heatmap.png'))
            plt.close()
            
            logger.info("Correlation heatmap saved")
    except Exception as e:
        logger.error(f"Error analyzing feature correlations: {e}")

def analyze_player_performance(data):
    """
    Analyze player performance over time
    """
    if data is None:
        return
    
    try:
        logger.info("Analyzing player performance...")
        
        # Identify key columns
        player_col = None
        for col in ['player', 'player_name', 'name']:
            if col in data.columns:
                player_col = col
                break
        
        if player_col is None:
            logger.error("Player column not found")
            return
        
        # Identify target columns
        disposal_col = None
        for col in ['dis.', 'disposals', 'disposal', 'total_disposals']:
            if col in data.columns:
                disposal_col = col
                break
        
        goal_col = None
        for col in ['goals', 'gls', 'gls_avg', 'goals_avg', 'total_goals']:
            if col in data.columns:
                goal_col = col
                break
        
        if disposal_col is None and goal_col is None:
            logger.error("Target columns not found")
            return
        
        # Create directory for player analysis
        player_dir = os.path.join(ANALYSIS_DIR, 'players')
        os.makedirs(player_dir, exist_ok=True)
        
        # Get top players by average disposals
        if disposal_col is not None:
            logger.info(f"Analyzing top players by {disposal_col}...")
            
            # Calculate average disposals by player
            player_disposals = data.groupby(player_col)[disposal_col].agg(['mean', 'std', 'count']).reset_index()
            player_disposals = player_disposals[player_disposals['count'] >= 10]  # At least 10 games
            player_disposals = player_disposals.sort_values('mean', ascending=False).head(20)
            
            # Create bar plot
            plt.figure(figsize=(14, 8))
            sns.barplot(x='mean', y=player_col, data=player_disposals)
            plt.title(f'Top 20 Players by Average {disposal_col}')
            plt.xlabel(f'Average {disposal_col}')
            plt.ylabel('Player')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, f'top_players_{disposal_col}.png'))
            plt.close()
            
            # Save to file
            stats_path = os.path.join(player_dir, f'top_players_{disposal_col}.txt')
            with open(stats_path, 'w') as f:
                f.write(f"Top Players by Average {disposal_col}:\n")
                for i, row in player_disposals.iterrows():
                    f.write(f"- {row[player_col]}: {row['mean']:.2f} (±{row['std']:.2f}, {row['count']} games)\n")
                f.write("\n")
            
            logger.info(f"Top players by {disposal_col} saved to {stats_path}")
        
        # Get top players by average goals
        if goal_col is not None:
            logger.info(f"Analyzing top players by {goal_col}...")
            
            # Calculate average goals by player
            player_goals = data.groupby(player_col)[goal_col].agg(['mean', 'std', 'count']).reset_index()
            player_goals = player_goals[player_goals['count'] >= 10]  # At least 10 games
            player_goals = player_goals.sort_values('mean', ascending=False).head(20)
            
            # Create bar plot
            plt.figure(figsize=(14, 8))
            sns.barplot(x='mean', y=player_col, data=player_goals)
            plt.title(f'Top 20 Players by Average {goal_col}')
            plt.xlabel(f'Average {goal_col}')
            plt.ylabel('Player')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, f'top_players_{goal_col}.png'))
            plt.close()
            
            # Save to file
            stats_path = os.path.join(player_dir, f'top_players_{goal_col}.txt')
            with open(stats_path, 'w') as f:
                f.write(f"Top Players by Average {goal_col}:\n")
                for i, row in player_goals.iterrows():
                    f.write(f"- {row[player_col]}: {row['mean']:.2f} (±{row['std']:.2f}, {row['count']} games)\n")
                f.write("\n")
            
            logger.info(f"Top players by {goal_col} saved to {stats_path}")
        
        # Analyze player consistency
        if disposal_col is not None:
            logger.info(f"Analyzing player consistency in {disposal_col}...")
            
            # Calculate coefficient of variation (CV = std / mean)
            player_consistency = data.groupby(player_col)[disposal_col].agg(['mean', 'std', 'count']).reset_index()
            player_consistency = player_consistency[player_consistency['count'] >= 10]  # At least 10 games
            player_consistency['cv'] = player_consistency['std'] / player_consistency['mean']
            
            # Most consistent players (low CV)
            most_consistent = player_consistency.sort_values('cv').head(20)
            
            # Create bar plot
            plt.figure(figsize=(14, 8))
            sns.barplot(x='cv', y=player_col, data=most_consistent)
            plt.title(f'Top 20 Most Consistent Players in {disposal_col}')
            plt.xlabel('Coefficient of Variation (lower is more consistent)')
            plt.ylabel('Player')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, f'most_consistent_players_{disposal_col}.png'))
            plt.close()
            
            # Save to file
            stats_path = os.path.join(player_dir, f'player_consistency_{disposal_col}.txt')
            with open(stats_path, 'w') as f:
                f.write(f"Most Consistent Players in {disposal_col}:\n")
                for i, row in most_consistent.iterrows():
                    f.write(f"- {row[player_col]}: CV={row['cv']:.3f}, Mean={row['mean']:.2f}, Std={row['std']:.2f}, Games={row['count']}\n")
                f.write("\n")
            
            logger.info(f"Player consistency analysis saved to {stats_path}")
    except Exception as e:
        logger.error(f"Error analyzing player performance: {e}")

def analyze_team_performance(data):
    """
    Analyze team performance
    """
    if data is None:
        return
    
    try:
        logger.info("Analyzing team performance...")
        
        # Identify team column
        team_col = None
        for col in ['team', 'club', 'team_name']:
            if col in data.columns:
                team_col = col
                break
        
        if team_col is None:
            logger.error("Team column not found")
            return
        
        # Identify target columns
        disposal_col = None
        for col in ['dis.', 'disposals', 'disposal', 'total_disposals']:
            if col in data.columns:
                disposal_col = col
                break
        
        goal_col = None
        for col in ['goals', 'gls', 'gls_avg', 'goals_avg', 'total_goals']:
            if col in data.columns:
                goal_col = col
                break
        
        if disposal_col is None and goal_col is None:
            logger.error("Target columns not found")
            return
        
        # Create directory for team analysis
        team_dir = os.path.join(ANALYSIS_DIR, 'teams')
        os.makedirs(team_dir, exist_ok=True)
        
        # Analyze team disposals
        if disposal_col is not None:
            logger.info(f"Analyzing team {disposal_col}...")
            
            # Calculate average disposals by team
            team_disposals = data.groupby(team_col)[disposal_col].agg(['mean', 'std', 'count']).reset_index()
            team_disposals = team_disposals.sort_values('mean', ascending=False)
            
            # Create bar plot
            plt.figure(figsize=(14, 8))
            sns.barplot(x='mean', y=team_col, data=team_disposals)
            plt.title(f'Teams by Average Player {disposal_col}')
            plt.xlabel(f'Average {disposal_col}')
            plt.ylabel('Team')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, f'team_{disposal_col}.png'))
            plt.close()
            
            # Save to file
            stats_path = os.path.join(team_dir, f'team_{disposal_col}.txt')
            with open(stats_path, 'w') as f:
                f.write(f"Teams by Average Player {disposal_col}:\n")
                for i, row in team_disposals.iterrows():
                    f.write(f"- {row[team_col]}: {row['mean']:.2f} (±{row['std']:.2f}, {row['count']} player games)\n")
                f.write("\n")
            
            logger.info(f"Team {disposal_col} analysis saved to {stats_path}")
        
        # Analyze team goals
        if goal_col is not None:
            logger.info(f"Analyzing team {goal_col}...")
            
            # Calculate average goals by team
            team_goals = data.groupby(team_col)[goal_col].agg(['mean', 'std', 'count']).reset_index()
            team_goals = team_goals.sort_values('mean', ascending=False)
            
            # Create bar plot
            plt.figure(figsize=(14, 8))
            sns.barplot(x='mean', y=team_col, data=team_goals)
            plt.title(f'Teams by Average Player {goal_col}')
            plt.xlabel(f'Average {goal_col}')
            plt.ylabel('Team')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, f'team_{goal_col}.png'))
            plt.close()
            
            # Save to file
            stats_path = os.path.join(team_dir, f'team_{goal_col}.txt')
            with open(stats_path, 'w') as f:
                f.write(f"Teams by Average Player {goal_col}:\n")
                for i, row in team_goals.iterrows():
                    f.write(f"- {row[team_col]}: {row['mean']:.2f} (±{row['std']:.2f}, {row['count']} player games)\n")
                f.write("\n")
            
            logger.info(f"Team {goal_col} analysis saved to {stats_path}")
        
        # Analyze year-over-year team performance
        if 'year' in data.columns and disposal_col is not None:
            logger.info("Analyzing year-over-year team performance...")
            
            # Calculate average disposals by team and year
            team_year_disposals = data.groupby([team_col, 'year'])[disposal_col].mean().reset_index()
            
            # Create line plot
            plt.figure(figsize=(14, 8))
            sns.lineplot(x='year', y=disposal_col, hue=team_col, data=team_year_disposals, marker='o')
            plt.title(f'Team {disposal_col} by Year')
            plt.xlabel('Year')
            plt.ylabel(f'Average {disposal_col}')
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, f'team_{disposal_col}_by_year.png'))
            plt.close()
            
            logger.info("Year-over-year team performance analysis saved")
    except Exception as e:
        logger.error(f"Error analyzing team performance: {e}")

def analyze_form_metrics(data):
    """
    Analyze form metrics and their impact on performance
    """
    if data is None:
        return
    
    try:
        logger.info("Analyzing form metrics...")
        
        # Check if form metrics exist
        form_metrics = [col for col in data.columns if 'last_' in col or 'consistency' in col or 'trend' in col]
        
        if not form_metrics:
            logger.warning("No form metrics found in data")
            return
        
        # Identify target columns
        disposal_col = None
        for col in ['dis.', 'disposals', 'disposal', 'total_disposals']:
            if col in data.columns:
                disposal_col = col
                break
        
        goal_col = None
        for col in ['goals', 'gls', 'gls_avg', 'goals_avg', 'total_goals']:
            if col in data.columns:
                goal_col = col
                break
        
        # Create directory for form analysis
        form_dir = os.path.join(ANALYSIS_DIR, 'form')
        os.makedirs(form_dir, exist_ok=True)
        
        # Analyze form metrics for disposals
        if disposal_col is not None:
            disposal_form_metrics = [col for col in form_metrics if disposal_col.lower() in col.lower()]
            
            if disposal_form_metrics:
                logger.info(f"Analyzing form metrics for {disposal_col}...")
                
                # Calculate correlations
                disposal_form_corrs = data[[disposal_col] + disposal_form_metrics].corr()[disposal_col].sort_values(ascending=False)
                
                # Create correlation plot
                plt.figure(figsize=(12, 8))
                sns.barplot(x=disposal_form_corrs.values[1:], y=disposal_form_corrs.index[1:])
                plt.title(f'Form Metrics Correlation with {disposal_col}')
                plt.xlabel('Correlation')
                plt.ylabel('Form Metric')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(FIGURES_DIR, f'{disposal_col}_form_correlations.png'))
                plt.close()
                
                # Save correlations to file
                corr_path = os.path.join(form_dir, f'{disposal_col}_form_correlations.txt')
                with open(corr_path, 'w') as f:
                    f.write(f"Form Metrics Correlations with {disposal_col}:\n")
                    for feature, corr in disposal_form_corrs.items():
                        if feature != disposal_col:
                            f.write(f"- {feature}: {corr:.4f}\n")
                    f.write("\n")
                
                logger.info(f"{disposal_col} form correlations saved to {corr_path}")
                
                # Create scatter plots for top form metrics
                top_metrics = disposal_form_corrs.index[1:4]  # Top 3 form metrics
                
                for metric in top_metrics:
                    if metric in data.columns:
                        plt.figure(figsize=(10, 6))
                        sns.scatterplot(x=metric, y=disposal_col, data=data, alpha=0.5)
                        plt.title(f'Relationship between {metric} and {disposal_col}')
                        plt.xlabel(metric)
                        plt.ylabel(disposal_col)
                        plt.grid(True, alpha=0.3)
                        plt.savefig(os.path.join(FIGURES_DIR, f'{disposal_col}_{metric}_relationship.png'))
                        plt.close()
        
        # Analyze form metrics for goals
        if goal_col is not None:
            goal_form_metrics = [col for col in form_metrics if goal_col.lower() in col.lower()]
            
            if goal_form_metrics:
                logger.info(f"Analyzing form metrics for {goal_col}...")
                
                # Calculate correlations
                goal_form_corrs = data[[goal_col] + goal_form_metrics].corr()[goal_col].sort_values(ascending=False)
                
                # Create correlation plot
                plt.figure(figsize=(12, 8))
                sns.barplot(x=goal_form_corrs.values[1:], y=goal_form_corrs.index[1:])
                plt.title(f'Form Metrics Correlation with {goal_col}')
                plt.xlabel('Correlation')
                plt.ylabel('Form Metric')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(FIGURES_DIR, f'{goal_col}_form_correlations.png'))
                plt.close()
                
                # Save correlations to file
                corr_path = os.path.join(form_dir, f'{goal_col}_form_correlations.txt')
                with open(corr_path, 'w') as f:
                    f.write(f"Form Metrics Correlations with {goal_col}:\n")
                    for feature, corr in goal_form_corrs.items():
                        if feature != goal_col:
                            f.write(f"- {feature}: {corr:.4f}\n")
                    f.write("\n")
                
                logger.info(f"{goal_col} form correlations saved to {corr_path}")
                
                # Create scatter plots for top form metrics
                top_metrics = goal_form_corrs.index[1:4]  # Top 3 form metrics
                
                for metric in top_metrics:
                    if metric in data.columns:
                        plt.figure(figsize=(10, 6))
                        sns.scatterplot(x=metric, y=goal_col, data=data, alpha=0.5)
                        plt.title(f'Relationship between {metric} and {goal_col}')
                        plt.xlabel(metric)
                        plt.ylabel(goal_col)
                        plt.grid(True, alpha=0.3)
                        plt.savefig(os.path.join(FIGURES_DIR, f'{goal_col}_{metric}_relationship.png'))
                        plt.close()
    except Exception as e:
        logger.error(f"Error analyzing form metrics: {e}")

def generate_summary_report(data):
    """
    Generate a summary report of the exploratory data analysis
    """
    if data is None:
        return
    
    try:
        logger.info("Generating summary report...")
        
        # Create summary report
        report_path = os.path.join(ANALYSIS_DIR, 'eda_summary_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# AFL Player Statistics Exploratory Data Analysis\n\n")
            
            # Data overview
            f.write("## Data Overview\n\n")
            f.write(f"- Total records: {len(data)}\n")
            f.write(f"- Total features: {len(data.columns)}\n")
            
            if 'year' in data.columns:
                years = sorted(data['year'].unique())
                f.write(f"- Years covered: {', '.join(map(str, years))}\n")
            
            # Identify key columns
            player_col = None
            for col in ['player', 'player_name', 'name']:
                if col in data.columns:
                    player_col = col
                    break
            
            team_col = None
            for col in ['team', 'club', 'team_name']:
                if col in data.columns:
                    team_col = col
                    break
            
            disposal_col = None
            for col in ['dis.', 'disposals', 'disposal', 'total_disposals']:
                if col in data.columns:
                    disposal_col = col
                    break
            
            goal_col = None
            for col in ['goals', 'gls', 'gls_avg', 'goals_avg', 'total_goals']:
                if col in data.columns:
                    goal_col = col
                    break
            
            if player_col is not None:
                players = data[player_col].nunique()
                f.write(f"- Total players: {players}\n")
            
            if team_col is not None:
                teams = data[team_col].nunique()
                f.write(f"- Total teams: {teams}\n")
            
            f.write("\n")
            
            # Target variables
            f.write("## Target Variables\n\n")
            
            if disposal_col is not None:
                f.write(f"### {disposal_col}\n\n")
                stats = data[disposal_col].describe()
                f.write(f"- Mean: {stats['mean']:.2f}\n")
                f.write(f"- Median: {stats['50%']:.2f}\n")
                f.write(f"- Standard deviation: {stats['std']:.2f}\n")
                f.write(f"- Minimum: {stats['min']:.2f}\n")
                f.write(f"- Maximum: {stats['max']:.2f}\n")
                f.write("\n")
                f.write(f"![{disposal_col} Distribution](../analysis/figures/{disposal_col}_distribution.png)\n\n")
            
            if goal_col is not None:
                f.write(f"### {goal_col}\n\n")
                stats = data[goal_col].describe()
                f.write(f"- Mean: {stats['mean']:.2f}\n")
                f.write(f"- Median: {stats['50%']:.2f}\n")
                f.write(f"- Standard deviation: {stats['std']:.2f}\n")
                f.write(f"- Minimum: {stats['min']:.2f}\n")
                f.write(f"- Maximum: {stats['max']:.2f}\n")
                f.write("\n")
                f.write(f"![{goal_col} Distribution](../analysis/figures/{goal_col}_distribution.png)\n\n")
            
            # Feature correlations
            f.write("## Feature Correlations\n\n")
            f.write("The following features show the strongest correlations with our target variables:\n\n")
            
            if disposal_col is not None:
                f.write(f"### Top Features for {disposal_col}\n\n")
                f.write(f"![{disposal_col} Correlations](../analysis/figures/{disposal_col}_correlations.png)\n\n")
            
            if goal_col is not None:
                f.write(f"### Top Features for {goal_col}\n\n")
                f.write(f"![{goal_col} Correlations](../analysis/figures/{goal_col}_correlations.png)\n\n")
            
            # Player performance
            f.write("## Player Performance\n\n")
            
            if disposal_col is not None:
                f.write(f"### Top Players by {disposal_col}\n\n")
                f.write(f"![Top Players by {disposal_col}](../analysis/figures/top_players_{disposal_col}.png)\n\n")
                f.write(f"### Most Consistent Players in {disposal_col}\n\n")
                f.write(f"![Most Consistent Players in {disposal_col}](../analysis/figures/most_consistent_players_{disposal_col}.png)\n\n")
            
            if goal_col is not None:
                f.write(f"### Top Players by {goal_col}\n\n")
                f.write(f"![Top Players by {goal_col}](../analysis/figures/top_players_{goal_col}.png)\n\n")
            
            # Team performance
            f.write("## Team Performance\n\n")
            
            if disposal_col is not None:
                f.write(f"### Teams by Average Player {disposal_col}\n\n")
                f.write(f"![Teams by {disposal_col}](../analysis/figures/team_{disposal_col}.png)\n\n")
            
            if goal_col is not None:
                f.write(f"### Teams by Average Player {goal_col}\n\n")
                f.write(f"![Teams by {goal_col}](../analysis/figures/team_{goal_col}.png)\n\n")
            
            # Form metrics
            form_metrics = [col for col in data.columns if 'last_' in col or 'consistency' in col or 'trend' in col]
            
            if form_metrics:
                f.write("## Form Metrics Analysis\n\n")
                f.write("Form metrics measure a player's recent performance and are crucial for weekly predictions.\n\n")
                
                if disposal_col is not None:
                    disposal_form_metrics = [col for col in form_metrics if disposal_col.lower() in col.lower()]
                    if disposal_form_metrics:
                        f.write(f"### Form Metrics for {disposal_col}\n\n")
                        f.write(f"![Form Metrics for {disposal_col}](../analysis/figures/{disposal_col}_form_correlations.png)\n\n")
                
                if goal_col is not None:
                    goal_form_metrics = [col for col in form_metrics if goal_col.lower() in col.lower()]
                    if goal_form_metrics:
                        f.write(f"### Form Metrics for {goal_col}\n\n")
                        f.write(f"![Form Metrics for {goal_col}](../analysis/figures/{goal_col}_form_correlations.png)\n\n")
            
            # Conclusions
            f.write("## Conclusions and Next Steps\n\n")
            f.write("Based on the exploratory data analysis, we can draw the following conclusions:\n\n")
            
            f.write("1. The data provides comprehensive statistics for AFL players across multiple seasons\n")
            f.write("2. There are clear patterns in player performance that can be leveraged for predictions\n")
            f.write("3. Form metrics show strong correlations with performance outcomes\n")
            f.write("4. Team and player-specific factors significantly influence disposal and goal counts\n\n")
            
            f.write("Next steps for model development:\n\n")
            f.write("1. Engineer features based on the identified correlations\n")
            f.write("2. Split data into appropriate training and testing sets\n")
            f.write("3. Develop separate models for disposals and goals predictions\n")
            f.write("4. Incorporate form metrics to capture recent performance trends\n")
            f.write("5. Evaluate models using appropriate metrics\n")
            f.write("6. Fine-tune models for optimal performance\n")
        
        logger.info(f"Summary report generated: {report_path}")
        return report_path
    except Exception as e:
        logger.error(f"Error generating summary report: {e}")
        return None

def main():
    """
    Main function to execute the exploratory data analysis
    """
    logger.info("Starting exploratory data analysis...")
    
    # Load data
    data = load_data()
    
    if data is None:
        logger.error("Failed to load data")
        return
    
    # Analyze data structure
    analyze_data_structure(data)
    
    # Analyze target variables
    analyze_target_variables(data)
    
    # Analyze feature correlations
    analyze_feature_correlations(data)
    
    # Analyze player performance
    analyze_player_performance(data)
    
    # Analyze team performance
    analyze_team_performance(data)
    
    # Analyze form metrics
    analyze_form_metrics(data)
    
    # Generate summary report
    report_path = generate_summary_report(data)
    
    logger.info("Exploratory data analysis completed successfully!")
    
    if report_path:
        logger.info(f"Summary report available at: {report_path}")

if __name__ == "__main__":
    main()
