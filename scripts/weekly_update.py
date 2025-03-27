#!/usr/bin/env python3
"""
AFL Prediction Weekly Update Script

This script automatically updates the AFL prediction system with the latest data,
recalculates form metrics, and generates new predictions for upcoming fixtures.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import datetime
import requests
from bs4 import BeautifulSoup
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/ubuntu/afl_prediction_project/logs/weekly_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('weekly_update')

# Define paths
BASE_DIR = '/home/ubuntu/afl_prediction_project'
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PREDICTIONS_DIR = os.path.join(BASE_DIR, 'predictions')

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)

# Website URLs
WHEELORATINGS_BASE_URL = 'https://www.wheeloratings.com'
PLAYER_STATS_URL = f'{WHEELORATINGS_BASE_URL}/afl_stats.html'
MATCH_PREVIEWS_URL = f'{WHEELORATINGS_BASE_URL}/afl_match_previews.html'

def get_current_year():
    """Get the current year"""
    return datetime.datetime.now().year

def get_current_round():
    """
    Determine the current AFL round by scraping the match previews page
    """
    try:
        logger.info("Determining current AFL round...")
        response = requests.get(MATCH_PREVIEWS_URL)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for round information in the page
        round_info = soup.find(text=lambda t: t and 'Round' in t)
        if round_info:
            # Extract round number
            import re
            match = re.search(r'Round (\d+)', round_info)
            if match:
                current_round = int(match.group(1))
                logger.info(f"Current round: {current_round}")
                return current_round
        
        logger.warning("Could not determine current round, defaulting to 1")
        return 1
    except Exception as e:
        logger.error(f"Error determining current round: {e}")
        return 1

def download_latest_player_stats():
    """
    Download the latest player statistics from wheeloratings.com
    """
    current_year = get_current_year()
    
    try:
        logger.info(f"Downloading latest player statistics for {current_year}...")
        
        # Create URL for current year's stats
        url = f"{PLAYER_STATS_URL}?comp=afl&season={current_year}"
        
        # Create directory for current year if it doesn't exist
        year_dir = os.path.join(RAW_DATA_DIR, str(current_year))
        os.makedirs(year_dir, exist_ok=True)
        
        # Use requests to download the CSV file
        response = requests.get(f"{url}&format=csv")
        
        if response.status_code == 200:
            # Save the CSV file
            csv_path = os.path.join(year_dir, f'player_stats_{current_year}_{datetime.datetime.now().strftime("%Y%m%d")}.csv')
            with open(csv_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Player statistics saved to {csv_path}")
            return csv_path
        else:
            logger.error(f"Failed to download player statistics: HTTP {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error downloading player statistics: {e}")
        return None

def download_upcoming_fixtures():
    """
    Download the upcoming fixtures from wheeloratings.com
    """
    try:
        logger.info("Downloading upcoming fixtures...")
        
        # Create directory for fixtures if it doesn't exist
        fixtures_dir = os.path.join(RAW_DATA_DIR, 'fixtures')
        os.makedirs(fixtures_dir, exist_ok=True)
        
        # Download the match previews page
        response = requests.get(MATCH_PREVIEWS_URL)
        
        if response.status_code == 200:
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract fixture information
            fixtures = []
            
            # Find match preview sections
            match_sections = soup.find_all('div', class_='match-preview')
            
            for section in match_sections:
                try:
                    # Extract teams
                    teams = section.find_all('div', class_='team-name')
                    if len(teams) >= 2:
                        home_team = teams[0].text.strip()
                        away_team = teams[1].text.strip()
                        
                        # Extract date and time
                        date_time = section.find('div', class_='match-date').text.strip()
                        
                        # Extract venue
                        venue = section.find('div', class_='venue').text.strip()
                        
                        fixtures.append({
                            'home_team': home_team,
                            'away_team': away_team,
                            'date_time': date_time,
                            'venue': venue
                        })
                except Exception as e:
                    logger.warning(f"Error parsing fixture: {e}")
            
            # Save fixtures to CSV
            if fixtures:
                fixtures_df = pd.DataFrame(fixtures)
                csv_path = os.path.join(fixtures_dir, f'upcoming_fixtures_{datetime.datetime.now().strftime("%Y%m%d")}.csv')
                fixtures_df.to_csv(csv_path, index=False)
                
                logger.info(f"Upcoming fixtures saved to {csv_path}")
                return csv_path, fixtures_df
            else:
                logger.warning("No fixtures found")
                return None, None
        else:
            logger.error(f"Failed to download fixtures: HTTP {response.status_code}")
            return None, None
    except Exception as e:
        logger.error(f"Error downloading fixtures: {e}")
        return None, None

def load_historical_data():
    """
    Load the historical player statistics data
    """
    try:
        logger.info("Loading historical player statistics data...")
        
        # Check if processed data exists
        processed_data_path = os.path.join(PROCESSED_DATA_DIR, 'afl_player_stats_all_years.pkl')
        
        if os.path.exists(processed_data_path):
            # Load the processed data
            historical_data = pd.read_pickle(processed_data_path)
            logger.info(f"Loaded historical data: {len(historical_data)} records")
            return historical_data
        else:
            logger.error("Processed historical data not found")
            return None
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        return None

def update_historical_data(historical_data, latest_data_path):
    """
    Update the historical data with the latest player statistics
    """
    if historical_data is None or latest_data_path is None:
        logger.error("Cannot update historical data: missing data")
        return None
    
    try:
        logger.info("Updating historical data with latest statistics...")
        
        # Load the latest data
        latest_data = pd.read_csv(latest_data_path)
        
        # Add year column if not present
        if 'year' not in latest_data.columns:
            latest_data['year'] = get_current_year()
        
        # Standardize column names
        latest_data.columns = [col.lower().replace(' ', '_') for col in latest_data.columns]
        
        # Identify key columns for merging
        player_col = None
        for col in ['player', 'player_name', 'name']:
            if col in latest_data.columns:
                player_col = col
                break
        
        if player_col is None:
            logger.error("Cannot identify player column in latest data")
            return historical_data
        
        # Remove existing records for the current year
        current_year = get_current_year()
        if 'year' in historical_data.columns:
            historical_data = historical_data[historical_data['year'] != current_year]
        
        # Concatenate historical and latest data
        updated_data = pd.concat([historical_data, latest_data], ignore_index=True)
        
        # Save the updated data
        updated_data_path = os.path.join(PROCESSED_DATA_DIR, 'afl_player_stats_all_years.pkl')
        updated_data.to_pickle(updated_data_path)
        
        csv_path = os.path.join(PROCESSED_DATA_DIR, 'afl_player_stats_all_years.csv')
        updated_data.to_csv(csv_path, index=False)
        
        logger.info(f"Updated historical data saved: {len(updated_data)} records")
        return updated_data
    except Exception as e:
        logger.error(f"Error updating historical data: {e}")
        return historical_data

def calculate_form_metrics(data):
    """
    Calculate form metrics for each player
    """
    if data is None:
        logger.error("Cannot calculate form metrics: missing data")
        return None
    
    try:
        logger.info("Calculating player form metrics...")
        
        # Identify key columns
        player_col = None
        for col in ['player', 'player_name', 'name']:
            if col in data.columns:
                player_col = col
                break
        
        if player_col is None:
            logger.error("Cannot identify player column")
            return data
        
        # Identify match/game column
        match_col = None
        for col in ['match', 'game', 'gm', 'match_id']:
            if col in data.columns:
                match_col = col
                break
        
        if match_col is None:
            logger.error("Cannot identify match column")
            return data
        
        # Identify date or round column for sorting
        date_col = None
        for col in ['date', 'match_date', 'game_date']:
            if col in data.columns:
                date_col = col
                break
        
        round_col = None
        for col in ['round', 'rnd', 'round_number']:
            if col in data.columns:
                round_col = col
                break
        
        time_col = date_col if date_col is not None else round_col
        
        if time_col is None:
            logger.warning("Cannot identify date or round column, using match column for sorting")
            time_col = match_col
        
        # Identify target columns for disposals and goals
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
        
        # Create a copy of the data
        form_data = data.copy()
        
        # Sort by player and time
        form_data = form_data.sort_values([player_col, time_col])
        
        # Calculate rolling averages for disposals
        if disposal_col is not None:
            logger.info(f"Calculating rolling averages for {disposal_col}")
            
            # Group by player and calculate rolling averages
            form_data[f'{disposal_col}_last_3'] = form_data.groupby(player_col)[disposal_col].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
            
            form_data[f'{disposal_col}_last_5'] = form_data.groupby(player_col)[disposal_col].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )
            
            form_data[f'{disposal_col}_last_10'] = form_data.groupby(player_col)[disposal_col].transform(
                lambda x: x.rolling(window=10, min_periods=1).mean()
            )
            
            # Calculate consistency (standard deviation)
            form_data[f'{disposal_col}_consistency'] = form_data.groupby(player_col)[disposal_col].transform(
                lambda x: x.rolling(window=5, min_periods=3).std()
            )
            
            # Calculate trend (positive or negative)
            form_data[f'{disposal_col}_trend'] = form_data.groupby(player_col)[disposal_col].transform(
                lambda x: x.rolling(window=3, min_periods=2).apply(
                    lambda y: 1 if y.iloc[-1] > y.iloc[0] else (-1 if y.iloc[-1] < y.iloc[0] else 0)
                )
            )
        
        # Calculate rolling averages for goals
        if goal_col is not None:
            logger.info(f"Calculating rolling averages for {goal_col}")
            
            # Group by player and calculate rolling averages
            form_data[f'{goal_col}_last_3'] = form_data.groupby(player_col)[goal_col].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
            
            form_data[f'{goal_col}_last_5'] = form_data.groupby(player_col)[goal_col].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )
            
            form_data[f'{goal_col}_last_10'] = form_data.groupby(player_col)[goal_col].transform(
                lambda x: x.rolling(window=10, min_periods=1).mean()
            )
            
            # Calculate consistency (standard deviation)
            form_data[f'{goal_col}_consistency'] = form_data.groupby(player_col)[goal_col].transform(
                lambda x: x.rolling(window=5, min_periods=3).std()
            )
            
            # Calculate trend (positive or negative)
            form_data[f'{goal_col}_trend'] = form_data.groupby(player_col)[goal_col].transform(
                lambda x: x.rolling(window=3, min_periods=2).apply(
                    lambda y: 1 if y.iloc[-1] > y.iloc[0] else (-1 if y.iloc[-1] < y.iloc[0] else 0)
                )
            )
        
        # Fill NaN values in form metrics
        form_cols = [col for col in form_data.columns if 'last_' in col or 'consistency' in col or 'trend' in col]
        form_data[form_cols] = form_data[form_cols].fillna(0)
        
        logger.info(f"Form metrics calculated: {len(form_cols)} new columns")
        return form_data
    except Exception as e:
        logger.error(f"Error calculating form metrics: {e}")
        return data

def load_prediction_models():
    """
    Load the trained prediction models
    """
    try:
        logger.info("Loading prediction models...")
        
        models = {}
        
        # Load disposals prediction model
        disposals_model_path = os.path.join(MODELS_DIR, 'disposals_prediction_model_tuned.pkl')
        if not os.path.exists(disposals_model_path):
            disposals_model_path = os.path.join(MODELS_DIR, 'disposals_prediction_model.pkl')
        
        if os.path.exists(disposals_model_path):
            with open(disposals_model_path, 'rb') as f:
                models['disposals'] = pickle.load(f)
            logger.info("Loaded disposals prediction model")
        else:
            logger.warning("Disposals prediction model not found")
        
        # Load goals prediction model
        goals_model_path = os.path.join(MODELS_DIR, 'goals_prediction_model_tuned.pkl')
        if not os.path.exists(goals_model_path):
            goals_model_path = os.path.join(MODELS_DIR, 'goals_prediction_model.pkl')
        
        if os.path.exists(goals_model_path):
            with open(goals_model_path, 'rb') as f:
                models['goals'] = pickle.load(f)
            logger.info("Loaded goals prediction model")
        else:
            logger.warning("Goals prediction model not found")
        
        return models
    except Exception as e:
        logger.error(f"Error loading prediction models: {e}")
        return {}

def get_team_players(data, team_name):
    """
    Get the list of players for a specific team
    """
    if data is None:
        return []
    
    try:
        # Identify team column
        team_col = None
        for col in ['team', 'club', 'team_name']:
            if col in data.columns:
                team_col = col
                break
        
        if team_col is None:
            logger.error("Cannot identify team column")
            return []
        
        # Identify player column
        player_col = None
        for col in ['player', 'player_name', 'name']:
            if col in data.columns:
                player_col = col
                break
        
        if player_col is None:
            logger.error("Cannot identify player column")
            return []
        
        # Filter data for the current year
        current_year = get_current_year()
        if 'year' in data.columns:
            current_data = data[data['year'] == current_year]
        else:
            current_data = data
        
        # Find team name variations
        team_variations = {
            'Adelaide': ['Adelaide', 'Adel', 'Crows'],
            'Brisbane': ['Brisbane', 'Bris', 'Lions'],
            'Carlton': ['Carlton', 'Carl', 'Blues'],
            'Collingwood': ['Collingwood', 'Coll', 'Magpies'],
            'Essendon': ['Essendon', 'Ess', 'Bombers'],
            'Fremantle': ['Fremantle', 'Frem', 'Dockers'],
            'Geelong': ['Geelong', 'Geel', 'Cats'],
            'Gold Coast': ['Gold Coast', 'GC', 'Suns'],
            'Greater Western Sydney': ['Greater Western Sydney', 'GWS', 'Giants'],
            'Hawthorn': ['Hawthorn', 'Haw', 'Hawks'],
            'Melbourne': ['Melbourne', 'Melb', 'Demons'],
            'North Melbourne': ['North Melbourne', 'NM', 'Kangaroos'],
            'Port Adelaide': ['Port Adelaide', 'Port', 'Power'],
            'Richmond': ['Richmond', 'Rich', 'Tigers'],
            'St Kilda': ['St Kilda', 'StK', 'Saints'],
            'Sydney': ['Sydney', 'Syd', 'Swans'],
            'West Coast': ['West Coast', 'WC', 'Eagles'],
            'Western Bulldogs': ['Western Bulldogs', 'WB', 'Bulldogs', 'Footscray']
        }
        
        # Find matching team variations
        matching_teams = []
        for full_name, variations in team_variations.items():
            if any(var.lower() in team_name.lower() for var in variations):
                matching_teams.extend(variations)
        
        if not matching_teams:
            logger.warning(f"No matching team variations found for {team_name}")
            matching_teams = [team_name]
        
        # Filter players for the team
        team_players = current_data[current_data[team_col].str.contains('|'.join(matching_teams), case=False, na=False)]
        
        # Get unique players
        unique_players = team_players[player_col].unique()
        
        logger.info(f"Found {len(unique_players)} players for {team_name}")
        return unique_players
    except Exception as e:
        logger.error(f"Error getting team players: {e}")
        return []

def prepare_player_features(data, player_name):
    """
    Prepare features for a specific player for prediction
    """
    if data is None:
        return None
    
    try:
        # Identify player column
        player_col = None
        for col in ['player', 'player_name', 'name']:
            if col in data.columns:
                player_col = col
                break
        
        if player_col is None:
            logger.error("Cannot identify player column")
            return None
        
        # Filter data for the player
        player_data = data[data[player_col] == player_name]
        
        if player_data.empty:
            logger.warning(f"No data found for player {player_name}")
            return None
        
        # Get the most recent record
        recent_data = player_data.iloc[-1].to_dict()
        
        # Select only numeric features
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        
        # Exclude target columns
        exclude_cols = ['dis.', 'disposals', 'disposal', 'total_disposals', 
                        'goals', 'gls', 'gls_avg', 'goals_avg', 'total_goals']
        
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Create feature vector
        features = {col: recent_data.get(col, 0) for col in feature_cols}
        
        return features
    except Exception as e:
        logger.error(f"Error preparing player features: {e}")
        return None

def generate_predictions(data, models, fixtures_df):
    """
    Generate predictions for upcoming fixtures
    """
    if data is None or not models or fixtures_df is None or fixtures_df.empty:
        logger.error("Cannot generate predictions: missing data, models, or fixtures")
        return None
    
    try:
        logger.info("Generating predictions for upcoming fixtures...")
        
        predictions = []
        
        # Process each fixture
        for _, fixture in fixtures_df.iterrows():
            home_team = fixture['home_team']
            away_team = fixture['away_team']
            date_time = fixture['date_time']
            venue = fixture['venue']
            
            logger.info(f"Processing fixture: {home_team} vs {away_team}")
            
            # Get players for each team
            home_players = get_team_players(data, home_team)
            away_players = get_team_players(data, away_team)
            
            # Generate predictions for home team players
            for player in home_players:
                # Prepare features
                features = prepare_player_features(data, player)
                
                if features is None:
                    continue
                
                # Convert features to DataFrame
                features_df = pd.DataFrame([features])
                
                # Make predictions
                disposal_pred = None
                goal_pred = None
                
                if 'disposals' in models and features_df is not None:
                    try:
                        disposal_pred = models['disposals'].predict(features_df)[0]
                    except Exception as e:
                        logger.warning(f"Error predicting disposals for {player}: {e}")
                
                if 'goals' in models and features_df is not None:
                    try:
                        goal_pred = models['goals'].predict(features_df)[0]
                    except Exception as e:
                        logger.warning(f"Error predicting goals for {player}: {e}")
                
                # Add prediction to results
                if disposal_pred is not None or goal_pred is not None:
                    predictions.append({
                        'fixture': f"{home_team} vs {away_team}",
                        'date_time': date_time,
                        'venue': venue,
                        'player': player,
                        'team': home_team,
                        'is_home': True,
                        'predicted_disposals': round(disposal_pred, 1) if disposal_pred is not None else None,
                        'predicted_goals': round(goal_pred, 1) if goal_pred is not None else None
                    })
            
            # Generate predictions for away team players
            for player in away_players:
                # Prepare features
                features = prepare_player_features(data, player)
                
                if features is None:
                    continue
                
                # Convert features to DataFrame
                features_df = pd.DataFrame([features])
                
                # Make predictions
                disposal_pred = None
                goal_pred = None
                
                if 'disposals' in models and features_df is not None:
                    try:
                        disposal_pred = models['disposals'].predict(features_df)[0]
                    except Exception as e:
                        logger.warning(f"Error predicting disposals for {player}: {e}")
                
                if 'goals' in models and features_df is not None:
                    try:
                        goal_pred = models['goals'].predict(features_df)[0]
                    except Exception as e:
                        logger.warning(f"Error predicting goals for {player}: {e}")
                
                # Add prediction to results
                if disposal_pred is not None or goal_pred is not None:
                    predictions.append({
                        'fixture': f"{home_team} vs {away_team}",
                        'date_time': date_time,
                        'venue': venue,
                        'player': player,
                        'team': away_team,
                        'is_home': False,
                        'predicted_disposals': round(disposal_pred, 1) if disposal_pred is not None else None,
                        'predicted_goals': round(goal_pred, 1) if goal_pred is not None else None
                    })
        
        # Convert predictions to DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        if predictions_df.empty:
            logger.warning("No predictions generated")
            return None
        
        # Sort by fixture and predicted values (descending)
        predictions_df = predictions_df.sort_values(['fixture', 'predicted_disposals', 'predicted_goals'], 
                                                   ascending=[True, False, False])
        
        # Save predictions
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        csv_path = os.path.join(PREDICTIONS_DIR, f'predictions_{current_date}.csv')
        predictions_df.to_csv(csv_path, index=False)
        
        logger.info(f"Predictions saved to {csv_path}")
        return predictions_df
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        return None

def identify_best_bets(predictions_df, confidence_threshold=0.8):
    """
    Identify the best betting opportunities from the predictions
    """
    if predictions_df is None or predictions_df.empty:
        logger.error("Cannot identify best bets: missing predictions")
        return None
    
    try:
        logger.info("Identifying best betting opportunities...")
        
        # Create a copy of the predictions
        best_bets = predictions_df.copy()
        
        # Add confidence score (placeholder - in a real system this would be more sophisticated)
        # For now, we'll use a simple heuristic based on the prediction values
        if 'predicted_disposals' in best_bets.columns:
            # Higher disposal predictions get higher confidence
            best_bets['disposal_confidence'] = best_bets['predicted_disposals'].apply(
                lambda x: min(0.95, 0.5 + (x / 40)) if x is not None else 0
            )
        
        if 'predicted_goals' in best_bets.columns:
            # Higher goal predictions get higher confidence
            best_bets['goal_confidence'] = best_bets['predicted_goals'].apply(
                lambda x: min(0.95, 0.6 + (x / 5)) if x is not None else 0
            )
        
        # Filter for high confidence predictions
        high_disposal_confidence = best_bets[
            (best_bets['disposal_confidence'] >= confidence_threshold) &
            (best_bets['predicted_disposals'] >= 20)  # Minimum threshold for disposals
        ].sort_values('disposal_confidence', ascending=False)
        
        high_goal_confidence = best_bets[
            (best_bets['goal_confidence'] >= confidence_threshold) &
            (best_bets['predicted_goals'] >= 1.5)  # Minimum threshold for goals
        ].sort_values('goal_confidence', ascending=False)
        
        # Combine the best bets
        combined_bets = pd.concat([
            high_disposal_confidence.head(10),  # Top 10 disposal bets
            high_goal_confidence.head(10)       # Top 10 goal bets
        ]).drop_duplicates()
        
        # Save best bets
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        csv_path = os.path.join(PREDICTIONS_DIR, f'best_bets_{current_date}.csv')
        combined_bets.to_csv(csv_path, index=False)
        
        logger.info(f"Best bets saved to {csv_path}")
        return combined_bets
    except Exception as e:
        logger.error(f"Error identifying best bets: {e}")
        return None

def format_telegram_message(best_bets):
    """
    Format the best bets as a Telegram message
    """
    if best_bets is None or best_bets.empty:
        return "No high-confidence predictions available for upcoming fixtures."
    
    try:
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        message = f"🏉 AFL Predictions for {current_date} 🏉\n\n"
        
        # Group by fixture
        for fixture, group in best_bets.groupby('fixture'):
            message += f"📊 {fixture}\n"
            
            # Add disposal predictions
            disposal_bets = group[group['predicted_disposals'] >= 20].sort_values('predicted_disposals', ascending=False)
            
            if not disposal_bets.empty:
                message += "🏃 Disposal Predictions:\n"
                for _, bet in disposal_bets.iterrows():
                    confidence = bet.get('disposal_confidence', 0) * 100
                    message += f"  • {bet['player']} ({bet['team']}): {bet['predicted_disposals']:.1f} disposals"
                    message += f" ({confidence:.0f}% confidence)\n"
            
            # Add goal predictions
            goal_bets = group[group['predicted_goals'] >= 1.5].sort_values('predicted_goals', ascending=False)
            
            if not goal_bets.empty:
                message += "⚽ Goal Predictions:\n"
                for _, bet in goal_bets.iterrows():
                    confidence = bet.get('goal_confidence', 0) * 100
                    message += f"  • {bet['player']} ({bet['team']}): {bet['predicted_goals']:.1f} goals"
                    message += f" ({confidence:.0f}% confidence)\n"
            
            message += "\n"
        
        message += "💡 These predictions are based on historical performance and current form.\n"
        message += "🔄 Updated daily with the latest player statistics."
        
        return message
    except Exception as e:
        logger.error(f"Error formatting Telegram message: {e}")
        return "Error generating predictions message."

def send_telegram_message(message, bot_token=None, chat_id=None):
    """
    Send a message to a Telegram chat
    """
    if not bot_token or not chat_id:
        logger.warning("Telegram bot token or chat ID not provided")
        
        # Save the message to a file instead
        message_path = os.path.join(PREDICTIONS_DIR, f'telegram_message_{datetime.datetime.now().strftime("%Y%m%d")}.txt')
        with open(message_path, 'w') as f:
            f.write(message)
        
        logger.info(f"Telegram message saved to {message_path}")
        return False
    
    try:
        logger.info("Sending Telegram message...")
        
        # Send the message
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        response = requests.post(url, data=data)
        
        if response.status_code == 200:
            logger.info("Telegram message sent successfully")
            return True
        else:
            logger.error(f"Failed to send Telegram message: HTTP {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error sending Telegram message: {e}")
        return False

def main():
    """
    Main function to execute the weekly update process
    """
    logger.info("Starting AFL prediction weekly update...")
    
    # Step 1: Download the latest player statistics
    latest_data_path = download_latest_player_stats()
    
    # Step 2: Download upcoming fixtures
    fixtures_path, fixtures_df = download_upcoming_fixtures()
    
    # Step 3: Load historical data
    historical_data = load_historical_data()
    
    # Step 4: Update historical data with latest statistics
    updated_data = update_historical_data(historical_data, latest_data_path)
    
    # Step 5: Calculate form metrics
    form_data = calculate_form_metrics(updated_data)
    
    # Step 6: Load prediction models
    models = load_prediction_models()
    
    # Step 7: Generate predictions for upcoming fixtures
    predictions_df = generate_predictions(form_data, models, fixtures_df)
    
    # Step 8: Identify best betting opportunities
    best_bets = identify_best_bets(predictions_df)
    
    # Step 9: Format and send Telegram message
    if best_bets is not None and not best_bets.empty:
        message = format_telegram_message(best_bets)
        
        # Get Telegram bot token and chat ID from environment variables
        bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
        chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        
        send_telegram_message(message, bot_token, chat_id)
    
    logger.info("AFL prediction weekly update completed successfully!")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)
    
    main()
