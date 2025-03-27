#!/usr/bin/env python3
"""
Initialize the AFL prediction system database with fixtures and predictions.
"""

import os
import sys
import pandas as pd
import pickle
from datetime import datetime
import logging

# Add the web_app directory to the Python path so we can import from app
web_app_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'web_app')
sys.path.append(web_app_dir)

from app import db, Fixture, Player, Prediction

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_fixtures():
    """Load fixtures from CSV into database."""
    fixtures_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                'data', 'raw', 'fixtures', 
                                'afl-2025-fixtures-WAustraliaStandardTime.csv')
    
    if not os.path.exists(fixtures_file):
        logger.error(f"Fixtures file not found: {fixtures_file}")
        return
    
    fixtures_df = pd.read_csv(fixtures_file)
    
    for _, row in fixtures_df.iterrows():
        fixture = Fixture(
            home_team=row['Home Team'],
            away_team=row['Away Team'],
            venue=row['Venue'],
            match_date=datetime.strptime(row['Date'], '%Y-%m-%d %H:%M:%S'),
            round_number=row['Round'],
            season=2025
        )
        db.session.add(fixture)
    
    db.session.commit()
    logger.info("Fixtures loaded into database")

def load_players():
    """Load current players into database."""
    players_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               'data', 'raw', 'current', 
                               'afl-player-stats-2025.csv')
    
    if not os.path.exists(players_file):
        logger.error(f"Players file not found: {players_file}")
        return
    
    players_df = pd.read_csv(players_file)
    
    for _, row in players_df.iterrows():
        player = Player(
            name=row['Player'],
            team=row['Team'],
            position=row.get('Position', ''),
            avg_disposals=row.get('Avg Disposals', 0.0),
            avg_goals=row.get('Avg Goals', 0.0),
            last_3_disposals=row.get('Last 3 Disposals', 0.0),
            last_5_disposals=row.get('Last 5 Disposals', 0.0),
            last_3_goals=row.get('Last 3 Goals', 0.0),
            last_5_goals=row.get('Last 5 Goals', 0.0)
        )
        db.session.add(player)
    
    db.session.commit()
    logger.info("Players loaded into database")

def generate_predictions():
    """Generate predictions for upcoming fixtures."""
    # Load the prediction pipeline
    pipeline_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                'models', 'multi_task_prediction_pipeline.pkl')
    
    if not os.path.exists(pipeline_file):
        logger.error(f"Prediction pipeline not found: {pipeline_file}")
        return
    
    with open(pipeline_file, 'rb') as f:
        pipeline = pickle.load(f)
    
    # Get upcoming fixtures
    upcoming_fixtures = Fixture.query.filter(Fixture.match_date > datetime.utcnow()).all()
    
    # Get all players
    players = Player.query.all()
    
    for fixture in upcoming_fixtures:
        for player in players:
            if player.team in [fixture.home_team, fixture.away_team]:
                # Prepare input data for prediction
                input_data = {
                    'Player': player.name,
                    'Team': player.team,
                    'Opponent': fixture.away_team if player.team == fixture.home_team else fixture.home_team,
                    'Avg Disposals': player.avg_disposals,
                    'Avg Goals': player.avg_goals,
                    'Last 3 Disposals': player.last_3_disposals,
                    'Last 5 Disposals': player.last_5_disposals,
                    'Last 3 Goals': player.last_3_goals,
                    'Last 5 Goals': player.last_5_goals
                }
                
                # Make prediction
                pred_disposals, pred_goals = pipeline.predict(pd.DataFrame([input_data]))
                
                # Create prediction record
                prediction = Prediction(
                    player_id=player.id,
                    fixture_id=fixture.id,
                    predicted_disposals=float(pred_disposals[0]),
                    predicted_goals=float(pred_goals[0]),
                    disposal_confidence=0.8,  # You might want to calculate this
                    goal_confidence=0.7      # You might want to calculate this
                )
                db.session.add(prediction)
    
    db.session.commit()
    logger.info("Predictions generated and stored in database")

def main():
    """Main function to initialize the system."""
    logger.info("Starting database initialization...")
    
    # Create all tables
    db.create_all()
    
    # Load fixtures
    load_fixtures()
    
    # Load players
    load_players()
    
    # Generate predictions
    generate_predictions()
    
    logger.info("Database initialization complete!")

if __name__ == "__main__":
    main()
