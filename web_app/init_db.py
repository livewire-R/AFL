#!/usr/bin/env python3
"""
Initialize the AFL prediction system database
"""

from app import app, db, User, Player, Fixture, Prediction
from datetime import datetime, timedelta
import pandas as pd
import os

def init_db():
    """Initialize the database with tables and test data"""
    with app.app_context():
        # Create all tables
        db.create_all()
        
        # Create a test user
        if not User.query.filter_by(username='test').first():
            user = User(username='test', email='test@example.com')
            user.set_password('test123')
            db.session.add(user)
            db.session.commit()
        
        # Load current player stats
        players_file = os.path.join('..', 'data', 'raw', 'current', 'afl-player-stats-2025.csv')
        if os.path.exists(players_file):
            players_df = pd.read_csv(players_file)
            
            # Load last 5 games stats
            last5_file = os.path.join('..', 'data', 'raw', 'current', 'afl-player-stats-last5.csv')
            last5_df = pd.read_csv(last5_file) if os.path.exists(last5_file) else None
            
            # Load last 10 games stats
            last10_file = os.path.join('..', 'data', 'raw', 'current', 'afl-player-stats-last10 (1).csv')
            last10_df = pd.read_csv(last10_file) if os.path.exists(last10_file) else None
            
            for _, row in players_df.iterrows():
                if not Player.query.filter_by(name=row['Player']).first():
                    # Get last 5 stats
                    last5_stats = last5_df[last5_df['Player'] == row['Player']].iloc[0] if last5_df is not None and not last5_df[last5_df['Player'] == row['Player']].empty else None
                    
                    # Get last 10 stats
                    last10_stats = last10_df[last10_df['Player'] == row['Player']].iloc[0] if last10_df is not None and not last10_df[last10_df['Player'] == row['Player']].empty else None
                    
                    # Calculate trends and consistency
                    disposal_trend = 0.0
                    goal_trend = 0.0
                    disposal_consistency = 0.8  # Default values
                    goal_consistency = 0.7
                    
                    if last5_stats is not None and last10_stats is not None:
                        disposal_trend = (last5_stats['RatingPoints_Avg'] / 5) - (last10_stats['RatingPoints_Avg'] / 10)
                        goal_trend = 0  # No goal data in CSV
                    
                    player = Player(
                        name=row['Player'],
                        team=row['Team'],
                        position=row['Position'],
                        avg_disposals=row['RatingPoints_Avg'],
                        avg_goals=0,  # No goal data in CSV
                        last_3_disposals=last5_stats['RatingPoints_Avg'] if last5_stats is not None else 0,
                        last_5_disposals=last5_stats['RatingPoints_Avg'] if last5_stats is not None else 0,
                        last_3_goals=0,  # No goal data in CSV
                        last_5_goals=0,  # No goal data in CSV
                        disposal_consistency=disposal_consistency,
                        goal_consistency=goal_consistency,
                        disposal_trend=disposal_trend,
                        goal_trend=goal_trend
                    )
                    db.session.add(player)
            
            db.session.commit()
            print(f"Loaded {len(players_df)} players")
        
        # Load fixtures
        fixtures_file = os.path.join('..', 'data', 'raw', 'fixtures', 'afl-2025-fixtures-WAustraliaStandardTime.csv')
        if os.path.exists(fixtures_file):
            fixtures_df = pd.read_csv(fixtures_file)
            
            for _, row in fixtures_df.iterrows():
                # Convert date string to datetime
                try:
                    match_date = pd.to_datetime(row['Date'])
                except:
                    print(f"Error parsing date: {row['Date']}")
                    continue
                
                if not Fixture.query.filter_by(
                    home_team=row['Home Team'],
                    away_team=row['Away Team'],
                    match_date=match_date
                ).first():
                    fixture = Fixture(
                        home_team=row['Home Team'],
                        away_team=row['Away Team'],
                        venue=row['Location'],  # Changed from 'Venue' to 'Location'
                        match_date=match_date,
                        round_number=row['Round Number'],  # Changed from 'Round' to 'Round Number'
                        season=2025
                    )
                    db.session.add(fixture)
            
            db.session.commit()
            print(f"Loaded {len(fixtures_df)} fixtures")
        
        # Generate predictions for upcoming fixtures
        players = Player.query.all()
        fixtures = Fixture.query.filter(Fixture.match_date > datetime.now()).all()
        
        prediction_count = 0
        for fixture in fixtures:
            home_players = [p for p in players if p.team == fixture.home_team]
            away_players = [p for p in players if p.team == fixture.away_team]
            
            for player in home_players + away_players:
                if not Prediction.query.filter_by(
                    player_id=player.id,
                    fixture_id=fixture.id
                ).first():
                    # Calculate prediction confidence based on player consistency
                    disposal_confidence = player.disposal_consistency if player.disposal_consistency else 0.8
                    goal_confidence = player.goal_consistency if player.goal_consistency else 0.7
                    
                    # Adjust predictions based on trends
                    disposal_adjustment = 1.0 + (player.disposal_trend if player.disposal_trend else 0)
                    goal_adjustment = 1.0 + (player.goal_trend if player.goal_trend else 0)
                    
                    prediction = Prediction(
                        player_id=player.id,
                        fixture_id=fixture.id,
                        predicted_disposals=player.avg_disposals * disposal_adjustment,
                        predicted_goals=player.avg_goals * goal_adjustment,
                        disposal_confidence=disposal_confidence,
                        goal_confidence=goal_confidence
                    )
                    db.session.add(prediction)
                    prediction_count += 1
        
        db.session.commit()
        print(f"Generated {prediction_count} predictions")
        print("Database initialized successfully!")

if __name__ == "__main__":
    init_db()
