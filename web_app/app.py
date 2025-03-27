#!/usr/bin/env python3
"""
AFL Prediction Web Application

This is the main application file for the AFL prediction web app.
It includes user authentication, database integration, and prediction display.
"""

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_migrate import Migrate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Define paths
BASE_DIR = r'C:\Users\ralph\OneDrive\Desktop\AFL'
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
STATIC_DIR = os.path.join(BASE_DIR, 'web_app', 'static')

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'afl_prediction_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'web_app', 'afl_prediction.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load prediction models
def load_prediction_models():
    try:
        # Try to load prediction pipeline
        pipeline_path = os.path.join(MODELS_DIR, 'prediction_pipeline.pkl')
        
        if os.path.exists(pipeline_path):
            with open(pipeline_path, 'rb') as f:
                pipeline = pickle.load(f)
            return pipeline
        else:
            # Try to load individual models
            disposal_model_path = os.path.join(MODELS_DIR, 'disposals_prediction_model_tuned.pkl')
            if not os.path.exists(disposal_model_path):
                disposal_model_path = os.path.join(MODELS_DIR, 'disposals_prediction_model.pkl')
            
            goal_model_path = os.path.join(MODELS_DIR, 'goals_prediction_model_tuned.pkl')
            if not os.path.exists(goal_model_path):
                goal_model_path = os.path.join(MODELS_DIR, 'goals_prediction_model.pkl')
            
            preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.pkl')
            
            if os.path.exists(disposal_model_path) and os.path.exists(goal_model_path) and os.path.exists(preprocessor_path):
                with open(disposal_model_path, 'rb') as f:
                    disposal_model = pickle.load(f)
                
                with open(goal_model_path, 'rb') as f:
                    goal_model = pickle.load(f)
                
                with open(preprocessor_path, 'rb') as f:
                    preprocessor = pickle.load(f)
                
                return {
                    'disposal_model': disposal_model,
                    'goal_model': goal_model,
                    'preprocessor': preprocessor
                }
            else:
                return None
    except Exception as e:
        print(f"Error loading prediction models: {e}")
        return None

# Define database models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('SavedPrediction', backref='user', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class SavedPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    player_name = db.Column(db.String(100), nullable=False)
    team = db.Column(db.String(100), nullable=False)
    opponent = db.Column(db.String(100), nullable=False)
    match_date = db.Column(db.DateTime, nullable=False)
    predicted_disposals = db.Column(db.Float)
    predicted_goals = db.Column(db.Float)
    actual_disposals = db.Column(db.Float, nullable=True)
    actual_goals = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'player_name': self.player_name,
            'team': self.team,
            'opponent': self.opponent,
            'match_date': self.match_date.strftime('%Y-%m-%d'),
            'predicted_disposals': self.predicted_disposals,
            'predicted_goals': self.predicted_goals,
            'actual_disposals': self.actual_disposals,
            'actual_goals': self.actual_goals,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }

class Fixture(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    home_team = db.Column(db.String(100), nullable=False)
    away_team = db.Column(db.String(100), nullable=False)
    venue = db.Column(db.String(100), nullable=False)
    match_date = db.Column(db.DateTime, nullable=False)
    round_number = db.Column(db.Integer, nullable=False)
    season = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'home_team': self.home_team,
            'away_team': self.away_team,
            'venue': self.venue,
            'match_date': self.match_date.strftime('%Y-%m-%d %H:%M:%S'),
            'round_number': self.round_number,
            'season': self.season
        }

class Player(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    team = db.Column(db.String(100), nullable=False)
    position = db.Column(db.String(100))
    avg_disposals = db.Column(db.Float)
    avg_goals = db.Column(db.Float)
    last_3_disposals = db.Column(db.Float)
    last_5_disposals = db.Column(db.Float)
    last_3_goals = db.Column(db.Float)
    last_5_goals = db.Column(db.Float)
    disposal_consistency = db.Column(db.Float)
    goal_consistency = db.Column(db.Float)
    disposal_trend = db.Column(db.Float)
    goal_trend = db.Column(db.Float)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'team': self.team,
            'position': self.position,
            'avg_disposals': self.avg_disposals,
            'avg_goals': self.avg_goals,
            'last_3_disposals': self.last_3_disposals,
            'last_5_disposals': self.last_5_disposals,
            'last_3_goals': self.last_3_goals,
            'last_5_goals': self.last_5_goals,
            'disposal_consistency': self.disposal_consistency,
            'goal_consistency': self.goal_consistency,
            'disposal_trend': self.disposal_trend,
            'goal_trend': self.goal_trend,
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S')
        }

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    player_id = db.Column(db.Integer, db.ForeignKey('player.id'), nullable=False)
    fixture_id = db.Column(db.Integer, db.ForeignKey('fixture.id'), nullable=False)
    predicted_disposals = db.Column(db.Float)
    predicted_goals = db.Column(db.Float)
    disposal_confidence = db.Column(db.Float)
    goal_confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    player = db.relationship('Player', backref='predictions')
    fixture = db.relationship('Fixture', backref='predictions')
    
    def to_dict(self):
        return {
            'id': self.id,
            'player_id': self.player_id,
            'player_name': self.player.name if self.player else None,
            'team': self.player.team if self.player else None,
            'fixture_id': self.fixture_id,
            'home_team': self.fixture.home_team if self.fixture else None,
            'away_team': self.fixture.away_team if self.fixture else None,
            'match_date': self.fixture.match_date.strftime('%Y-%m-%d %H:%M:%S') if self.fixture else None,
            'predicted_disposals': self.predicted_disposals,
            'predicted_goals': self.predicted_goals,
            'disposal_confidence': self.disposal_confidence,
            'goal_confidence': self.goal_confidence,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def index():
    """Home page"""
    # Get upcoming fixtures
    upcoming_fixtures = Fixture.query.filter(Fixture.match_date > datetime.utcnow()).order_by(Fixture.match_date).limit(5).all()
    
    # Get recent predictions
    recent_predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(10).all()
    
    # Get top disposal predictions for upcoming fixtures
    top_disposal_predictions = Prediction.query.join(Fixture).filter(
        Fixture.match_date > datetime.utcnow()
    ).order_by(Prediction.predicted_disposals.desc()).limit(5).all()
    
    # Get top goal predictions for upcoming fixtures
    top_goal_predictions = Prediction.query.join(Fixture).filter(
        Fixture.match_date > datetime.utcnow()
    ).order_by(Prediction.predicted_goals.desc()).limit(5).all()
    
    return render_template(
        'index.html',
        upcoming_fixtures=upcoming_fixtures,
        recent_predictions=recent_predictions,
        top_disposal_predictions=top_disposal_predictions,
        top_goal_predictions=top_goal_predictions,
        user=current_user
    )

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validate input
        if not username or not email or not password or not confirm_password:
            flash('All fields are required', 'danger')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))
        
        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'danger')
            return redirect(url_for('register'))
        
        # Create new user
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Validate input
        if not username or not password:
            flash('Username and password are required', 'danger')
            return redirect(url_for('login'))
        
        # Check if user exists
        user = User.query.filter_by(username=username).first()
        
        if not user or not user.check_password(password):
            flash('Invalid username or password', 'danger')
            return redirect(url_for('login'))
        
        # Log in user
        login_user(user)
        flash('Login successful!', 'success')
        
        # Redirect to next page or home
        next_page = request.args.get('next')
        return redirect(next_page or url_for('index'))
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    """User profile page"""
    # Get user's saved predictions
    saved_predictions = SavedPrediction.query.filter_by(user_id=current_user.id).order_by(SavedPrediction.created_at.desc()).all()
    
    return render_template('profile.html', user=current_user, saved_predictions=saved_predictions)

@app.route('/fixtures')
def fixtures():
    """Fixtures page"""
    # Get upcoming fixtures
    upcoming_fixtures = Fixture.query.filter(Fixture.match_date > datetime.utcnow()).order_by(Fixture.match_date).all()
    
    # Get past fixtures
    past_fixtures = Fixture.query.filter(Fixture.match_date <= datetime.utcnow()).order_by(Fixture.match_date.desc()).limit(10).all()
    
    return render_template('fixtures.html', upcoming_fixtures=upcoming_fixtures, past_fixtures=past_fixtures)

@app.route('/predictions')
def predictions():
    """Predictions page"""
    # Get upcoming fixtures
    upcoming_fixtures = Fixture.query.filter(Fixture.match_date > datetime.utcnow()).order_by(Fixture.match_date).all()
    
    # Get predictions for upcoming fixtures
    upcoming_predictions = []
    
    for fixture in upcoming_fixtures:
        fixture_predictions = Prediction.query.filter_by(fixture_id=fixture.id).all()
        
        if fixture_predictions:
            upcoming_predictions.append({
                'fixture': fixture,
                'predictions': fixture_predictions
            })
    
    return render_template('predictions.html', upcoming_predictions=upcoming_predictions)

@app.route('/player/<int:player_id>')
def player_detail(player_id):
    """Player detail page"""
    # Get player
    player = Player.query.get_or_404(player_id)
    
    # Get player's predictions
    player_predictions = Prediction.query.filter_by(player_id=player_id).join(Fixture).order_by(Fixture.match_date).all()
    
    # Create player form chart
    create_player_form_chart(player)
    
    return render_template(
        'player_detail.html',
        player=player,
        player_predictions=player_predictions,
        form_chart_path=f'images/player_{player_id}_form.png'
    )

@app.route('/fixture/<int:fixture_id>')
def fixture_detail(fixture_id):
    """Fixture detail page"""
    # Get fixture
    fixture = Fixture.query.get_or_404(fixture_id)
    
    # Get predictions for this fixture
    fixture_predictions = Prediction.query.filter_by(fixture_id=fixture_id).all()
    
    # Group predictions by team
    home_team_predictions = [p for p in fixture_predictions if p.player.team == fixture.home_team]
    away_team_predictions = [p for p in fixture_predictions if p.player.team == fixture.away_team]
    
    return render_template(
        'fixture_detail.html',
        fixture=fixture,
        home_team_predictions=home_team_predictions,
        away_team_predictions=away_team_predictions
    )

@app.route('/save_prediction', methods=['POST'])
@login_required
def save_prediction():
    """Save a prediction for the current user"""
    if request.method == 'POST':
        prediction_id = request.form.get('prediction_id')
        
        # Get prediction
        prediction = Prediction.query.get_or_404(prediction_id)
        
        # Create saved prediction
        saved_prediction = SavedPrediction(
            user_id=current_user.id,
            player_name=prediction.player.name,
            team=prediction.player.team,
            opponent=prediction.fixture.away_team if prediction.player.team == prediction.fixture.home_team else prediction.fixture.home_team,
            match_date=prediction.fixture.match_date,
            predicted_disposals=prediction.predicted_disposals,
            predicted_goals=prediction.predicted_goals
        )
        
        db.session.add(saved_prediction)
        db.session.commit()
        
        flash('Prediction saved successfully!', 'success')
        return redirect(url_for('predictions'))

@app.route('/api/fixtures')
def api_fixtures():
    """API endpoint for fixtures"""
    # Get upcoming fixtures
    upcoming_fixtures = Fixture.query.filter(Fixture.match_date > datetime.utcnow()).order_by(Fixture.match_date).all()
    
    # Convert to dict
    fixtures_dict = [fixture.to_dict() for fixture in upcoming_fixtures]
    
    return jsonify(fixtures_dict)

@app.route('/api/predictions')
def api_predictions():
    """API endpoint for predictions"""
    # Get upcoming predictions
    upcoming_predictions = Prediction.query.join(Fixture).filter(
        Fixture.match_date > datetime.utcnow()
    ).order_by(Fixture.match_date).all()
    
    # Convert to dict
    predictions_dict = [prediction.to_dict() for prediction in upcoming_predictions]
    
    return jsonify(predictions_dict)

@app.route('/api/players')
def api_players():
    """API endpoint for players"""
    # Get all players
    players = Player.query.all()
    
    # Convert to dict
    players_dict = [player.to_dict() for player in players]
    
    return jsonify(players_dict)

# Helper functions
def create_player_form_chart(player):
    """Create a chart showing player form over time"""
    try:
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create data
        form_data = {
            'Last 3 Disposals': player.last_3_disposals,
            'Last 5 Disposals': player.last_5_disposals,
            'Average Disposals': player.avg_disposals,
            'Last 3 Goals': player.last_3_goals,
            'Last 5 Goals': player.last_5_goals,
            'Average Goals': player.avg_goals
        }
        
        # Create bar chart
        plt.bar(form_data.keys(), form_data.values())
        plt.title(f'{player.name} Form')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(STATIC_DIR, 'images', f'player_{player.id}_form.png')
        os.makedirs(os.path.dirname(chart_path), exist_ok=True)
        plt.savefig(chart_path)
        plt.close()
        
        return True
    except Exception as e:
        print(f"Error creating player form chart: {e}")
        return False

# Create database tables
with app.app_context():
    db.create_all()

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
