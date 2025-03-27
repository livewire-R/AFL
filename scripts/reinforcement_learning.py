#!/usr/bin/env python3
"""
Reinforcement Learning System for AFL Predictions

This script implements a reinforcement learning system that learns optimal prediction
strategies over time based on prediction success for AFL player disposals and goals.
"""

import os
import pandas as pd
import numpy as np
import pickle
import logging
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_field_mapping import get_rl_state_columns, get_rl_reward_columns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('reinforcement_learning')

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
RL_DIR = os.path.join(MODELS_DIR, 'reinforcement_learning')

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RL_DIR, exist_ok=True)

class PredictionEnvironment:
    """
    Environment for reinforcement learning that tracks prediction performance
    and provides rewards based on prediction accuracy.
    """
    def __init__(self, prediction_history_file=None):
        self.prediction_history_file = prediction_history_file or os.path.join(RL_DIR, 'prediction_history.json')
        self.prediction_history = self._load_prediction_history()
        self.current_state = self._get_current_state()
        
    def _load_prediction_history(self):
        """Load prediction history from file or initialize if not exists"""
        if os.path.exists(self.prediction_history_file):
            try:
                with open(self.prediction_history_file, 'r') as f:
                    history = json.load(f)
                logger.info(f"Loaded prediction history with {len(history)} records")
                return history
            except Exception as e:
                logger.error(f"Error loading prediction history: {e}")
                return []
        else:
            logger.info("No prediction history found, initializing empty history")
            return []
    
    def _save_prediction_history(self):
        """Save prediction history to file"""
        try:
            with open(self.prediction_history_file, 'w') as f:
                json.dump(self.prediction_history, f, indent=2)
            logger.info(f"Saved prediction history with {len(self.prediction_history)} records")
        except Exception as e:
            logger.error(f"Error saving prediction history: {e}")
    
    def _get_current_state(self):
        """
        Get the current state of the environment based on recent prediction performance
        
        State includes:
        - Recent disposal prediction accuracy
        - Recent goal prediction accuracy
        - Recent disposal prediction confidence
        - Recent goal prediction confidence
        - Player form trend
        - Opponent strength
        """
        if not self.prediction_history:
            # Default state if no history
            return {
                'disposal_accuracy': 0.5,
                'goal_accuracy': 0.5,
                'disposal_confidence': 0.5,
                'goal_confidence': 0.5,
                'player_form_trend': 0,
                'opponent_strength': 0.5
            }
        
        # Get recent predictions (last 20)
        recent_predictions = self.prediction_history[-20:] if len(self.prediction_history) > 20 else self.prediction_history
        
        # Calculate disposal accuracy
        disposal_predictions = [p for p in recent_predictions if 'actual_disposals' in p and p['actual_disposals'] is not None]
        if disposal_predictions:
            disposal_errors = [abs(p['predicted_disposals'] - p['actual_disposals']) for p in disposal_predictions]
            avg_disposal_error = sum(disposal_errors) / len(disposal_errors)
            max_error = 15  # Assuming max error of 15 disposals
            disposal_accuracy = max(0, 1 - (avg_disposal_error / max_error))
        else:
            disposal_accuracy = 0.5
        
        # Calculate goal accuracy
        goal_predictions = [p for p in recent_predictions if 'actual_goals' in p and p['actual_goals'] is not None]
        if goal_predictions:
            goal_errors = [abs(p['predicted_goals'] - p['actual_goals']) for p in goal_predictions]
            avg_goal_error = sum(goal_errors) / len(goal_errors)
            max_error = 5  # Assuming max error of 5 goals
            goal_accuracy = max(0, 1 - (avg_goal_error / max_error))
        else:
            goal_accuracy = 0.5
        
        # Calculate average confidence
        disposal_confidence = sum([p.get('disposal_confidence', 0.5) for p in recent_predictions]) / len(recent_predictions)
        goal_confidence = sum([p.get('goal_confidence', 0.5) for p in recent_predictions]) / len(recent_predictions)
        
        # Calculate player form trend (positive or negative)
        if len(disposal_predictions) >= 2:
            sorted_predictions = sorted(disposal_predictions, key=lambda p: p['match_date'])
            early_performance = [p['actual_disposals'] for p in sorted_predictions[:len(sorted_predictions)//2]]
            late_performance = [p['actual_disposals'] for p in sorted_predictions[len(sorted_predictions)//2:]]
            player_form_trend = (sum(late_performance) / len(late_performance)) - (sum(early_performance) / len(early_performance))
            # Normalize trend
            player_form_trend = max(-10, min(10, player_form_trend)) / 10
        else:
            player_form_trend = 0
        
        # Calculate opponent strength (based on how well players perform against them)
        opponent_performances = {}
        for p in recent_predictions:
            opponent = p.get('opponent')
            if opponent and 'actual_disposals' in p and p['actual_disposals'] is not None:
                if opponent not in opponent_performances:
                    opponent_performances[opponent] = []
                opponent_performances[opponent].append(p['actual_disposals'])
        
        if opponent_performances:
            avg_performances = {opp: sum(perfs) / len(perfs) for opp, perfs in opponent_performances.items()}
            all_avgs = list(avg_performances.values())
            min_avg = min(all_avgs)
            max_avg = max(all_avgs)
            range_avg = max_avg - min_avg if max_avg > min_avg else 1
            opponent_strength = (sum(all_avgs) / len(all_avgs) - min_avg) / range_avg
        else:
            opponent_strength = 0.5
        
        return {
            'disposal_accuracy': disposal_accuracy,
            'goal_accuracy': goal_accuracy,
            'disposal_confidence': disposal_confidence,
            'goal_confidence': goal_confidence,
            'player_form_trend': player_form_trend,
            'opponent_strength': opponent_strength
        }
    
    def add_prediction(self, prediction):
        """
        Add a new prediction to the history
        
        Args:
            prediction: Dictionary containing prediction details
        """
        # Ensure prediction has required fields
        required_fields = ['player_name', 'team', 'opponent', 'match_date', 
                          'predicted_disposals', 'predicted_goals']
        
        for field in required_fields:
            if field not in prediction:
                logger.error(f"Prediction missing required field: {field}")
                return False
        
        # Add timestamp
        prediction['timestamp'] = datetime.now().isoformat()
        
        # Add to history
        self.prediction_history.append(prediction)
        
        # Save history
        self._save_prediction_history()
        
        # Update current state
        self.current_state = self._get_current_state()
        
        return True
    
    def update_actual_results(self, player_name, match_date, actual_disposals=None, actual_goals=None):
        """
        Update predictions with actual results
        
        Args:
            player_name: Name of the player
            match_date: Date of the match
            actual_disposals: Actual disposal count
            actual_goals: Actual goal count
        """
        match_date_str = match_date.isoformat() if isinstance(match_date, datetime) else match_date
        
        # Find matching predictions
        updated = False
        for prediction in self.prediction_history:
            if (prediction['player_name'] == player_name and 
                prediction['match_date'].split('T')[0] == match_date_str.split('T')[0]):
                
                if actual_disposals is not None:
                    prediction['actual_disposals'] = actual_disposals
                
                if actual_goals is not None:
                    prediction['actual_goals'] = actual_goals
                
                updated = True
        
        if updated:
            # Save history
            self._save_prediction_history()
            
            # Update current state
            self.current_state = self._get_current_state()
            
            logger.info(f"Updated actual results for {player_name} on {match_date_str}")
            return True
        else:
            logger.warning(f"No matching prediction found for {player_name} on {match_date_str}")
            return False
    
    def get_state(self):
        """Get current environment state"""
        return self.current_state
    
    def get_reward(self, prediction, actual_result):
        """
        Calculate reward based on prediction accuracy
        
        Args:
            prediction: Dictionary with predicted values
            actual_result: Dictionary with actual values
        
        Returns:
            reward: Numerical reward value
        """
        reward = 0
        
        # Reward for disposal prediction accuracy
        if 'predicted_disposals' in prediction and 'actual_disposals' in actual_result:
            disposal_error = abs(prediction['predicted_disposals'] - actual_result['actual_disposals'])
            max_error = 15  # Assuming max error of 15 disposals
            disposal_accuracy = max(0, 1 - (disposal_error / max_error))
            
            # Higher reward for higher confidence that was correct
            confidence = prediction.get('disposal_confidence', 0.5)
            if disposal_accuracy > 0.8:  # Very accurate prediction
                reward += confidence * 2
            elif disposal_accuracy > 0.6:  # Moderately accurate
                reward += confidence
            else:  # Poor prediction
                reward -= (1 - disposal_accuracy) * confidence
        
        # Reward for goal prediction accuracy
        if 'predicted_goals' in prediction and 'actual_goals' in actual_result:
            goal_error = abs(prediction['predicted_goals'] - actual_result['actual_goals'])
            max_error = 5  # Assuming max error of 5 goals
            goal_accuracy = max(0, 1 - (goal_error / max_error))
            
            # Higher reward for higher confidence that was correct
            confidence = prediction.get('goal_confidence', 0.5)
            if goal_accuracy > 0.8:  # Very accurate prediction
                reward += confidence * 2
            elif goal_accuracy > 0.6:  # Moderately accurate
                reward += confidence
            else:  # Poor prediction
                reward -= (1 - goal_accuracy) * confidence
        
        return reward
    
    def get_performance_metrics(self):
        """
        Calculate performance metrics for predictions
        
        Returns:
            metrics: Dictionary of performance metrics
        """
        if not self.prediction_history:
            return {
                'disposal_mae': None,
                'disposal_rmse': None,
                'goal_mae': None,
                'goal_rmse': None,
                'disposal_accuracy_rate': None,
                'goal_accuracy_rate': None,
                'total_predictions': 0,
                'predictions_with_results': 0
            }
        
        # Get predictions with actual results
        disposal_predictions = [p for p in self.prediction_history 
                              if 'predicted_disposals' in p and 'actual_disposals' in p 
                              and p['actual_disposals'] is not None]
        
        goal_predictions = [p for p in self.prediction_history 
                          if 'predicted_goals' in p and 'actual_goals' in p 
                          and p['actual_goals'] is not None]
        
        metrics = {
            'total_predictions': len(self.prediction_history),
            'predictions_with_results': len(disposal_predictions)
        }
        
        # Calculate disposal metrics
        if disposal_predictions:
            y_true = [p['actual_disposals'] for p in disposal_predictions]
            y_pred = [p['predicted_disposals'] for p in disposal_predictions]
            
            metrics['disposal_mae'] = mean_absolute_error(y_true, y_pred)
            metrics['disposal_rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            
            # Calculate accuracy rate (within 3 disposals)
            accurate_count = sum(1 for p in disposal_predictions 
                               if abs(p['predicted_disposals'] - p['actual_disposals']) <= 3)
            metrics['disposal_accuracy_rate'] = accurate_count / len(disposal_predictions)
        else:
            metrics['disposal_mae'] = None
            metrics['disposal_rmse'] = None
            metrics['disposal_accuracy_rate'] = None
        
        # Calculate goal metrics
        if goal_predictions:
            y_true = [p['actual_goals'] for p in goal_predictions]
            y_pred = [p['predicted_goals'] for p in goal_predictions]
            
            metrics['goal_mae'] = mean_absolute_error(y_true, y_pred)
            metrics['goal_rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            
            # Calculate accuracy rate (within 1 goal)
            accurate_count = sum(1 for p in goal_predictions 
                               if abs(p['predicted_goals'] - p['actual_goals']) <= 1)
            metrics['goal_accuracy_rate'] = accurate_count / len(goal_predictions)
        else:
            metrics['goal_mae'] = None
            metrics['goal_rmse'] = None
            metrics['goal_accuracy_rate'] = None
        
        return metrics
    
    def visualize_performance(self, output_file=None):
        """
        Create visualizations of prediction performance
        
        Args:
            output_file: Path to save visualization
        """
        if not self.prediction_history:
            logger.warning("No prediction history available for visualization")
            return
        
        # Get predictions with actual results
        disposal_predictions = [p for p in self.prediction_history 
                              if 'predicted_disposals' in p and 'actual_disposals' in p 
                              and p['actual_disposals'] is not None]
        
        goal_predictions = [p for p in self.prediction_history 
                          if 'predicted_goals' in p and 'actual_goals' in p 
                          and p['actual_goals'] is not None]
        
        if not disposal_predictions and not goal_predictions:
            logger.warning("No predictions with actual results available for visualization")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot disposal predictions vs actual
        if disposal_predictions:
            ax = axes[0, 0]
            x = [p['actual_disposals'] for p in disposal_predictions]
            y = [p['predicted_disposals'] for p in disposal_predictions]
            ax.scatter(x, y, alpha=0.6)
            
            # Add perfect prediction line
            min_val = min(min(x), min(y))
            max_val = max(max(x), max(y))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            ax.set_xlabel('Actual Disposals')
            ax.set_ylabel('Predicted Disposals')
            ax.set_title('Disposal Predictions vs Actual')
            
            # Add error histogram
            ax = axes[0, 1]
            errors = [p['predicted_disposals'] - p['actual_disposals'] for p in disposal_predictions]
            ax.hist(errors, bins=20, alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--')
            ax.set_xlabel('Prediction Error (Predicted - Actual)')
            ax.set_ylabel('Frequency')
            ax.set_title('Disposal Prediction Error Distribution')
        
        # Plot goal predictions vs actual
        if goal_predictions:
            ax = axes[1, 0]
            x = [p['actual_goals'] for p in goal_predictions]
            y = [p['predicted_goals'] for p in goal_predictions]
            ax.scatter(x, y, alpha=0.6)
            
            # Add perfect prediction line
            min_val = min(min(x), min(y))
            max_val = max(max(x), max(y))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            ax.set_xlabel('Actual Goals')
            ax.set_ylabel('Predicted Goals')
            ax.set_title('Goal Predictions vs Actual')
            
            # Add error histogram
            ax = axes[1, 1]
            errors = [p['predicted_goals'] - p['actual_goals'] for p in goal_predictions]
            ax.hist(errors, bins=20, alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--')
            ax.set_xlabel('Prediction Error (Predicted - Actual)')
            ax.set_ylabel('Frequency')
            ax.set_title('Goal Prediction Error Distribution')
        
        plt.tight_layout()
        
        # Save or show
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Saved performance visualization to {output_file}")
        else:
            output_file = os.path.join(RESULTS_DIR, 'prediction_performance.png')
            plt.savefig(output_file)
            logger.info(f"Saved performance visualization to {output_file}")
        
        plt.close()

class RLAgent:
    """
    Reinforcement Learning Agent that learns to adjust prediction strategies
    based on past performance.
    """
    def __init__(self, model_file=None):
        self.model_file = model_file or os.path.join(RL_DIR, 'rl_agent_model.pkl')
        self.q_table = self._load_model()
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.2
        self.min_exploration_rate = 0.01
        self.exploration_decay = 0.995
        
    def _load_model(self):
        """Load Q-table from file or initialize if not exists"""
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    q_table = pickle.load(f)
                logger.info(f"Loaded RL agent model with {len(q_table)} state-action pairs")
                return q_table
            except Exception as e:
                logger.error(f"Error loading RL agent model: {e}")
                return {}
        else:
            logger.info("No RL agent model found, initializing empty Q-table")
            return {}
    
    def _save_model(self):
        """Save Q-table to file"""
        try:
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.q_table, f)
            logger.info(f"Saved RL agent model with {len(self.q_table)} state-action pairs")
        except Exception as e:
            logger.error(f"Error saving RL agent model: {e}")
    
    def _state_to_key(self, state):
        """Convert state dictionary to hashable key"""
        # Discretize continuous values
        discretized = {}
        for key, value in state.items():
            if isinstance(value, (int, float)):
                # Discretize to 10 levels
                discretized[key] = round(value * 10) / 10
            else:
                discretized[key] = value
        
        # Create sorted tuple of key-value pairs
        return tuple(sorted(discretized.items()))
    
    def get_action(self, state, available_actions):
        """
        Select action based on current state using epsilon-greedy policy
        
        Args:
            state: Current environment state
            available_actions: List of available actions
        
        Returns:
            selected_action: The selected action
        """
        state_key = self._state_to_key(state)
        
        # Exploration: random action
        if np.random.random() < self.exploration_rate:
            return np.random.choice(available_actions)
        
        # Exploitation: best known action
        if state_key in self.q_table:
            q_values = self.q_table[state_key]
            # Filter to only available actions
            available_q = {a: q_values.get(a, 0) for a in available_actions}
            if available_q:
                return max(available_q, key=available_q.get)
        
        # If state not in Q-table or no Q-values for available actions, choose random
        return np.random.choice(available_actions)
    
    def update(self, state, action, reward, next_state, available_next_actions):
        """
        Update Q-table based on experience
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            available_next_actions: Actions available in next state
        """
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        # Initialize state in Q-table if not exists
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        # Initialize next state in Q-table if not exists
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {}
        
        # Get current Q-value
        current_q = self.q_table[state_key].get(action, 0)
        
        # Get max Q-value for next state
        next_q_values = self.q_table[next_state_key]
        # Filter to only available actions
        available_next_q = {a: next_q_values.get(a, 0) for a in available_next_actions}
        max_next_q = max(available_next_q.values()) if available_next_q else 0
        
        # Update Q-value
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
        
        # Decay exploration rate
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)
        
        # Save model periodically
        if np.random.random() < 0.1:  # 10% chance to save
            self._save_model()
    
    def adjust_prediction(self, base_prediction, state):
        """
        Adjust prediction based on learned strategy
        
        Args:
            base_prediction: Base prediction from statistical model
            state: Current environment state
        
        Returns:
            adjusted_prediction: Adjusted prediction
        """
        # Define possible adjustment actions
        actions = [
            'no_change',
            'increase_disposal_small',
            'increase_disposal_medium',
            'increase_disposal_large',
            'decrease_disposal_small',
            'decrease_disposal_medium',
            'decrease_disposal_large',
            'increase_goal_small',
            'increase_goal_medium',
            'increase_goal_large',
            'decrease_goal_small',
            'decrease_goal_medium',
            'decrease_goal_large',
        ]
        
        # Get action from agent
        action = self.get_action(state, actions)
        
        # Apply adjustment based on action
        adjusted_prediction = base_prediction.copy()
        
        if action == 'no_change':
            pass  # No adjustment
        
        # Disposal adjustments
        elif action == 'increase_disposal_small':
            adjusted_prediction['predicted_disposals'] += 1
            adjusted_prediction['disposal_adjustment'] = 1
        elif action == 'increase_disposal_medium':
            adjusted_prediction['predicted_disposals'] += 2
            adjusted_prediction['disposal_adjustment'] = 2
        elif action == 'increase_disposal_large':
            adjusted_prediction['predicted_disposals'] += 3
            adjusted_prediction['disposal_adjustment'] = 3
        elif action == 'decrease_disposal_small':
            adjusted_prediction['predicted_disposals'] = max(0, adjusted_prediction['predicted_disposals'] - 1)
            adjusted_prediction['disposal_adjustment'] = -1
        elif action == 'decrease_disposal_medium':
            adjusted_prediction['predicted_disposals'] = max(0, adjusted_prediction['predicted_disposals'] - 2)
            adjusted_prediction['disposal_adjustment'] = -2
        elif action == 'decrease_disposal_large':
            adjusted_prediction['predicted_disposals'] = max(0, adjusted_prediction['predicted_disposals'] - 3)
            adjusted_prediction['disposal_adjustment'] = -3
        
        # Goal adjustments
        elif action == 'increase_goal_small':
            adjusted_prediction['predicted_goals'] += 0.5
            adjusted_prediction['goal_adjustment'] = 0.5
        elif action == 'increase_goal_medium':
            adjusted_prediction['predicted_goals'] += 1
            adjusted_prediction['goal_adjustment'] = 1
        elif action == 'increase_goal_large':
            adjusted_prediction['predicted_goals'] += 1.5
            adjusted_prediction['goal_adjustment'] = 1.5
        elif action == 'decrease_goal_small':
            adjusted_prediction['predicted_goals'] = max(0, adjusted_prediction['predicted_goals'] - 0.5)
            adjusted_prediction['goal_adjustment'] = -0.5
        elif action == 'decrease_goal_medium':
            adjusted_prediction['predicted_goals'] = max(0, adjusted_prediction['predicted_goals'] - 1)
            adjusted_prediction['goal_adjustment'] = -1
        elif action == 'decrease_goal_large':
            adjusted_prediction['predicted_goals'] = max(0, adjusted_prediction['predicted_goals'] - 1.5)
            adjusted_prediction['goal_adjustment'] = -1.5
        
        # Record action taken
        adjusted_prediction['rl_action'] = action
        
        return adjusted_prediction

def load_multi_task_model():
    """
    Load the multi-task learning model
    
    Returns:
        model: Loaded multi-task model
    """
    model_path = os.path.join(MODELS_DIR, 'multi_task_model.h5')
    pipeline_path = os.path.join(MODELS_DIR, 'multi_task_prediction_pipeline.pkl')
    
    if os.path.exists(pipeline_path):
        try:
            with open(pipeline_path, 'rb') as f:
                pipeline = pickle.load(f)
            logger.info("Loaded multi-task prediction pipeline")
            return pipeline
        except Exception as e:
            logger.error(f"Error loading multi-task prediction pipeline: {e}")
    
    logger.error("Multi-task model not found")
    return None

def make_base_predictions(multi_task_model, player_data):
    """
    Make base predictions using the multi-task learning model
    
    Args:
        multi_task_model: Loaded multi-task model
        player_data: DataFrame of player data
    
    Returns:
        predictions: List of prediction dictionaries
    """
    if multi_task_model is None:
        logger.error("No multi-task model available for predictions")
        return []
    
    try:
        # Extract components from pipeline
        model = multi_task_model['model']
        preprocessor = multi_task_model['preprocessor']
        feature_cols = multi_task_model['feature_cols']
        
        # Prepare features
        X = player_data[feature_cols].copy()
        X_scaled = preprocessor.transform(X)
        
        # Make predictions
        raw_predictions = model.predict(X_scaled)
        
        # Format predictions
        predictions = []
        for i, row in player_data.iterrows():
            prediction = {
                'player_name': row['player_name'],
                'team': row['team'],
                'opponent': row['opponent'],
                'match_date': row['match_date'].isoformat() if isinstance(row['match_date'], datetime) else row['match_date'],
                'predicted_disposals': float(raw_predictions['disposals'][i][0]),
                'predicted_goals': float(raw_predictions['goals'][i][0]),
                'disposal_confidence': 0.8,  # Default confidence
                'goal_confidence': 0.7,  # Default confidence
            }
            predictions.append(prediction)
        
        logger.info(f"Made {len(predictions)} base predictions using multi-task model")
        return predictions
    
    except Exception as e:
        logger.error(f"Error making base predictions: {e}")
        return []

def apply_reinforcement_learning(base_predictions, rl_agent, environment):
    """
    Apply reinforcement learning to adjust base predictions
    
    Args:
        base_predictions: List of base predictions from statistical model
        rl_agent: Reinforcement learning agent
        environment: Prediction environment
    
    Returns:
        adjusted_predictions: List of adjusted predictions
    """
    adjusted_predictions = []
    
    for base_prediction in base_predictions:
        # Get current state
        state = environment.get_state()
        
        # Add player-specific state information if available
        player_name = base_prediction['player_name']
        player_predictions = [p for p in environment.prediction_history 
                             if p['player_name'] == player_name]
        
        if player_predictions:
            # Calculate player-specific metrics
            player_disposal_predictions = [p for p in player_predictions 
                                         if 'actual_disposals' in p and p['actual_disposals'] is not None]
            
            if player_disposal_predictions:
                # Calculate recent accuracy
                recent_errors = [abs(p['predicted_disposals'] - p['actual_disposals']) 
                               for p in player_disposal_predictions[-5:]]
                avg_error = sum(recent_errors) / len(recent_errors)
                max_error = 15
                player_accuracy = max(0, 1 - (avg_error / max_error))
                
                # Add to state
                state['player_specific_accuracy'] = player_accuracy
            
            # Check if player has played against this opponent before
            opponent = base_prediction['opponent']
            opponent_predictions = [p for p in player_predictions 
                                   if p['opponent'] == opponent and 
                                   'actual_disposals' in p and p['actual_disposals'] is not None]
            
            if opponent_predictions:
                # Calculate average performance against this opponent
                avg_disposals = sum([p['actual_disposals'] for p in opponent_predictions]) / len(opponent_predictions)
                
                # Compare to overall average
                if player_disposal_predictions:
                    overall_avg = sum([p['actual_disposals'] for p in player_disposal_predictions]) / len(player_disposal_predictions)
                    opponent_factor = avg_disposals / overall_avg if overall_avg > 0 else 1
                    
                    # Add to state
                    state['opponent_factor'] = min(2, max(0.5, opponent_factor))
        
        # Adjust prediction using RL agent
        adjusted_prediction = rl_agent.adjust_prediction(base_prediction, state)
        adjusted_predictions.append(adjusted_prediction)
    
    logger.info(f"Applied reinforcement learning to adjust {len(adjusted_predictions)} predictions")
    return adjusted_predictions

def update_rl_agent(rl_agent, environment, old_predictions, new_results):
    """
    Update RL agent based on prediction results
    
    Args:
        rl_agent: Reinforcement learning agent
        environment: Prediction environment
        old_predictions: Previous predictions
        new_results: Actual results
    """
    # Match predictions with results
    for result in new_results:
        player_name = result['player_name']
        match_date = result['match_date']
        
        # Find matching prediction
        matching_predictions = [p for p in old_predictions 
                              if p['player_name'] == player_name and 
                              p['match_date'].split('T')[0] == match_date.split('T')[0]]
        
        if not matching_predictions:
            continue
        
        prediction = matching_predictions[0]
        
        # Get state at time of prediction
        state = environment.get_state()
        
        # Get action taken
        action = prediction.get('rl_action', 'no_change')
        
        # Calculate reward
        reward = environment.get_reward(prediction, result)
        
        # Get next state (current state)
        next_state = environment.get_state()
        
        # Define available actions
        available_actions = [
            'no_change',
            'increase_disposal_small',
            'increase_disposal_medium',
            'increase_disposal_large',
            'decrease_disposal_small',
            'decrease_disposal_medium',
            'decrease_disposal_large',
            'increase_goal_small',
            'increase_goal_medium',
            'increase_goal_large',
            'decrease_goal_small',
            'decrease_goal_medium',
            'decrease_goal_large',
        ]
        
        # Update agent
        rl_agent.update(state, action, reward, next_state, available_actions)
    
    logger.info(f"Updated RL agent with {len(new_results)} new results")

def main():
    """
    Main function to run the reinforcement learning system
    """
    logger.info("Starting reinforcement learning system...")
    
    # Initialize environment and agent
    environment = PredictionEnvironment()
    rl_agent = RLAgent()
    
    # Load multi-task model
    multi_task_model = load_multi_task_model()
    
    # Check if we have new player data to make predictions
    player_data_file = os.path.join(PROCESSED_DATA_DIR, 'upcoming_players.csv')
    if os.path.exists(player_data_file):
        try:
            player_data = pd.read_csv(player_data_file)
            logger.info(f"Loaded player data for {len(player_data)} players")
            
            # Make base predictions
            base_predictions = make_base_predictions(multi_task_model, player_data)
            
            # Apply reinforcement learning
            adjusted_predictions = apply_reinforcement_learning(base_predictions, rl_agent, environment)
            
            # Save predictions
            predictions_file = os.path.join(RESULTS_DIR, 'rl_adjusted_predictions.json')
            with open(predictions_file, 'w') as f:
                json.dump(adjusted_predictions, f, indent=2)
            logger.info(f"Saved {len(adjusted_predictions)} adjusted predictions to {predictions_file}")
            
            # Add predictions to environment
            for prediction in adjusted_predictions:
                environment.add_prediction(prediction)
        
        except Exception as e:
            logger.error(f"Error processing player data: {e}")
    
    # Check if we have new results to update the agent
    results_file = os.path.join(PROCESSED_DATA_DIR, 'new_results.csv')
    if os.path.exists(results_file):
        try:
            results_data = pd.read_csv(results_file)
            logger.info(f"Loaded results data for {len(results_data)} players")
            
            # Format results
            new_results = []
            for i, row in results_data.iterrows():
                result = {
                    'player_name': row['player_name'],
                    'match_date': row['match_date'],
                    'actual_disposals': row['disposals'] if 'disposals' in row else None,
                    'actual_goals': row['goals'] if 'goals' in row else None
                }
                new_results.append(result)
            
            # Update environment with actual results
            for result in new_results:
                environment.update_actual_results(
                    result['player_name'],
                    result['match_date'],
                    result['actual_disposals'],
                    result['actual_goals']
                )
            
            # Get old predictions
            old_predictions = environment.prediction_history
            
            # Update RL agent
            update_rl_agent(rl_agent, environment, old_predictions, new_results)
            
            # Visualize performance
            environment.visualize_performance()
            
            # Get performance metrics
            metrics = environment.get_performance_metrics()
            metrics_file = os.path.join(RESULTS_DIR, 'rl_performance_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved performance metrics to {metrics_file}")
        
        except Exception as e:
            logger.error(f"Error processing results data: {e}")
    
    logger.info("Reinforcement learning system completed successfully")

if __name__ == "__main__":
    main()
