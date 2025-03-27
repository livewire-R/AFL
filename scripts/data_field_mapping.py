#!/usr/bin/env python3
"""
Data field mapping for AFL prediction system

This script provides mappings between abbreviated field names and full column names
in the AFL data files. It's used by other scripts to ensure consistent handling of
data fields across the system.
"""

# Mapping from abbreviated names to full column names
FIELD_MAPPING = {
    # General
    'Age': 'Age',
    'Player Rating': 'RatingPoints_Avg',
    'CV Avg': 'CoachesVotes_Avg',
    'TOG': 'TimeOnGround',
    
    # Disposals
    'Kick': 'Kicks',
    'HB': 'Handballs',
    'Dis.': 'Disposals',
    'Eff %': 'DisposalEfficiency',
    'Kick Eff %': 'KickingEfficiency',
    'HB Eff %': 'HandballEfficiency',
    'Kick %': 'KickPercentage',
    'In 50s': 'Inside50s',
    'Reb 50s': 'Rebound50s',
    'Mtrs Gnd': 'MetresGained',
    'Mtrs / D': 'MetresGainedPerDisposal',
    'Clng': 'Clangers',
    'D/Cg': 'DisposalsPerClanger',
    'TO': 'Turnovers',
    'D/TO': 'DisposalsPerTurnover',
    
    # Possessions
    'CP': 'ContestedPossessions',
    'UP': 'UncontestedPossessions',
    'Poss': 'TotalPossessions',
    'CP%': 'ContestedPossessionRate',
    'Int Poss': 'Intercepts',
    'GB Gets': 'GroundBallGets',
    'F50 GBG': 'GroundBallGetsForward50',
    'Hard Gets': 'HardBallGets',
    'Loose Gets': 'LooseBallGets',
    'Post Cl CP': 'PostClearanceContestedPossessions',
    'Post Cl GBG': 'PostClearanceGroundBallGets',
    'Gath HO': 'GathersFromHitout',
    'Crumb': 'CrumbingPossessions',
    'HB Rec': 'HandballReceives',
    
    # Clearances
    'CBA': 'CentreBounceAttendances',
    'CBA %': 'CentreBounceAttendancePercentage',
    'Ctr Clr': 'CentreClearances',
    'CC / CBA': 'CentreClearanceRate',
    'Stp Clr': 'StoppageClearances',
    'Tot Clr': 'TotalClearances',
    
    # Marks
    'Tot Marks': 'Marks',
    'CM': 'ContestedMarks',
    'In 50': 'MarksInside50',
    'Int Mks': 'InterceptMarks',
    'On lead': 'MarksOnLead',
    
    # Scoring
    'Goals': 'Goals_Total',
    'Gls Avg': 'Goals_Avg',
    'Beh': 'Behinds',
    'Shots': 'ShotsAtGoal',
    'Goal Ass.': 'GoalAssists',
    'Acc': 'GoalAccuracy',
    'SI': 'ScoreInvolvements',
    'SI %': 'ScoreInvolvementPercentage',
    'Launch': 'ScoreLaunches',
    'Off 1v1': 'ContestOffensiveOneOnOnes',
    'Win %': 'ContestOffensiveWinPercentage',
    
    # Expected Scores
    'xSc Shots': 'xScoreShots',
    'xSc': 'xScorePerShot',
    'Rating': 'xScoreRatingPerShot',
    'Shots (set)': 'xScoreShots_Set',
    'xSc (Set)': 'xScorePerShot_Set',
    'Rating (Set)': 'xScoreRatingPerShot_Set',
    'Shots (Gen)': 'xScoreShots_General',
    'xSc (Gen)': 'xScorePerShot_General',
    'Rating (Gen)': 'xScoreRatingPerShot_General',
    
    # Defence
    'Def 1v1': 'ContestDefensiveOneOnOnes',
    'Loss %': 'ContestDefensiveLossPercentage',
    'Tack': 'Tackles',
    'Tack In 50': 'TacklesInside50',
    'Press. Acts': 'PressureActs',
    'Def. PA': 'PressureActsDefensiveHalf',
    
    # Ruck contests
    'Ruck Cont': 'RuckContests',
    'RC %': 'RuckContestPercentage',
    'Hit Outs': 'Hitouts',
    'Win % (Ruck)': 'HitoutsWinPercentage',
    'To Adv': 'HitoutsToAdvantage',
    'Adv %': 'HitoutsToAdvantagePercentage',
    'Ruck HBG': 'RuckHardBallGets',
    
    # Other
    'Kick In %': 'KickInPercentage',
    'KI Play On %': 'KickInsPlayOnPercentage',
    'Bounce': 'Bounces',
    '1%s': 'OnePercenters',
}

# Reverse mapping from full column names to abbreviated names
REVERSE_FIELD_MAPPING = {v: k for k, v in FIELD_MAPPING.items()}

# Key fields used for multi-task learning model
MULTI_TASK_FIELDS = {
    # Input features for the model
    'features': [
        'Kicks', 'Handballs', 'Disposals', 'DisposalEfficiency', 
        'Inside50s', 'Rebound50s', 'MetresGained', 'ContestedPossessions',
        'UncontestedPossessions', 'TotalPossessions', 'Marks', 'ContestedMarks',
        'MarksInside50', 'Tackles', 'TacklesInside50', 'TimeOnGround',
        'CentreClearances', 'StoppageClearances', 'TotalClearances'
    ],
    
    # Target variables for prediction
    'targets': {
        'disposals': 'Disposals',
        'goals': 'Goals_Total'
    }
}

# Fields used for reinforcement learning
RL_FIELDS = {
    # State representation fields
    'state_fields': [
        'Disposals', 'Goals_Total', 'TimeOnGround', 'ContestedPossessions',
        'UncontestedPossessions', 'Inside50s', 'Marks', 'MarksInside50',
        'TotalClearances', 'Tackles'
    ],
    
    # Reward calculation fields
    'reward_fields': {
        'disposals': 'Disposals',
        'goals': 'Goals_Total'
    }
}

def get_full_column_name(abbreviated_name):
    """
    Get the full column name for an abbreviated field name
    
    Args:
        abbreviated_name: The abbreviated field name
        
    Returns:
        The full column name or the original name if not found
    """
    return FIELD_MAPPING.get(abbreviated_name, abbreviated_name)

def get_abbreviated_name(full_column_name):
    """
    Get the abbreviated name for a full column name
    
    Args:
        full_column_name: The full column name
        
    Returns:
        The abbreviated name or the original name if not found
    """
    return REVERSE_FIELD_MAPPING.get(full_column_name, full_column_name)

def get_multi_task_feature_columns():
    """
    Get the list of column names used as features in the multi-task learning model
    
    Returns:
        List of column names
    """
    return MULTI_TASK_FIELDS['features']

def get_multi_task_target_columns():
    """
    Get the dictionary of target column names for the multi-task learning model
    
    Returns:
        Dictionary mapping target names to column names
    """
    return MULTI_TASK_FIELDS['targets']

def get_rl_state_columns():
    """
    Get the list of column names used for state representation in reinforcement learning
    
    Returns:
        List of column names
    """
    return RL_FIELDS['state_fields']

def get_rl_reward_columns():
    """
    Get the dictionary of column names used for reward calculation in reinforcement learning
    
    Returns:
        Dictionary mapping reward names to column names
    """
    return RL_FIELDS['reward_fields']
