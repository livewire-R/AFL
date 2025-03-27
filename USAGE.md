# AFL Prediction System - Usage Instructions

This document provides instructions on how to use the enhanced AFL prediction system.

## System Overview

The AFL prediction system uses machine learning to predict player disposals and goals for upcoming AFL matches. The system features:

- Multi-task learning for simultaneous prediction of disposals and goals
- Reinforcement learning that improves prediction strategies over time
- Interactive web interface with team logos and visualizations
- Support for manual data updates as rounds progress
- Comprehensive data field mapping with standardized abbreviations

## Directory Structure

```
AFL/
├── data/
│   ├── raw/             # Raw data files (manually updated)
│   │   ├── historical/  # Historical player statistics
│   │   ├── current/     # Current season statistics (update weekly)
│   │   ├── fixtures/    # Upcoming match fixtures
│   │   └── templates/   # Template CSV files
│   └── processed/       # Processed data for ML models
├── models/              # Trained ML models
├── scripts/             # Python scripts
├── web_app/             # Flask web application
└── results/             # Prediction results and visualizations
```

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Initialize the database:
   ```
   cd web_app
   python app.py
   ```

## Updating Data

### Weekly Data Updates

1. After each round, prepare the player statistics in CSV format
2. Save the file in `data/raw/current/` using the naming convention:
   ```
   player_stats_round_XX_YYYY.csv
   ```
   Where XX is the round number and YYYY is the year

3. Update fixtures as needed in `data/raw/fixtures/` using the naming convention:
   ```
   fixtures_round_XX_YYYY.csv
   ```

4. The system will automatically incorporate this new data when generating predictions

### Data Format

Use the template files in `data/raw/templates/` as a guide for the expected data format:

- `historical_player_stats_template.csv`: Template for historical player statistics
- `player_stats_template.csv`: Template for current season player statistics
- `fixtures_template.csv`: Template for fixture data

For a comprehensive reference of all data field abbreviations and naming conventions, see the `data/ABBREVIATIONS.md` file. This document explains all field names used in the data files, including:

- General statistics (Age, Player Rating, etc.)
- Disposal statistics (Kicks, Handballs, etc.)
- Possession statistics (Contested, Uncontested, etc.)
- Clearance statistics (Centre, Stoppage, etc.)
- Marking statistics (Contested Marks, Marks Inside 50, etc.)
- Scoring statistics (Goals, Behinds, etc.)
- Expected Score (xScore) statistics
- Defensive statistics
- Ruck contest statistics
- And more

## Running the System

### Generate Predictions

To generate predictions using the multi-task learning model:

```
cd AFL
python scripts/multi_task_learning.py
```

### Apply Reinforcement Learning

To apply reinforcement learning to adjust predictions:

```
cd AFL
python scripts/reinforcement_learning.py
```

### Start the Web Application

To start the web interface:

```
cd AFL/web_app
python app.py
```

Then access the web interface at http://localhost:5000

## Web Interface Features

### Home Page

- Dashboard with system overview
- Upcoming fixtures with team logos
- Top disposal and goal predictions
- Recent updates and system performance

### Fixtures Page

- List of all upcoming fixtures
- Team logos and match details
- Links to detailed fixture pages

### Predictions Page

- Comprehensive list of player predictions
- Sortable by player, team, or prediction value
- Confidence indicators for each prediction
- Interactive elements to explore prediction details

### Player Details

- Individual player statistics and form
- Historical performance visualization
- Prediction history and accuracy

## Advanced Features

### Multi-Task Learning

The multi-task learning model simultaneously predicts disposals and goals, leveraging shared patterns between these statistics. This approach improves overall prediction accuracy compared to separate models.

### Reinforcement Learning

The reinforcement learning system continuously improves prediction strategies based on past performance. It:

1. Tracks prediction accuracy over time
2. Identifies patterns in prediction errors
3. Adjusts future predictions to minimize errors
4. Visualizes performance improvements

## Troubleshooting

### Common Issues

- **Missing data files**: Ensure all required CSV files are in the correct directories
- **Incorrect data format**: Check against the template files in `data/raw/templates/`
- **Web interface not showing team logos**: Verify that the SVG files exist in `web_app/static/images/teams/`

### Error Logs

Check the following locations for error logs:

- Console output when running scripts
- Flask application logs in the terminal
- Browser console for JavaScript errors

## Maintenance

### Regular Tasks

- Update player statistics weekly in `data/raw/current/`
- Update fixtures as needed in `data/raw/fixtures/`
- Periodically retrain models to incorporate new data

### System Backup

Regularly backup the following directories:

- `data/raw/`: Contains all manually updated data
- `models/`: Contains trained models
- `web_app/afl_prediction.db`: Contains user data and saved predictions
