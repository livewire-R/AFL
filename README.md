# AFL Prediction System

A machine learning system that predicts AFL player disposals and goals using data from wheeloratings.com.

## Features

- Predicts player disposals and goals for upcoming AFL matches
- Updates weekly to incorporate the latest form data
- Modern web interface with user authentication
- Visualizes player form and prediction confidence
- Allows users to save and track predictions

## Project Structure

```
afl_prediction_project/
├── data/
│   ├── raw/             # Raw data downloaded from wheeloratings.com
│   └── processed/       # Processed data for ML models
├── models/              # Trained ML models
├── scripts/
│   ├── preprocess_data.py
│   ├── analyze_features.py
│   ├── model_development.py
│   ├── evaluate_models.py
│   ├── weekly_update.py
│   └── exploratory_data_analysis.py
└── web_app/
    ├── app.py           # Flask application
    ├── static/
    │   ├── css/
    │   ├── js/
    │   └── images/
    └── templates/       # HTML templates
```

## Installation

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

## Usage

1. Start the web application:
   ```
   cd web_app
   python app.py
   ```
2. Access the web interface at http://localhost:5000
3. Register an account and login
4. View predictions for upcoming fixtures
5. Save predictions to your profile

## Weekly Updates

The system automatically updates weekly when new games are played. The update process:

1. Downloads the latest player statistics from wheeloratings.com
2. Calculates form metrics based on recent performances
3. Generates new predictions for upcoming fixtures
4. Updates the web application with the latest data

## Technologies Used

- Python 3.10
- Flask web framework
- SQLAlchemy ORM
- Scikit-learn for machine learning
- Pandas for data processing
- Matplotlib and Seaborn for visualization
- Bootstrap 5 for frontend styling

## Data Source

All player statistics are sourced from [wheeloratings.com](https://www.wheeloratings.com/afl_index.html), which provides comprehensive AFL player data from 2012-2025.
