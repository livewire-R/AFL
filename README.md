# AFL Prediction System - README

## Overview

This enhanced AFL prediction system uses advanced machine learning techniques to predict player disposals and goals for upcoming AFL matches. The system features multi-task learning, reinforcement learning, and an interactive web interface.

## Key Features

- **Multi-task Learning**: Simultaneously predicts disposals and goals with improved accuracy
- **Reinforcement Learning**: Learns optimal prediction strategies over time based on past performance
- **Manual Data Updates**: Structured directories for updating data as rounds progress
- **Interactive Web Interface**: Enhanced with team logos, animations, and visualizations
- **Comprehensive Data Field Mapping**: Standardized abbreviations and naming conventions for all AFL statistics

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
│   ├── multi_task_learning.py       # Multi-task learning implementation
│   ├── reinforcement_learning.py    # Reinforcement learning implementation
│   ├── create_team_logos.py         # Script to create team logos
│   └── [other scripts]              # Original and supporting scripts
├── web_app/             # Flask web application
│   ├── static/          # Static assets (CSS, JS, images)
│   ├── templates/       # HTML templates
│   └── app.py           # Flask application
├── CHANGES.md           # Detailed description of all changes and enhancements
├── USAGE.md             # Comprehensive usage instructions
└── README.md            # This file
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

## Quick Start

1. Start the web application:
   ```
   cd web_app
   python app.py
   ```
2. Access the web interface at http://localhost:5000
3. Update data weekly in the `data/raw/current/` directory
4. Run prediction scripts as needed:
   ```
   python scripts/multi_task_learning.py
   python scripts/reinforcement_learning.py
   ```

## Documentation

- **CHANGES.md**: Detailed description of all changes and enhancements made to the system
- **USAGE.md**: Comprehensive instructions on how to use the system

## Technologies Used

- Python 3.10
- TensorFlow for multi-task learning
- Flask web framework
- SQLAlchemy ORM
- Bootstrap 5 for frontend styling
- Chart.js for data visualization

## Data Sources

Player statistics should be sourced from FootyWire.com or the official AFL website and manually saved to the appropriate directories.
