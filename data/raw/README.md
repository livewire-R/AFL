# AFL Data Directory

This directory contains the raw data files for the AFL prediction system. The data is organized into three main categories:

## Directory Structure

- `historical/`: Contains historical player statistics and match data from previous seasons
- `current/`: Contains the current season's player statistics and match data (updated weekly)
- `fixtures/`: Contains upcoming match fixtures and schedules

## Data Update Process

As requested, you will be manually saving the data in these directories as the rounds go past. Here's how to manage the data:

1. **Historical Data**: Place all historical player statistics in the `historical/` directory. This data doesn't need to be updated frequently as it represents past seasons.

2. **Current Data**: Update the `current/` directory weekly with the latest player statistics after each round. This ensures the prediction models have the most recent data to work with.

3. **Fixtures**: Place the upcoming fixtures in the `fixtures/` directory and update them as needed throughout the season.

## File Formats

The system supports the following file formats:
- CSV files (.csv)
- Excel files (.xlsx)
- JSON files (.json)

## Recommended Naming Convention

To maintain consistency, please use the following naming conventions:

- Historical data: `historical_player_stats_YYYY.csv` (where YYYY is the year)
- Current data: `player_stats_round_XX_YYYY.csv` (where XX is the round number and YYYY is the year)
- Fixtures: `fixtures_round_XX_YYYY.csv` (where XX is the round number and YYYY is the year)

## Data Processing

After you manually update these raw data files, the system will automatically process them when you run the appropriate scripts. The processed data will be stored in the `../processed/` directory.
