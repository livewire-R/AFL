# Historical AFL Player Statistics

This directory should contain historical player statistics from previous seasons. These files serve as the foundation for training the machine learning models.

## Expected Files

Place your historical player statistics files in this directory using the following naming convention:
- `historical_player_stats_YYYY.csv` (where YYYY is the year)

## Example File Structure

```
historical_player_stats_2020.csv
historical_player_stats_2021.csv
historical_player_stats_2022.csv
historical_player_stats_2023.csv
historical_player_stats_2024.csv
```

## Required Data Columns

For optimal model performance, your historical data files should include the following columns:
- Player name
- Team
- Opposition
- Match date
- Venue
- Disposals
- Goals
- Fantasy points (if available)
- Other relevant statistics (kicks, handballs, marks, etc.)

## Data Sources

As mentioned in your requirements, data can be sourced from:
- FootyWire.com (https://www.footywire.com/afl/footy/ft_players)
- AFL.com official website
