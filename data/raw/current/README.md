# Current Season AFL Player Statistics

This directory should contain the current season's player statistics that you will update manually each week as the rounds progress.

## Expected Files

Place your current season player statistics files in this directory using the following naming convention:
- `player_stats_round_XX_YYYY.csv` (where XX is the round number and YYYY is the year)

## Example File Structure

```
player_stats_round_01_2025.csv
player_stats_round_02_2025.csv
player_stats_round_03_2025.csv
...
```

## Required Data Columns

For optimal model performance, your current data files should include the following columns:
- Player name
- Team
- Opposition
- Match date
- Venue
- Disposals
- Goals
- Fantasy points (if available)
- Other relevant statistics (kicks, handballs, marks, etc.)

## Update Process

1. After each round is completed, download or prepare the player statistics data
2. Save the file in this directory using the naming convention above
3. The system will automatically incorporate this new data when generating predictions

## Data Sources

As mentioned in your requirements, data can be sourced from:
- FootyWire.com (https://www.footywire.com/afl/footy/ft_players)
- AFL.com official website
