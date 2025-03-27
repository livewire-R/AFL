#!/usr/bin/env python3
"""
Create simple SVG team logos for the AFL web interface

This script creates simple SVG team logos with team initials and team colors
for use in the web interface.
"""

import os
import json

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, 'web_app/static/images/teams')

# Ensure directory exists
os.makedirs(IMAGES_DIR, exist_ok=True)

# Define AFL teams with their colors and initials
AFL_TEAMS = {
    'adelaide': {
        'name': 'Adelaide Crows',
        'initials': 'ADL',
        'colors': ['#002b5c', '#e21a23', '#ffc424']
    },
    'brisbane': {
        'name': 'Brisbane Lions',
        'initials': 'BRL',
        'colors': ['#7a0045', '#00275d', '#ffc424']
    },
    'carlton': {
        'name': 'Carlton Blues',
        'initials': 'CAR',
        'colors': ['#0e1e2d', '#0072ce']
    },
    'collingwood': {
        'name': 'Collingwood Magpies',
        'initials': 'COL',
        'colors': ['#000000', '#ffffff']
    },
    'essendon': {
        'name': 'Essendon Bombers',
        'initials': 'ESS',
        'colors': ['#000000', '#ff0000']
    },
    'fremantle': {
        'name': 'Fremantle Dockers',
        'initials': 'FRE',
        'colors': ['#2a0d54', '#ffffff']
    },
    'geelong': {
        'name': 'Geelong Cats',
        'initials': 'GEE',
        'colors': ['#1c3c63', '#ffffff']
    },
    'gold_coast': {
        'name': 'Gold Coast Suns',
        'initials': 'GCS',
        'colors': ['#e21937', '#ffc72c']
    },
    'gws': {
        'name': 'GWS Giants',
        'initials': 'GWS',
        'colors': ['#f47920', '#ffffff', '#231f20']
    },
    'hawthorn': {
        'name': 'Hawthorn Hawks',
        'initials': 'HAW',
        'colors': ['#4d2004', '#ffc424']
    },
    'melbourne': {
        'name': 'Melbourne Demons',
        'initials': 'MEL',
        'colors': ['#00246c', '#e4002b']
    },
    'north_melbourne': {
        'name': 'North Melbourne Kangaroos',
        'initials': 'NTH',
        'colors': ['#003f98', '#ffffff']
    },
    'port_adelaide': {
        'name': 'Port Adelaide Power',
        'initials': 'POR',
        'colors': ['#008aab', '#000000', '#ffffff']
    },
    'richmond': {
        'name': 'Richmond Tigers',
        'initials': 'RIC',
        'colors': ['#000000', '#ffc424']
    },
    'st_kilda': {
        'name': 'St Kilda Saints',
        'initials': 'STK',
        'colors': ['#ed1c24', '#000000', '#ffffff']
    },
    'sydney': {
        'name': 'Sydney Swans',
        'initials': 'SYD',
        'colors': ['#ed171f', '#ffffff']
    },
    'west_coast': {
        'name': 'West Coast Eagles',
        'initials': 'WCE',
        'colors': ['#003087', '#f2a900']
    },
    'western_bulldogs': {
        'name': 'Western Bulldogs',
        'initials': 'WBD',
        'colors': ['#0057b8', '#e31937', '#ffffff']
    }
}

def create_svg_logo(team_data, save_path):
    """
    Create a simple SVG logo for a team
    
    Args:
        team_data: Dictionary with team information
        save_path: Path to save the SVG file
    """
    initials = team_data['initials']
    colors = team_data['colors']
    
    # Use first color as background and second as text color
    bg_color = colors[0]
    text_color = colors[1] if len(colors) > 1 else '#ffffff'
    
    # Create SVG content
    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">
    <circle cx="50" cy="50" r="45" fill="{bg_color}" />
    <text x="50" y="65" font-family="Arial, sans-serif" font-size="30" font-weight="bold" text-anchor="middle" fill="{text_color}">{initials}</text>
</svg>"""
    
    # Save SVG file
    with open(save_path, 'w') as f:
        f.write(svg_content)
    
    print(f"Created SVG logo for {team_data['name']} at {save_path}")

def main():
    """
    Main function to create all team logos
    """
    print(f"Creating AFL team logos in {IMAGES_DIR}")
    
    # Save team data to JSON for use in web app
    teams_json_path = os.path.join(IMAGES_DIR, 'teams.json')
    with open(teams_json_path, 'w') as f:
        json.dump(AFL_TEAMS, f, indent=2)
    print(f"Saved team data to {teams_json_path}")
    
    # Create SVG logos for each team
    for team_id, team_data in AFL_TEAMS.items():
        save_path = os.path.join(IMAGES_DIR, f"{team_id}.svg")
        create_svg_logo(team_data, save_path)
    
    print(f"Created {len(AFL_TEAMS)} team logos")

if __name__ == "__main__":
    main()
