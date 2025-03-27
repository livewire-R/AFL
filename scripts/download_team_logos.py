#!/usr/bin/env python3
"""
Download AFL team logos for the web interface

This script downloads AFL team logos from the internet and saves them to the
static/images/teams directory for use in the web interface.
"""

import os
import requests
from urllib.parse import urlparse
import time

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, 'web_app/static/images/teams')

# Ensure directory exists
os.makedirs(IMAGES_DIR, exist_ok=True)

# Define AFL teams and their logo URLs
AFL_TEAMS = {
    'adelaide': 'https://resources.afl.com.au/afl/photos/afl-logo-Adelaide.png',
    'brisbane': 'https://resources.afl.com.au/afl/photos/afl-logo-Brisbane.png',
    'carlton': 'https://resources.afl.com.au/afl/photos/afl-logo-Carlton.png',
    'collingwood': 'https://resources.afl.com.au/afl/photos/afl-logo-Collingwood.png',
    'essendon': 'https://resources.afl.com.au/afl/photos/afl-logo-Essendon.png',
    'fremantle': 'https://resources.afl.com.au/afl/photos/afl-logo-Fremantle.png',
    'geelong': 'https://resources.afl.com.au/afl/photos/afl-logo-Geelong.png',
    'gold_coast': 'https://resources.afl.com.au/afl/photos/afl-logo-Gold-Coast.png',
    'gws': 'https://resources.afl.com.au/afl/photos/afl-logo-GWS.png',
    'hawthorn': 'https://resources.afl.com.au/afl/photos/afl-logo-Hawthorn.png',
    'melbourne': 'https://resources.afl.com.au/afl/photos/afl-logo-Melbourne.png',
    'north_melbourne': 'https://resources.afl.com.au/afl/photos/afl-logo-North-Melbourne.png',
    'port_adelaide': 'https://resources.afl.com.au/afl/photos/afl-logo-Port-Adelaide.png',
    'richmond': 'https://resources.afl.com.au/afl/photos/afl-logo-Richmond.png',
    'st_kilda': 'https://resources.afl.com.au/afl/photos/afl-logo-St-Kilda.png',
    'sydney': 'https://resources.afl.com.au/afl/photos/afl-logo-Sydney.png',
    'west_coast': 'https://resources.afl.com.au/afl/photos/afl-logo-West-Coast.png',
    'western_bulldogs': 'https://resources.afl.com.au/afl/photos/afl-logo-Western-Bulldogs.png'
}

def download_image(url, save_path):
    """
    Download an image from a URL and save it to the specified path
    
    Args:
        url: URL of the image to download
        save_path: Path to save the image to
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded {url} to {save_path}")
        return True
    
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def main():
    """
    Main function to download all team logos
    """
    print(f"Downloading AFL team logos to {IMAGES_DIR}")
    
    success_count = 0
    for team_name, logo_url in AFL_TEAMS.items():
        # Get file extension from URL
        parsed_url = urlparse(logo_url)
        file_ext = os.path.splitext(parsed_url.path)[1]
        
        # Create save path
        save_path = os.path.join(IMAGES_DIR, f"{team_name}{file_ext}")
        
        # Download image
        if download_image(logo_url, save_path):
            success_count += 1
        
        # Add a small delay to avoid overwhelming the server
        time.sleep(0.5)
    
    print(f"Downloaded {success_count} of {len(AFL_TEAMS)} team logos")

if __name__ == "__main__":
    main()
