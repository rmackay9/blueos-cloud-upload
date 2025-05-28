#!/usr/bin/env python3

import json
import os
import logging
from pathlib import Path

logger = logging.getLogger("cloud_upload.settings")

# Settings file path - stored in the extension's persistent storage directory
SETTINGS_FILE = Path('/app/settings/cloud_upload_settings.json')

# Default settings
DEFAULT_SETTINGS = {
    'cloud': {
        'provider': 'google',
        'username': '',
        'password': '',
        'directories': ['/app/logs']
    }
}


# get the dictionary of settings from the settings file
def get_settings():
    """
    Load settings from the settings file.
    Creates default settings file if it doesn't exist.

    Returns:
        dict: The settings dictionary
    """
    try:
        if not SETTINGS_FILE.exists():
            logger.info(f"Settings file not found, creating default at {SETTINGS_FILE}")
            save_settings(DEFAULT_SETTINGS)
            return DEFAULT_SETTINGS

        with open(SETTINGS_FILE, 'r') as f:
            settings = json.load(f)
            logger.debug(f"Loaded settings: {settings}")
            return settings
    except Exception as e:
        logger.error(f"Error loading settings: {e}")
        logger.info("Using default settings")
        # Try to save default settings for next time
        try:
            save_settings(DEFAULT_SETTINGS)
        except Exception:
            logger.exception("Failed to save default settings")

        return DEFAULT_SETTINGS


# save settings to the settings file
def save_settings(settings):
    """
    Save settings to the settings file

    Args:
        settings (dict): Settings dictionary to save
    """
    try:
        # Ensure parent directory exists
        os.makedirs(SETTINGS_FILE.parent, exist_ok=True)

        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
            logger.debug(f"Saved settings: {settings}")
    except Exception as e:
        logger.error(f"Error saving settings: {e}")

# Cloud settings management functions
def update_cloud_settings(provider, username, password, directories):
    """
    Update cloud provider settings

    Args:
        provider (str): Cloud provider name
        username (str): Username/email for the provider
        password (str): Password/API key for the provider
        directories (list): List of directories to upload

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        settings = get_settings()

        # Update cloud settings
        settings['cloud'] = {
            'provider': provider,
            'username': username,
            'password': password,
            'directories': directories
        }

        save_settings(settings)
        logger.info(f"Cloud settings updated for provider: {provider}")
        return True
    except Exception as e:
        logger.error(f"Error updating cloud settings: {e}")
        return False


def get_cloud_settings():
    """
    Get saved cloud provider settings

    Returns:
        dict: Cloud settings dictionary
    """
    settings = get_settings()
    return settings.get('cloud', DEFAULT_SETTINGS['cloud'])


def get_cloud_directories():
    """
    Get the list of directories to upload

    Returns:
        list: List of directory paths
    """
    cloud_settings = get_cloud_settings()
    return cloud_settings.get('directories', [])


def get_cloud_provider():
    """
    Get the selected cloud provider

    Returns:
        str: Cloud provider name
    """
    cloud_settings = get_cloud_settings()
    return cloud_settings.get('provider', 'google')


def get_cloud_credentials():
    """
    Get cloud provider credentials

    Returns:
        tuple: (username, password)
    """
    cloud_settings = get_cloud_settings()
    return cloud_settings.get('username', ''), cloud_settings.get('password', '')
