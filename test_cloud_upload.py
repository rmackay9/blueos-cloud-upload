#!/usr/bin/env python3
"""
Unit tests for the cloud upload functionality
"""

import unittest
import json
import tempfile
import os
from pathlib import Path
import sys

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from settings import (
    get_settings, save_settings, update_cloud_settings,
    get_cloud_settings, get_cloud_directories, get_cloud_provider
)


class TestCloudUploadSettings(unittest.TestCase):
    """Test cloud upload settings functionality"""

    def setUp(self):
        """Set up test environment"""
        # Create a temporary settings file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.original_settings_file = None

    def tearDown(self):
        """Clean up test environment"""
        # Clean up temporary files
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_default_settings(self):
        """Test that default settings are properly structured"""
        settings = get_cloud_settings()

        self.assertIn('provider', settings)
        self.assertIn('username', settings)
        self.assertIn('password', settings)
        self.assertIn('directories', settings)

        # Check default values
        self.assertEqual(settings['provider'], 'google')
        self.assertIsInstance(settings['directories'], list)

    def test_update_cloud_settings(self):
        """Test updating cloud settings"""
        # Test data
        provider = 'dropbox'
        username = 'test@example.com'
        password = 'test_access_token'
        directories = ['/home/pi/videos', '/home/pi/logs']

        # Update settings
        result = update_cloud_settings(provider, username, password, directories)
        self.assertTrue(result)

        # Verify settings were saved
        settings = get_cloud_settings()
        self.assertEqual(settings['provider'], provider)
        self.assertEqual(settings['username'], username)
        self.assertEqual(settings['password'], password)
        self.assertEqual(settings['directories'], directories)

    def test_get_cloud_provider(self):
        """Test getting cloud provider"""
        # Test with default settings
        provider = get_cloud_provider()
        self.assertEqual(provider, 'google')

        # Test after updating
        update_cloud_settings('dropbox', 'test@example.com', 'token', [])
        provider = get_cloud_provider()
        self.assertEqual(provider, 'dropbox')

    def test_get_cloud_directories(self):
        """Test getting cloud directories"""
        test_directories = ['/test/dir1', '/test/dir2']
        update_cloud_settings('dropbox', 'test@example.com', 'token', test_directories)

        directories = get_cloud_directories()
        self.assertEqual(directories, test_directories)


class TestCloudUploadValidation(unittest.TestCase):
    """Test cloud upload validation functionality"""

    def test_dropbox_available_import(self):
        """Test that dropbox can be imported if available"""
        try:
            import dropbox
            self.assertTrue(True, "Dropbox library is available")
        except ImportError:
            self.skipTest("Dropbox library not installed")

    def test_required_packages(self):
        """Test that required packages can be imported"""
        try:
            import fastapi
            import uvicorn
            import pydantic
            self.assertTrue(True, "All required packages are available")
        except ImportError as e:
            self.fail(f"Required package not available: {e}")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
