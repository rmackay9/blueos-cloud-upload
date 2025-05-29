#!/usr/bin/env python3
"""
Simple test script for Dropbox upload functionality
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the app directory to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent / "app"))

try:
    from main import test_dropbox_connection, upload_file_to_dropbox
    import dropbox
    DROPBOX_AVAILABLE = True
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install dropbox fastapi uvicorn pydantic")
    DROPBOX_AVAILABLE = False
    sys.exit(1)


def test_dropbox_connection_sync(access_token: str):
    """Test Dropbox connection synchronously"""
    print("Testing Dropbox connection...")
    success, message = test_dropbox_connection(access_token)
    print(f"Connection test: {'✓' if success else '✗'} {message}")
    return success


async def test_file_upload(access_token: str, test_file_path: str):
    """Test file upload to Dropbox"""
    if not Path(test_file_path).exists():
        print(f"Test file not found: {test_file_path}")
        return False

    print(f"Testing file upload: {test_file_path}")

    try:
        dbx = dropbox.Dropbox(access_token)
        file_path = Path(test_file_path)
        remote_path = f"/test_uploads/{file_path.name}"

        success, message = await upload_file_to_dropbox(dbx, file_path, remote_path)
        print(f"Upload test: {'✓' if success else '✗'} {message}")
        return success
    except Exception as e:
        print(f"Upload test: ✗ Error: {e}")
        return False


def main():
    """Main test function"""
    print("Dropbox Upload Test Script")
    print("=" * 30)

    if not DROPBOX_AVAILABLE:
        print("Dropbox library not available!")
        return

    # Get access token from environment variable or user input
    access_token = os.getenv('DROPBOX_ACCESS_TOKEN')

    if not access_token:
        print("Please provide a Dropbox access token:")
        print("1. Set DROPBOX_ACCESS_TOKEN environment variable, or")
        print("2. Enter it when prompted")
        print()
        access_token = input("Enter Dropbox access token (or press Enter to skip): ").strip()

    if not access_token:
        print("No access token provided. Skipping tests.")
        print("\nTo get a Dropbox access token:")
        print("1. Go to https://www.dropbox.com/developers/apps")
        print("2. Create a new app or use an existing one")
        print("3. Generate an access token in the OAuth 2 section")
        return

    # Test connection
    if not test_dropbox_connection_sync(access_token):
        print("Connection test failed. Please check your access token.")
        return

    print()

    # Test file upload
    test_file = Path(__file__).parent / "test_data" / "test_file.txt"
    if test_file.exists():
        asyncio.run(test_file_upload(access_token, str(test_file)))
    else:
        print(f"Test file not found: {test_file}")
        print("Creating test file...")
        test_file.parent.mkdir(exist_ok=True)
        test_file.write_text("This is a test file for Dropbox upload functionality.")
        asyncio.run(test_file_upload(access_token, str(test_file)))

    print("\nTest completed!")


if __name__ == "__main__":
    main()
