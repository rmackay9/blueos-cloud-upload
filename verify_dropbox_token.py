#!/usr/bin/env python3
"""
Simple script to verify Dropbox access token
"""

import sys

try:
    import dropbox
    from dropbox.exceptions import AuthError, ApiError
except ImportError:
    print("Error: dropbox library not installed")
    print("Please install it with: pip install dropbox")
    sys.exit(1)

def verify_token(access_token):
    """Verify that a Dropbox access token is valid"""
    print("Testing Dropbox access token...")
    print(f"Token: {access_token[:10]}...{access_token[-10:] if len(access_token) > 20 else access_token}")
    print()

    try:
        # Create Dropbox client
        dbx = dropbox.Dropbox(access_token)

        # Try to get account info
        account_info = dbx.users_get_current_account()

        print("✓ SUCCESS: Token is valid!")
        print(f"Account: {account_info.name.display_name}")
        print(f"Email: {account_info.email}")
        print(f"Account ID: {account_info.account_id}")

        # Try to list root folder to test permissions
        try:
            result = dbx.files_list_folder("")
            print(f"✓ Permissions OK: Can access files (found {len(result.entries)} items in root)")
        except Exception as e:
            print(f"⚠ Warning: Limited permissions - {str(e)}")

        return True

    except AuthError as e:
        print("✗ FAILED: Invalid access token")
        print("Error:", str(e))
        print()
        print("Troubleshooting:")
        print("1. Make sure you copied the complete access token")
        print("2. Check that the token hasn't expired")
        print("3. Verify the app permissions are set correctly")
        return False

    except Exception as e:
        print("✗ ERROR:", str(e))
        return False

def main():
    print("Dropbox Access Token Verification Tool")
    print("=" * 40)

    # Get token from user input
    token = input("Enter your Dropbox access token: ").strip()

    if not token:
        print("No token provided.")
        return

    print()
    verify_token(token)

    print()
    print("If the token is valid, use it in the Cloud Upload extension:")
    print("- Username/Email: (leave blank or enter your email)")
    print("- Password/API Key: (paste the access token)")

if __name__ == "__main__":
    main()
