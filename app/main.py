#!/usr/bin/env python3

# Cloud Upload Python backend
# Implements these features required by the index.html frontend:
# - Save setting including cloud service provider, username, password and list of directories
# - Scan directories to count files and calculate total size
# - Ping provider (placeholder)
# - Upload files to the cloud (placeholder)

import logging.handlers
import subprocess
import asyncio
import sys
import zipfile
import io
import os
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, Any, List
from pydantic import BaseModel

# Import cloud provider libraries
try:
    import dropbox
    from dropbox.exceptions import AuthError, ApiError
    DROPBOX_AVAILABLE = True
except ImportError:
    dropbox = None
    DROPBOX_AVAILABLE = False
import dropbox
from dropbox.exceptions import AuthError, ApiError

# Import the settings module
from app import settings

# Define the downloads directory path
DOWNLOADS_DIR = Path("/app/downloads")

# Configure console logging
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Create logger
logger = logging.getLogger("cloud_upload")
logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

app = FastAPI()

# Pydantic models for request bodies
class CloudSettings(BaseModel):
    provider: str
    username: str
    password: str
    directories: List[str]

class DirectoriesScan(BaseModel):
    directories: List[str]

class CloudCredentials(BaseModel):
    provider: str
    username: str
    password: str


# Ensure downloads directory exists
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Downloads directory set up at {DOWNLOADS_DIR}")


# Helper function to check if camera is reachable
def is_camera_reachable(ip: str) -> tuple[bool, str]:
    """Check if a camera is reachable at the specified IP address

    Args:
        ip: IP address to ping

    Returns:
        Tuple of (is_reachable, message)
    """
    logger.debug(f"Pinging camera at {ip}")
    try:
        # Run ping command with 2 second timeout, 3 packets
        result = subprocess.run(
            ["ping", "-c", "3", "-W", "2", ip],
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode == 0:
            logger.info(f"Camera at {ip} is reachable")
            return True, f"Camera at {ip} is reachable"
        else:
            logger.warning(f"Camera at {ip} is not reachable")
            return False, f"Camera at {ip} is not reachable"
    except Exception as e:
        logger.error(f"Error pinging camera: {str(e)}")
        return False, f"Error pinging camera: {str(e)}"


# save camera settings using the settings module
@app.post("/camera/save-settings")
async def save_camera_settings(type: str, ip: str) -> Dict[str, Any]:
    """Save camera settings to persistent storage"""
    logger.info(f"Saving camera settings: type={type}, ip={ip}")
    success = settings.update_camera_ip(type, ip)

    if success:
        return {"success": True, "message": f"Camera settings saved for {type}"}
    else:
        return {"success": False, "message": "Failed to save camera settings"}


# get camera settings using the settings module
@app.post("/camera/get-settings")
async def get_camera_settings() -> Dict[str, Any]:
    """Get saved camera settings"""
    logger.info("Getting camera settings")

    try:
        # Get the last used camera settings
        last_used = settings.get_last_used()

        # Get IP addresses for both camera types
        siyi_ip = settings.get_camera_ip('siyi')
        xfrobot_ip = settings.get_camera_ip('xfrobot')

        return {
            "success": True,
            "last_used": last_used,
            "cameras": {
                "siyi": {"ip": siyi_ip},
                "xfrobot": {"ip": xfrobot_ip}
            }
        }
    except Exception as e:
        logger.exception(f"Error getting camera settings: {str(e)}")
        return {"success": False, "message": f"Error: {str(e)}"}


# ping camera at the specified IP address
@app.post("/camera/ping")
async def ping_camera(ip: str) -> Dict[str, Any]:
    """Ping a camera at specified IP address"""
    logger.info(f"Ping request received for camera at {ip}")
    is_reachable, message = is_camera_reachable(ip)
    return {"success": is_reachable, "message": message}


# download images and video files from camera
@app.post("/camera/download")
async def download_images(type: str, ip: str):
    """Download images from camera based on type and IP address"""
    return StreamingResponse(
        download_generator(type, ip),
        media_type="text/event-stream"
    )


# download image generator function for streaming progress to the frontend
async def download_generator(type: str, ip: str):
    """Generator function for streaming download progress"""
    logger.info(f"Download request received for {type} camera at {ip}")

    # Save the camera settings when a download is requested
    settings.update_camera_ip(type, ip)

    # Correct SSE format with "data:" prefix and double newline
    yield f"data: Connecting to camera at {ip}\n\n"

    try:
        # check if camera is reachable
        is_reachable, message = is_camera_reachable(ip)
        if not is_reachable:
            logger.warning(f"Camera at {ip} is not reachable, aborting download")
            yield f"data: Error: {message}. Please check the connection and try again\n\n"
            return

        # Set up download directory
        DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

        # Get script path
        script_dir = Path(__file__).parent
        script_path = script_dir / f"{type}-download.py"

        if not script_path.exists():
            logger.error(f"Download script {script_path} not found")
            yield f"data: Error: download script for {type} camera not found\n\n"
            return

        # Build command
        cmd = f"python3 {script_path} --ipaddr {ip} --dest {DOWNLOADS_DIR}"

        # display download started message
        yield f"data: Started download from {type} camera at {ip}\n\n"
        yield f"data: This may take a while depending on the number of files...\n\n"
        yield f"data: Files will be saved to: {DOWNLOADS_DIR}\n\n"

        # Execute command
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Set up heartbeat task
        heartbeat_interval = 5  # seconds
        last_message_time = asyncio.get_event_loop().time()

        # Process stdout in real-time
        while True:
            try:
                # set a timeout to send heartbeats if case no lines received
                line = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=heartbeat_interval
                )

                # process incoming lines
                if line:
                    line_text = line.decode('utf-8').rstrip()
                    logger.debug(f"Script output: {line_text}")
                    yield f"data: {line_text}\n\n"
                    last_message_time = asyncio.get_event_loop().time()
                else:
                    # end of output
                    break

            except asyncio.TimeoutError:
                # send heartbeat
                current_time = asyncio.get_event_loop().time()
                if current_time - last_message_time >= heartbeat_interval:
                    logger.debug("Sending heartbeat to keep connection alive")
                    # Heartbeat in correct SSE format
                    yield ":\n\n"
                    last_message_time = current_time

                # check if process is still running
                if process.returncode is not None:
                    break

        # Wait for process to complete
        await process.wait()

        # Check result
        if process.returncode == 0:
            yield "data: Download completed successfully!\n\n"
        else:
            error = await process.stderr.read()
            error_text = error.decode('utf-8')
            yield f"data: Download failed with Error: {error_text}\n\n"

    except Exception as e:
        logger.exception(f"Error in download process: {str(e)}")
        yield f"data: Error during download: {str(e)}\n\n"

    # Final heartbeat before closing
    yield ":\n\n"


# Count image and video files in the downloads directory
@app.post("/camera/count-files")
async def count_files() -> Dict[str, Any]:
    """Count the number of image and video files in the downloads directory"""
    logger.info("Counting files in downloads directory")

    try:
        if not DOWNLOADS_DIR.exists():
            return {
                "success": True,
                "images": 0,
                "videos": 0
            }

        # Count files by extension
        image_count = 0
        video_count = 0

        # Common image and video extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv']

        for file in DOWNLOADS_DIR.iterdir():
            if file.is_file():
                lower_name = file.name.lower()
                if any(lower_name.endswith(ext) for ext in image_extensions):
                    image_count += 1
                elif any(lower_name.endswith(ext) for ext in video_extensions):
                    video_count += 1

        return {
            "success": True,
            "images": image_count,
            "videos": video_count
        }
    except Exception as e:
        logger.exception(f"Error counting files: {str(e)}")
        return {
            "success": False,
            "message": f"Error: {str(e)}",
            "images": 0,
            "videos": 0
        }


# download ZIP file of all image and video files in the downloads directory
@app.post("/camera/download-zip")
async def download_zip(request_token: str = None):
    """Create a ZIP archive of all files in the downloads directory and serve it for download"""
    logger.info("Creating ZIP archive of all downloaded files")

    try:
        if not DOWNLOADS_DIR.exists() or not any(DOWNLOADS_DIR.iterdir()):
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": "No files available to download"}
            )

        # Create a ZIP file in memory
        zip_buffer = io.BytesIO()

        # Get current date for the filename
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"camera_files_{current_date}.zip"

        # Create the ZIP file with all files in the downloads directory
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
            for file_path in DOWNLOADS_DIR.iterdir():
                if file_path.is_file():
                    # Add file to the ZIP with just the filename (not the full path)
                    zip_file.write(file_path, arcname=file_path.name)

        # Reset buffer position to the beginning
        zip_buffer.seek(0)

        # Return the ZIP file as a downloadable response
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        logger.exception(f"Error creating ZIP archive: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error creating ZIP archive: {str(e)}"}
        )


# delete all files in the downloads directory
@app.delete("/camera/delete-files")
async def delete_files() -> Dict[str, Any]:
    """Delete all files from the downloads directory"""
    logger.info("Deleting files from downloads directory")

    try:
        if not DOWNLOADS_DIR.exists():
            DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
            return {
                "success": True,
                "message": "No files to delete",
                "deleted_count": 0
            }

        # Count files before deletion
        file_count = 0
        for item in DOWNLOADS_DIR.iterdir():
            if item.is_file():
                file_count += 1

        if file_count == 0:
            return {
                "success": True,
                "message": "No files to delete",
                "deleted_count": 0
            }

        # Delete all files (but keep the directory)
        for item in DOWNLOADS_DIR.iterdir():
            if item.is_file():
                item.unlink()

        return {
            "success": True,
            "message": f"Successfully deleted {file_count} files",
            "deleted_count": file_count
        }
    except Exception as e:
        logger.exception(f"Error deleting files: {str(e)}")
        return {
            "success": False,
            "message": f"Error: {str(e)}",
            "deleted_count": 0
        }


# Helper function to scan directories and calculate file counts and sizes
def scan_directories(directories: List[str]) -> Dict[str, Any]:
    """Scan directories to count files and calculate total size

    Args:
        directories: List of directory paths to scan

    Returns:
        Dict containing total_files, total_size, and breakdown by directory
    """
    total_files = 0
    total_size_bytes = 0
    directory_info = {}

    for directory in directories:
        dir_path = Path(directory)
        dir_files = 0
        dir_size = 0

        try:
            if dir_path.exists() and dir_path.is_dir():
                # Recursively scan all files in the directory
                for file_path in dir_path.rglob('*'):
                    if file_path.is_file():
                        try:
                            file_size = file_path.stat().st_size
                            dir_files += 1
                            dir_size += file_size
                        except OSError:
                            logger.warning(f"Could not access file: {file_path}")
                            continue
            else:
                logger.warning(f"Directory does not exist or is not accessible: {directory}")

        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")

        directory_info[directory] = {
            'files': dir_files,
            'size_bytes': dir_size,
            'size_mb': round(dir_size / (1024 * 1024), 2)
        }

        total_files += dir_files
        total_size_bytes += dir_size

    total_size_mb = round(total_size_bytes / (1024 * 1024), 2)

    return {
        'total_files': total_files,
        'total_size': f"{total_size_mb}",
        'total_size_bytes': total_size_bytes,
        'directories': directory_info        }


# ==================== CLOUD PROVIDER HELPER FUNCTIONS ====================

def test_dropbox_connection(access_token: str) -> tuple[bool, str]:
    """Test Dropbox connection with the provided access token

    Args:
        access_token: Dropbox access token

    Returns:
        Tuple of (success, message)
    """
    if not DROPBOX_AVAILABLE:
        return False, "Dropbox library not available"

    try:
        dbx = dropbox.Dropbox(access_token)
        # Try to get account info to test the connection
        account_info = dbx.users_get_current_account()
        return True, f"Connected to Dropbox account: {account_info.email}"
    except AuthError:
        return False, "Invalid Dropbox access token"
    except Exception as e:
        return False, f"Error connecting to Dropbox: {str(e)}"


async def upload_file_to_dropbox(dbx, local_file_path: Path, remote_path: str) -> tuple[bool, str]:
    """Upload a single file to Dropbox

    Args:
        dbx: Dropbox client instance
        local_file_path: Path to the local file
        remote_path: Destination path in Dropbox

    Returns:
        Tuple of (success, message)
    """
    try:
        file_size = local_file_path.stat().st_size

        # For files larger than 150MB, use upload session (chunked upload)
        if file_size > 150 * 1024 * 1024:
            return await upload_large_file_to_dropbox(dbx, local_file_path, remote_path)

        # For smaller files, use regular upload
        with open(local_file_path, 'rb') as f:
            file_data = f.read()

        dbx.files_upload(
            file_data,
            remote_path,
            mode=dropbox.files.WriteMode.overwrite
        )

        return True, f"Uploaded {local_file_path.name} ({file_size} bytes)"

    except Exception as e:
        return False, f"Failed to upload {local_file_path.name}: {str(e)}"


async def upload_large_file_to_dropbox(dbx, local_file_path: Path, remote_path: str) -> tuple[bool, str]:
    """Upload a large file to Dropbox using chunked upload

    Args:
        dbx: Dropbox client instance
        local_file_path: Path to the local file
        remote_path: Destination path in Dropbox

    Returns:
        Tuple of (success, message)
    """
    try:
        file_size = local_file_path.stat().st_size
        chunk_size = 4 * 1024 * 1024  # 4MB chunks

        with open(local_file_path, 'rb') as f:
            # Start upload session
            upload_session_start_result = dbx.files_upload_session_start(
                f.read(chunk_size)
            )
            cursor = dropbox.files.UploadSessionCursor(
                session_id=upload_session_start_result.session_id,
                offset=f.tell(),
            )

            # Upload remaining chunks
            while f.tell() < file_size:
                if (file_size - f.tell()) <= chunk_size:
                    # Final chunk
                    commit = dropbox.files.CommitInfo(path=remote_path, mode=dropbox.files.WriteMode.overwrite)
                    dbx.files_upload_session_finish(f.read(chunk_size), cursor, commit)
                else:
                    # Regular chunk
                    dbx.files_upload_session_append_v2(f.read(chunk_size), cursor)
                    cursor.offset = f.tell()

        return True, f"Uploaded large file {local_file_path.name} ({file_size} bytes)"

    except Exception as e:
        return False, f"Failed to upload large file {local_file_path.name}: {str(e)}"


# ==================== CLOUD UPLOAD ENDPOINTS ====================

@app.post("/cloud/save-settings")
async def save_cloud_settings(cloud_settings: CloudSettings) -> Dict[str, Any]:
    """Save cloud provider settings to persistent storage"""
    logger.info(f"Saving cloud settings for provider: {cloud_settings.provider}")

    try:
        success = settings.update_cloud_settings(
            cloud_settings.provider,
            cloud_settings.username,
            cloud_settings.password,
            cloud_settings.directories
        )

        if success:
            return {
                "success": True,
                "message": f"Cloud settings saved for {cloud_settings.provider}"
            }
        else:
            return {
                "success": False,
                "message": "Failed to save cloud settings"
            }
    except Exception as e:
        logger.exception(f"Error saving cloud settings: {str(e)}")
        return {
            "success": False,
            "message": f"Error saving settings: {str(e)}"
        }


@app.post("/cloud/get-settings")
async def get_cloud_settings() -> Dict[str, Any]:
    """Get saved cloud provider settings"""
    logger.info("Getting cloud settings")

    try:
        cloud_settings = settings.get_cloud_settings()
        return {
            "success": True,
            "settings": cloud_settings
        }
    except Exception as e:
        logger.exception(f"Error getting cloud settings: {str(e)}")
        return {
            "success": False,
            "message": f"Error: {str(e)}"
        }


@app.post("/cloud/scan-directories")
async def scan_cloud_directories(directories_request: DirectoriesScan) -> Dict[str, Any]:
    """Scan directories to count files and calculate total size"""
    logger.info(f"Scanning directories: {directories_request.directories}")

    try:
        if not directories_request.directories:
            return {
                "success": False,
                "message": "No directories provided"
            }

        scan_result = scan_directories(directories_request.directories)

        return {
            "success": True,
            "total_files": scan_result["total_files"],
            "total_size": scan_result["total_size"],
            "total_size_bytes": scan_result["total_size_bytes"],
            "directories": scan_result["directories"]
        }

    except Exception as e:
        logger.exception(f"Error scanning directories: {str(e)}")
        return {
            "success": False,
            "message": f"Error scanning directories: {str(e)}"
        }


@app.post("/cloud/ping")
async def ping_cloud_provider(credentials: CloudCredentials) -> Dict[str, Any]:
    """Ping cloud provider to check connectivity"""
    logger.info(f"Ping request for {credentials.provider} provider")

    try:
        if not credentials.username or not credentials.password:
            return {
                "success": False,
                "message": "Username and password/API key are required"
            }

        if credentials.provider == "dropbox":
            # For Dropbox, use the access token (password field) to test connection
            success, message = test_dropbox_connection(credentials.password)
            return {
                "success": success,
                "message": message
            }
        else:
            # Placeholder implementation for other providers
            provider_urls = {
                "google": "storage.googleapis.com",
                "aws": "s3.amazonaws.com",
                "azure": "blob.core.windows.net"
            }

            url = provider_urls.get(credentials.provider, "unknown")

            # For now, just return success with a message for non-Dropbox providers
            return {
                "success": True,
                "message": f"Ping to {credentials.provider} ({url}) successful (simulated)"
            }

    except Exception as e:
        logger.exception(f"Error pinging cloud provider: {str(e)}")
        return {
            "success": False,
            "message": f"Error pinging provider: {str(e)}"
        }


@app.post("/cloud/upload")
async def upload_to_cloud(cloud_settings: CloudSettings):
    """Upload files to cloud provider (placeholder for streaming response)"""
    logger.info(f"Upload request for {cloud_settings.provider} provider")

    # Return streaming response for real-time progress updates
    return StreamingResponse(
        upload_generator(cloud_settings),
        media_type="text/event-stream"
    )


async def upload_generator(cloud_settings: CloudSettings):
    """Generator function for streaming upload progress"""
    try:
        # Validate settings
        if not cloud_settings.directories:
            yield f"data: Error: No directories specified for upload\n\n"
            return

        if not cloud_settings.username or not cloud_settings.password:
            yield f"data: Error: Username and password/API key are required\n\n"
            return

        # Save settings
        settings.update_cloud_settings(
            cloud_settings.provider,
            cloud_settings.username,
            cloud_settings.password,
            cloud_settings.directories
        )

        yield f"data: Starting upload to {cloud_settings.provider}...\n\n"

        # Scan directories first
        yield f"data: Scanning directories for files...\n\n"
        scan_result = scan_directories(cloud_settings.directories)

        total_files = scan_result["total_files"]
        total_size = scan_result["total_size"]

        if total_files == 0:
            yield f"data: No files found in specified directories\n\n"
            return

        yield f"data: Found {total_files} files ({total_size} MB) to upload\n\n"

        # Handle Dropbox uploads
        if cloud_settings.provider == "dropbox":
            async for message in handle_dropbox_upload(cloud_settings, scan_result):
                yield message
        else:
            # Placeholder for other providers
            async for message in handle_other_provider_upload(cloud_settings, scan_result):
                yield message

    except Exception as e:
        logger.exception(f"Error during upload: {str(e)}")
        yield f"data: Error during upload: {str(e)}\n\n"


async def handle_dropbox_upload(cloud_settings: CloudSettings, scan_result: Dict[str, Any]):
    """Handle Dropbox-specific upload logic"""
    if not DROPBOX_AVAILABLE:
        yield f"data: Error: Dropbox library not available\n\n"
        return

    try:
        # Initialize Dropbox client
        dbx = dropbox.Dropbox(cloud_settings.password)  # password field contains access token

        # Test connection
        try:
            account_info = dbx.users_get_current_account()
            yield f"data: Connected to Dropbox account: {account_info.email}\n\n"
        except AuthError:
            yield f"data: Error: Invalid Dropbox access token\n\n"
            return
        except Exception as e:
            yield f"data: Error connecting to Dropbox: {str(e)}\n\n"
            return

        # Upload files from each directory
        uploaded_count = 0
        failed_count = 0
        total_files = scan_result["total_files"]

        for directory in cloud_settings.directories:
            dir_path = Path(directory)
            dir_info = scan_result["directories"].get(directory, {})
            dir_files = dir_info.get("files", 0)

            if dir_files == 0:
                yield f"data: Skipping empty directory: {directory}\n\n"
                continue

            yield f"data: Processing directory: {directory} ({dir_files} files)\n\n"

            if not dir_path.exists() or not dir_path.is_dir():
                yield f"data: Warning: Directory not found or not accessible: {directory}\n\n"
                continue

            # Process all files in the directory recursively
            for file_path in dir_path.rglob('*'):
                if file_path.is_file():
                    # Create remote path (preserve directory structure)
                    relative_path = file_path.relative_to(dir_path)
                    remote_path = f"/{dir_path.name}/{relative_path}".replace("\\", "/")

                    try:
                        success, message = await upload_file_to_dropbox(dbx, file_path, remote_path)
                        if success:
                            uploaded_count += 1
                            yield f"data: {message}\n\n"
                        else:
                            failed_count += 1
                            yield f"data: {message}\n\n"
                    except Exception as e:
                        failed_count += 1
                        yield f"data: ✗ Error uploading {file_path.name}: {str(e)}\n\n"

                    # Brief pause to prevent overwhelming the API
                    await asyncio.sleep(0.1)

        # Final summary
        yield f"data: Successfully uploaded: {uploaded_count} files\n\n"
        if failed_count > 0:
            yield f"data: Failed uploads: {failed_count} files\n\n"

    except Exception as e:
        logger.exception(f"Error during Dropbox upload: {str(e)}")
        yield f"data: Error during Dropbox upload: {str(e)}\n\n"


async def handle_other_provider_upload(cloud_settings: CloudSettings, scan_result: Dict[str, Any]):
    """Handle uploads for providers other than Dropbox (placeholder)"""
    total_files = scan_result["total_files"]
    total_size = scan_result["total_size"]

    # Placeholder implementation for other providers
    for i, directory in enumerate(cloud_settings.directories, 1):
        dir_info = scan_result["directories"].get(directory, {})
        dir_files = dir_info.get("files", 0)
        dir_size = dir_info.get("size_mb", 0)

        yield f"data: Processing directory {i}/{len(cloud_settings.directories)}: {directory}\n\n"
        yield f"data: Directory contains {dir_files} files ({dir_size} MB)\n\n"

        # Simulate upload progress
        await asyncio.sleep(1)

    yield f"data: Upload completed successfully! (simulated)\n\n"
    yield f"data: Total files uploaded: {total_files}\n\n"
    yield f"data: Total size uploaded: {total_size} MB\n\n"


# Mount static files AFTER defining API routes
# Use absolute path to handle Docker container environment
static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

# Set up logging for the app
log_dir = Path('/app/logs')
log_dir.mkdir(parents=True, exist_ok=True)
fh = logging.handlers.RotatingFileHandler(log_dir / 'lumber.log', maxBytes=2**16, backupCount=1)
logger.addHandler(fh)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
