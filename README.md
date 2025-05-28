# Cloud Upload

BlueOS Extension to upload logs, images and videos to a cloud provider (e.g. Google Cloud, AWS S3, Azure Blob Storage, Dropbox)

## Supported Cloud Providers

### Dropbox (Fully Implemented)
- **Setup**: Create a Dropbox app at https://www.dropbox.com/developers/apps
- **Authentication**: Use the generated access token as the password/API key
- **Username**: Email address (optional, for display purposes only)

### Other Providers (Placeholder Implementation)
- Google Cloud Storage
- Amazon S3
- Azure Blob Storage

## Usage

1. Select your cloud provider from the dropdown
2. Enter your credentials:
   - **For Dropbox**: Enter your access token in the password field
   - **For other providers**: Enter appropriate credentials (placeholder)
3. Add the directories you want to upload
4. Click "Ping Provider" to test your connection
5. Click "Scan Directories" to see how many files will be uploaded
6. Click "Upload All Files" to start the upload process

## Getting Dropbox Access Token

1. Go to https://www.dropbox.com/developers/apps
2. Click "Create app"
3. Choose "Scoped access" and "Full Dropbox" access
4. Give your app a name
5. Go to the "Permissions" tab and enable:
   - `files.metadata.write`
   - `files.content.write`
   - `files.content.read`
6. Go to the "Settings" tab
7. Under "OAuth 2", click "Generate access token"
8. Copy the generated token and use it as the password in the Cloud Upload interface

## Developer Information

To build and publish for Ubuntu, RPI3, RPI4, RPI5

- Open docker desktop (required only on Windows WSL2 machines)
- docker buildx build --platform linux/amd64,linux/arm/v7,linux/arm64/v8 . -t YOURDOCKERHUBUSER/YOURDOCKERHUBREPO:latest --output type=registry
- login to https://hub.docker.com/repositories/ and confirm the image has appeared

To manually install the extension in BlueOS

- Start BlueOS on RPI, open Chrome browser and connect to BlueOS (e.g. via WifiAP use http://blueos-hotspot.local/, if on same network use http://blueos-avahi.local/)
- Open BlueOS Extensions tab, select Installed
  - Push "+" button on the bottom right
  - Under "Create Extension" fill in these fields
    - Extension Identifier: YOURDOCKERHUBUSER.YOURDOCKERHUBREPO
    - Extension Name: Cloud Upload
    - Docker image: YOURDOCKERHUBUSER/YOURDOCKERHUBREPO
    - Dockertag: latest
    - Settings: add the lines below after replacing the capitalised values with your DockerHub username and repository name

```
{
  "ExposedPorts": {
    "8000/tcp": {}
  },
  "HostConfig": {
    "Binds":[
      "/usr/blueos/extensions/cloud-upload/downloads:/app/downloads",
      "/usr/blueos/extensions/cloud-upload/settings:/app/settings",
      "/usr/blueos/extensions/cloud-upload/logs:/app/logs",
      "/usr/blueos/ardupilot_logs:/ardupilot_logs",
      "/usr/blueos/extensions:/extensions",
      "/usr/blueos/userdata:/userdata"
      ],
    "PortBindings": {
      "8000/tcp": [
        {
          "HostPort": ""
        }
      ]
    }
  }
}
```

  - "Cloud Upload" should appear in list of installed extensions and "Status" should appear as "Up xx seconds"

To test on an Ubuntu PC

- Ensure the PC and camera is on the same ethernet subnet
- Open docker desktop (required only on Windows WSL2 machines)
- docker build -t YOURDOCKERHUBUSER/blueos-cloud-upload:latest .
- docker run -p 8000:8000 YOURDOCKERHUBUSER/blueos-cloud-upload:latest
- On docker desktop, Containers, a new image should appear with "Port(s)" field, "8000:8000".  Click to open a browser
- Within the web browser the Cloud Upload page should appear, set the "Cloud Provider", "username" and "password" fields
- Select the directories to upload
- Press "Ping" to check the connection
- Press "Upload" to upload all files
