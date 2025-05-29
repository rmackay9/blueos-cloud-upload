# Cloud Upload

BlueOS Extension to upload logs, images and videos to a cloud provider (e.g. Google Cloud)

## Usage

Users should select the Cloud provider they are using and the directories they would like to upload to the cloud

Push the "Ping" button to confirm the cloud provided can be reached

Push the "Upload" button to upload all files

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
      "/usr/blueos/extensions/YOURDOCKERHUBUSER.YOURDOCKERHUBREPO/downloads:/app/downloads",
      "/usr/blueos/extensions/YOURDOCKERHUBUSER.YOURDOCKERHUBREPO/settings:/app/settings",
      "/usr/blueos/extensions/YOURDOCKERHUBUSER.YOURDOCKERHUBREPO/logs:/app/logs"
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
