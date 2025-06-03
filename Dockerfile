FROM python:3.11-slim

# Install ping utility and other required packages
RUN apt-get update && \
    apt-get install -y iputils-ping zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

COPY app /app
RUN python -m pip install /app --extra-index-url https://www.piwheels.org/simple && \
    python -m pip install fastapi uvicorn requests dropbox

EXPOSE 8000/tcp

# application version.  This should match the register_service file's version
LABEL version="0.0.1"

# Permissions for the container
# "Binds" section maps the host PC directories to the application directories
LABEL permissions='\
{\
  "ExposedPorts": {\
    "8000/tcp": {}\
  },\
  "HostConfig": {\
    "Binds":[\
      "/usr/blueos/extensions/cloud-upload/downloads:/app/downloads",\
      "/usr/blueos/extensions/cloud-upload/settings:/app/settings",\
      "/usr/blueos/extensions/cloud-upload/logs:/app/logs",\
      "/usr/blueos/ardupilot_logs:/ardupilot_logs",\
      "/usr/blueos/extensions:/extensions",\
      "/usr/blueos/userdata:/userdata"\
    ],\
    "ExtraHosts": ["host.docker.internal:host-gateway"],\
    "PortBindings": {\
      "8000/tcp": [\
        {\
          "HostPort": ""\
        }\
      ]\
    }\
  }\
}'

LABEL authors='[\
    {\
        "name": "Randy Mackay",\
        "email": "rmackay9@yahoo.com"\
    }\
]'

LABEL company='{\
    "about": "ArduPilot",\
    "name": "ArduPilot",\
    "email": "rmackay9@yahoo.com"\
}'

LABEL readme='https://github.com/rmackay9/blueos-camera-download/blob/main/README.md'
LABEL type="device-integration"
LABEL tags='[\
  "data-collection"\
]'
LABEL links='{\
        "source": "https://github.com/rmackay9/blueos-camera-download"\
    }'
LABEL requirements="core >= 1.1"

ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0"]
