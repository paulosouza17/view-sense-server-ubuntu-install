#!/bin/bash

# ViewSense YOLO Server - Auto Installer for Ubuntu
# Usage: sudo ./install.sh

set -e

GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}Starting ViewSense YOLO Server Installation...${NC}"

# Check for root/sudo
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (sudo ./install.sh)"
  exit 1
fi

APP_DIR=$(pwd)
USER_NAME=${SUDO_USER:-$USER}

echo -e "${GREEN}Updating Package Lists...${NC}"
apt update

echo -e "${GREEN}Installing System Dependencies...${NC}"
# Ubuntu 24.04 (Noble) compatibility fixes:
# - python3.11 might not be available, python3 (3.12) is default
# - libgl1-mesa-glx is deprecated, use libgl1
apt install -y python3 python3-venv python3-pip python3-dev build-essential libgl1 git

# Create Virtual Environment
echo -e "${GREEN}Setting up Python Virtual Environment...${NC}"
if [ ! -d "venv" ]; then
    # Use generic python3 command which points to the system's latest (3.12 on Noble)
    python3 -m venv venv
    chown -R $USER_NAME:$USER_NAME venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate and Install Requirements
echo -e "${GREEN}Installing Python Requirements...${NC}"
source venv/bin/activate
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    # Install dependencies
    pip install -r requirements.txt
else
    echo "requirements.txt not found!"
    exit 1
fi

# Download YOLO model to cache it
echo -e "${GREEN}Downloading default YOLO model...${NC}"
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Create Systemd Service
echo -e "${GREEN}Creating Systemd Service...${NC}"
SERVICE_FILE="/etc/systemd/system/viewsense-yolo.service"

cat > $SERVICE_FILE <<EOF
[Unit]
Description=ViewSense YOLO Detection Server
After=network.target

[Service]
User=$USER_NAME
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8080
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

echo "Service file created at $SERVICE_FILE"

# Enable and Start Service
systemctl daemon-reload
systemctl enable viewsense-yolo
# We don't start it immediately to allow user to config
# systemctl start viewsense-yolo

echo -e "${GREEN}Installation Complete!${NC}"
echo "------------------------------------------------"
echo "1. Edit config.yaml with your camera settings."
echo "2. Start the service: sudo systemctl start viewsense-yolo"
echo "3. Check logs: sudo journalctl -u viewsense-yolo -f"
echo "------------------------------------------------"
