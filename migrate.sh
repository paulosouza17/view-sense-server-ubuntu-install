#!/bin/bash

# ViewSense Migration & Update Script
# Usage: ./migrate.sh

echo "------------------------------------------------"
echo "Starting Migration & Update from GitHub..."
echo "------------------------------------------------"

# 1. Update Code
echo "[1/4] Pulling latest changes from GitHub..."
git fetch origin
git reset --hard origin/main

# 2. Update Dependencies
echo "[2/4] Updating Python dependencies..."
if [ -d "venv" ]; then
    ./venv/bin/pip install -r requirements.txt
else
    echo "Virtual environment not found! Run install.sh first."
    exit 1
fi

# 3. Recreate Systemd Service (to apply path/user fixes)
echo "[3/4] Recreating system service configuration..."
APP_DIR=$(pwd)
CURRENT_USER=$(whoami)
SERVICE_FILE="/etc/systemd/system/viewsense-yolo.service"

sudo bash -c "cat > $SERVICE_FILE <<EOF
[Unit]
Description=ViewSense YOLO Server
After=network.target

[Service]
User=$CURRENT_USER
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8080
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF"

echo "Service file updated at $SERVICE_FILE"

# 4. Restart Service
echo "[4/4] Restarting service..."
sudo systemctl daemon-reload
sudo systemctl unmask viewsense-yolo
sudo systemctl enable viewsense-yolo
sudo systemctl restart viewsense-yolo

echo "------------------------------------------------"
echo "Migration Complete!"
echo "Service Status:"
sudo systemctl status viewsense-yolo --no-pager
echo "------------------------------------------------"
