#!/bin/bash

# ViewSense Migration & Update Script
# Usage: ./migrate.sh

echo "------------------------------------------------"
echo "Starting Migration & Update from GitHub..."
echo "------------------------------------------------"

# 1. Protect Configuration
echo "[1/5] Backing up local configuration..."
if [ -f "config.yaml" ]; then
    cp config.yaml config.yaml.migration_backup
    echo "✅ Configuration backed up to config.yaml.migration_backup"
fi

# 2. Update Code
echo "[2/5] Pulling latest changes from GitHub..."
git fetch origin
git reset --hard origin/main

# 3. Restore Configuration
echo "[3/5] Restoring local configuration..."
if [ -f "config.yaml.migration_backup" ]; then
    mv config.yaml.migration_backup config.yaml
    echo "✅ Configuration restored."
else
    echo "⚠️ No previous configuration found. Using repository default."
fi

# 4. Update Dependencies
echo "[4/5] Updating Python dependencies..."
if [ -d "venv" ]; then
    ./venv/bin/pip install -r requirements.txt
else
    echo "Virtual environment not found! Run install.sh first."
    exit 1
fi

# 5. Recreate Systemd Service (to apply path/user fixes)
echo "[5/5] Refreshing system service..."
APP_DIR=$(pwd)
CURRENT_USER=$(whoami)
SERVICE_FILE="/etc/systemd/system/viewsense-yolo.service"

# Only recreate if purely necessary, but good for ensuring consistency
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

sudo systemctl daemon-reload
sudo systemctl unmask viewsense-yolo
sudo systemctl enable viewsense-yolo
sudo systemctl restart viewsense-yolo

echo "------------------------------------------------"
echo "Migration Complete!"
echo "Service Status:"
sudo systemctl status viewsense-yolo --no-pager
echo "------------------------------------------------"
