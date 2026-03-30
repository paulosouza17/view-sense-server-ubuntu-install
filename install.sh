#!/bin/bash

# ViewSense PM2 Edge Worker - Automated Installer for Ubuntu/Debian
# Usage: sudo ./install.sh

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting ViewSense AI Edge Worker Installation...${NC}"

if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}Please run as root (sudo ./install.sh)${NC}"
  exit 1
fi

USER_NAME=${SUDO_USER:-$USER}
HOME_DIR=$(getent passwd "$USER_NAME" | cut -d: -f6)
WORKER_DIR="$HOME_DIR/viewsense-ai-worker"

echo -e "${GREEN}[1/5] Installing System Dependencies...${NC}"
apt-get update
apt-get install -y curl git build-essential python3 python3-venv python3-pip python3-dev libgl1 libglib2.0-0 ffmpeg

echo -e "${GREEN}[2/5] Setting up Node.js & PM2...${NC}"
if ! command -v pm2 &> /dev/null; then
  echo "Installing Node.js and PM2..."
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
  npm install -g pm2
else
  echo "PM2 is already installed."
fi

echo -e "${GREEN}[3/5] Downloading Edge Worker Source Code...${NC}"
if [ -d "$WORKER_DIR" ]; then
    echo -e "${RED}Directory $WORKER_DIR already exists. Backing up...${NC}"
    mv "$WORKER_DIR" "${WORKER_DIR}_backup_$(date +%s)"
fi

# Use sparse-checkout to grab ONLY the edge-worker-mac folder from the monorepo
sudo -u $USER_NAME mkdir -p "$WORKER_DIR"
cd "$WORKER_DIR"
sudo -u $USER_NAME git init
sudo -u $USER_NAME git remote add origin https://github.com/paulosouza17/vision-audit-hub.git
sudo -u $USER_NAME git config core.sparseCheckout true
sudo -u $USER_NAME bash -c 'echo "edge-worker-mac/*" >> .git/info/sparse-checkout'
echo "Pulling worker code from Github..."
sudo -u $USER_NAME git pull origin main
# Move contents up one level and clean git
mv edge-worker-mac/* .
rm -rf edge-worker-mac .git

echo -e "${GREEN}[4/5] Setting up Python Virtual Environment...${NC}"
if [ ! -d "venv" ]; then
    sudo -u $USER_NAME python3 -m venv venv
    echo "Virtual environment created."
fi

echo -e "${GREEN}[5/5] Installing Python Requirements & PM2 Ecosystem...${NC}"
sudo -H -u $USER_NAME bash -c "source venv/bin/activate && pip install --upgrade pip"
if [ -f "requirements.txt" ]; then
    sudo -H -u $USER_NAME bash -c "source venv/bin/activate && pip install -r requirements.txt"
else
    echo -e "${RED}Fatal Error: requirements.txt not found after download!${NC}"
    exit 1
fi

if [ ! -f "ecosystem.config.js" ]; then
    cat > ecosystem.config.js <<EOF
module.exports = {
  apps: [
    {
      name: "viewsense-rtmp",
      script: "venv/bin/python",
      args: "rtmp_engine.py",
      interpreter: "none",
      autorestart: true,
      max_restarts: 50,
      watch: false,
    },
    {
      name: "viewsense-health",
      script: "venv/bin/python",
      args: "server_health.py",
      interpreter: "none",
      autorestart: true,
      watch: false,
    },
    {
      name: "viewsense-ai-worker",
      script: "venv/bin/python",
      args: "detector.py",
      interpreter: "none",
      autorestart: true,
      watch: false,
    }
  ]
};
EOF
    chown $USER_NAME:$USER_NAME ecosystem.config.js
fi

# Run PM2 Startup directly
env PATH=$PATH:/usr/bin /usr/lib/node_modules/pm2/bin/pm2 startup systemd -u $USER_NAME --hp $HOME_DIR

echo -e "${GREEN}====================================================${NC}"
echo -e "${GREEN}ViewSense Edge Worker Core Successfully Installed!${NC}"
echo -e "Next steps:"
echo -e "1. CD into the worker directory:\n   cd $WORKER_DIR"
echo -e "2. Download your camera config.yaml using the specific cURL command from your Panel."
echo -e "3. Start the AI processes:\n   pm2 start ecosystem.config.js && pm2 save"
echo -e "${GREEN}====================================================${NC}"
