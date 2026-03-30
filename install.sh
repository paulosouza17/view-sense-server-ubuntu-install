#!/bin/bash

# ViewSense PM2 Edge Worker - Automated Installer for Ubuntu/Debian
# Usage: sudo ./install.sh (run from inside the cloned folder)

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting ViewSense AI Edge Worker Installation...${NC}"

if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}Please run as root (sudo ./install.sh)${NC}"
  exit 1
fi

# The worker files are in this same directory
SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
USER_NAME=${SUDO_USER:-$USER}
HOME_DIR=$(getent passwd "$USER_NAME" | cut -d: -f6)
WORKER_DIR="$HOME_DIR/viewsense-ai-worker"

echo -e "${GREEN}[1/6] Installing System Dependencies...${NC}"
apt-get update -qq
apt-get install -y curl git build-essential python3 python3-venv python3-pip python3-dev libgl1 libglib2.0-0 ffmpeg

echo -e "${GREEN}[2/6] Setting up Node.js 20 & PM2...${NC}"
if ! command -v pm2 &> /dev/null; then
  echo "Installing Node.js 20..."
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
  npm install -g pm2
  echo "PM2 installed successfully."
else
  echo "PM2 already installed ($(pm2 --version))."
fi

echo -e "${GREEN}[3/6] Copying Edge Worker files to $WORKER_DIR...${NC}"
if [ -d "$WORKER_DIR" ]; then
  echo "Backing up existing install..."
  mv "$WORKER_DIR" "${WORKER_DIR}_backup_$(date +%s)"
fi
mkdir -p "$WORKER_DIR"
# Copy worker source files (exclude installer files and git)
rsync -a --exclude='.git' --exclude='install.sh' --exclude='install_mac.sh' --exclude='install_ubuntu.sh' --exclude='*.md' --exclude='Dockerfile' --exclude='docker-compose.yml' "$SOURCE_DIR/" "$WORKER_DIR/"
chown -R "$USER_NAME:$USER_NAME" "$WORKER_DIR"
echo "Files copied to $WORKER_DIR"

echo -e "${GREEN}[4/6] Setting up Python Virtual Environment...${NC}"
cd "$WORKER_DIR"
if [ ! -d "venv" ]; then
  sudo -u "$USER_NAME" python3 -m venv venv
  echo "Virtual environment created."
fi

echo -e "${GREEN}[5/6] Installing Python Requirements...${NC}"
sudo -H -u "$USER_NAME" bash -c "source '$WORKER_DIR/venv/bin/activate' && pip install --upgrade pip --quiet"
if [ -f "$WORKER_DIR/requirements.txt" ]; then
  echo "Installing packages (this may take a few minutes)..."
  sudo -H -u "$USER_NAME" bash -c "source '$WORKER_DIR/venv/bin/activate' && pip install -r '$WORKER_DIR/requirements.txt' --quiet"
  echo "Python packages installed."
else
  echo -e "${RED}requirements.txt not found!${NC}"
  exit 1
fi

echo -e "${GREEN}[6/6] Configuring PM2 Ecosystem...${NC}"
cd "$WORKER_DIR"

if [ ! -f "ecosystem.config.js" ]; then
  cat > ecosystem.config.js <<EOF
module.exports = {
  apps: [
    {
      name: "viewsense-ai-worker",
      script: "$WORKER_DIR/venv/bin/python",
      args: "$WORKER_DIR/main.py",
      cwd: "$WORKER_DIR",
      interpreter: "none",
      autorestart: true,
      max_restarts: 50,
      watch: false,
      env: { PYTHONUNBUFFERED: "1" }
    }
  ]
};
EOF
  chown "$USER_NAME:$USER_NAME" ecosystem.config.js
  echo "ecosystem.config.js created."
fi

# Configure PM2 to start on system boot
echo "Configuring PM2 startup..."
env PATH=$PATH:/usr/bin /usr/lib/node_modules/pm2/bin/pm2 startup systemd -u "$USER_NAME" --hp "$HOME_DIR" 2>/dev/null || true

echo ""
echo -e "${GREEN}====================================================${NC}"
echo -e "${GREEN}  ViewSense Edge Worker installed successfully! ✅  ${NC}"
echo -e "${GREEN}====================================================${NC}"
echo ""
echo "Worker directory: $WORKER_DIR"
echo ""
echo "NEXT STEPS:"
echo ""
echo "  1. Download your config.yaml using the command shown in the panel:"
echo "     curl -H \"x-api-key: YOUR_KEY\" \"YOUR_BOOTSTRAP_URL\" -o $WORKER_DIR/config.yaml"
echo ""
echo "  2. Start the AI processes:"
echo "     cd $WORKER_DIR && pm2 start ecosystem.config.js && pm2 save"
echo ""
echo "  3. View logs:"
echo "     pm2 logs"
echo ""
