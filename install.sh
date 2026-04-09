#!/bin/bash

# ViewSense PM2 Edge Worker - Automated Installer for Ubuntu/Debian
# Includes: AI Worker (Python/YOLO) + RTMP Ingest Server (Node.js)
# Usage: sudo ./install.sh (run from inside the cloned folder)

set -e

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting ViewSense AI Edge Worker Installation...${NC}"

if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}Please run as root (sudo ./install.sh)${NC}"
  exit 1
fi

SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
USER_NAME=${SUDO_USER:-$USER}
HOME_DIR=$(getent passwd "$USER_NAME" | cut -d: -f6)
WORKER_DIR="/root/viewsense-ai-worker"
VIEWSENSE_DIR="/opt/viewsense"

# ─── [1/7] System Dependencies ───────────────────────────────────────────────
echo -e "${GREEN}[1/7] Installing System Dependencies...${NC}"
apt-get update -qq
apt-get install -y curl git build-essential python3 python3-venv python3-pip \
  python3-dev libgl1 libglib2.0-0 ffmpeg rsync

# ─── [2/7] Node.js 20 & PM2 ──────────────────────────────────────────────────
echo -e "${GREEN}[2/7] Setting up Node.js 20 & PM2...${NC}"
if ! command -v node &>/dev/null || [[ "$(node -e 'process.stdout.write(process.version)')" < "v20" ]]; then
  echo "Installing Node.js 20..."
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi
if ! command -v pm2 &>/dev/null; then
  npm install -g pm2
  echo "PM2 installed."
else
  echo "PM2 already installed ($(pm2 --version))."
fi

# ─── [3/7] AI Worker Files ───────────────────────────────────────────────────
echo -e "${GREEN}[3/7] Copying AI Worker files to $WORKER_DIR...${NC}"
if [ -d "$WORKER_DIR" ]; then
  echo "Backing up existing install..."
  mv "$WORKER_DIR" "${WORKER_DIR}_backup_$(date +%s)"
fi
mkdir -p "$WORKER_DIR"
rsync -a \
  --exclude='.git' --exclude='install.sh' --exclude='*.md' \
  --exclude='Dockerfile' --exclude='docker-compose.yml' \
  --exclude='rtmp-ingest.cjs' \
  "$SOURCE_DIR/" "$WORKER_DIR/"
chown -R "$USER_NAME:$USER_NAME" "$WORKER_DIR"
echo "AI Worker files copied to $WORKER_DIR"

# ─── [4/7] Python Virtual Environment ────────────────────────────────────────
echo -e "${GREEN}[4/7] Setting up Python Virtual Environment...${NC}"
cd "$WORKER_DIR"
if [ ! -d "venv" ]; then
  sudo -u "$USER_NAME" python3 -m venv venv
  echo "Virtual environment created."
fi

# ─── [5/7] Python Requirements ───────────────────────────────────────────────
echo -e "${GREEN}[5/7] Installing Python Requirements...${NC}"
sudo -H -u "$USER_NAME" bash -c "source '$WORKER_DIR/venv/bin/activate' && pip install --upgrade pip --quiet"
if [ -f "$WORKER_DIR/requirements.txt" ]; then
  echo "Installing packages (this may take a few minutes)..."
  sudo -H -u "$USER_NAME" bash -c "source '$WORKER_DIR/venv/bin/activate' && pip install -r '$WORKER_DIR/requirements.txt' --quiet"
  echo "Python packages installed."
else
  echo -e "${RED}requirements.txt not found!${NC}"
  exit 1
fi

# ─── [6/7] RTMP Ingest Server ────────────────────────────────────────────────
echo -e "${GREEN}[6/7] Installing RTMP Ingest Server (node-media-server)...${NC}"
mkdir -p "$VIEWSENSE_DIR/scripts" "$VIEWSENSE_DIR/media/snapshots"

# Copy rtmp-ingest.cjs to /opt/viewsense/scripts/
if [ -f "$SOURCE_DIR/rtmp-ingest.cjs" ]; then
  cp "$SOURCE_DIR/rtmp-ingest.cjs" "$VIEWSENSE_DIR/scripts/rtmp-ingest.cjs"
else
  echo -e "${YELLOW}⚠ rtmp-ingest.cjs not found, downloading from GitHub...${NC}"
  curl -fsSL "https://raw.githubusercontent.com/paulosouza17/view-sense-server-ubuntu-install/main/rtmp-ingest.cjs" \
    -o "$VIEWSENSE_DIR/scripts/rtmp-ingest.cjs"
fi

# Install node-media-server locally inside /opt/viewsense/scripts/
(cd "$VIEWSENSE_DIR/scripts" && \
  npm init -y --silent > /dev/null 2>&1 && \
  npm install node-media-server --save --silent > /dev/null 2>&1)

# Initialize active_streams.json
echo "[]" > "$VIEWSENSE_DIR/active_streams.json"
chmod +x "$VIEWSENSE_DIR/scripts/rtmp-ingest.cjs"

# Open firewall ports if ufw is active
if command -v ufw &>/dev/null && ufw status | grep -q "Status: active"; then
  ufw allow 55935/tcp comment "ViewSense RTMP Ingest" 2>/dev/null || true
  ufw allow 8001/tcp  comment "ViewSense HLS"         2>/dev/null || true
  ufw allow 8002/tcp  comment "ViewSense Snapshots"   2>/dev/null || true
  echo "UFW: Portas 55935, 8001, 8002 liberadas."
fi

echo "RTMP server installed at $VIEWSENSE_DIR/scripts/"

# ─── [7/7] PM2 Ecosystem (AI Worker + RTMP) ──────────────────────────────────
echo -e "${GREEN}[7/7] Configuring PM2 Ecosystem (AI Worker + RTMP)...${NC}"
cd "$WORKER_DIR"

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
    },
    {
      name: "viewsense-rtmp",
      script: "$VIEWSENSE_DIR/scripts/rtmp-ingest.cjs",
      cwd: "$VIEWSENSE_DIR/scripts",
      interpreter: "node",
      autorestart: true,
      max_restarts: 20,
      watch: false,
      env: { NODE_ENV: "production" }
    }
  ]
};
EOF

chown "$USER_NAME:$USER_NAME" ecosystem.config.js
echo "ecosystem.config.js criado com AI Worker + RTMP."

# Configure PM2 start on system boot
PM2_BIN=$(which pm2 2>/dev/null || echo "/usr/lib/node_modules/pm2/bin/pm2")
env PATH=$PATH:/usr/bin $PM2_BIN startup systemd -u "$USER_NAME" --hp "$HOME_DIR" 2>/dev/null || true

echo ""
echo -e "${GREEN}====================================================${NC}"
echo -e "${GREEN}  ViewSense Edge Worker installed successfully! ✅  ${NC}"
echo -e "${GREEN}====================================================${NC}"
echo ""
echo "Worker directory : $WORKER_DIR"
echo "RTMP directory   : $VIEWSENSE_DIR"
echo ""
echo "RTMP Ingest : rtmp://SERVER_IP:55935/live/{streamKey}"
echo "HLS Output  : http://SERVER_IP:8001/live/{streamKey}/index.m3u8"
echo "Snapshots   : http://SERVER_IP:8002/snapshots/{streamKey}.jpg"
echo ""
echo "NEXT STEPS:"
echo ""
echo "  1. Download your config.yaml:"
echo "     curl -sf -H \"x-api-key: YOUR_KEY\" \"YOUR_BOOTSTRAP_URL\" \\"
echo "       | python3 -c \"import sys,json; print(json.load(sys.stdin)['config_yaml'])\" \\"
echo "       > $WORKER_DIR/config.yaml"
echo ""
echo "  2. Start all processes:"
echo "     cd $WORKER_DIR && pm2 start ecosystem.config.js && pm2 save"
echo ""
echo "  3. View logs:"
echo "     pm2 logs"
echo ""
