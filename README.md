# ViewSense YOLO Server

This project is a complete Python application for Ubuntu servers that connects to RTSP/RTMP cameras, detects objects using YOLOv8, tracks them with ByteTrack, counts line crossings, and sends data to the ViewSense API.

## Features
- **Object Detection:** Uses YOLOv8 (nano/medium) for real-time person and vehicle detection.
- **Tracking:** Implements ByteTrack for robust object tracking.
- **Line Counting:** Counts objects crossing defined lines (In/Out).
- **API Integration:** Sends detection events to ViewSense API with retry logic.
- **Camera Management:** Handles multiple RTSP streams with auto-reconnection.
- **API Control:** FastAPI endpoints for status, metrics, and configuration reloading.

## Prerequisites
- **OS:** Ubuntu Server 20.04/22.04 (or compatible Linux) / macOS
- **Python:** 3.11+
- **Hardware:** NVIDIA GPU recommended (optional)

## Installation

### Option 1: Auto Installer (Ubuntu Recommended)
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/paulosouza17/view-sense-server-ubuntu-install.git
    cd view-sense-server-ubuntu-install
    ```

2.  **Run the installer:**
    ```bash
    chmod +x install.sh
    sudo ./install.sh
    ```
    *This script installs system dependencies, creates a virtual environment, and sets up a systemd service.*

3.  **Configure:**
    Edit `config.yaml` to add your API keys and camera details.
    ```bash
    nano config.yaml
    ```

4.  **Start the Service:**
    ```bash
    sudo systemctl start viewsense-yolo
    ```

### Option 2: Docker
1.  **Configure `config.yaml` first.**
2.  **Run with Docker Compose:**
    ```bash
    docker-compose up -d
    ```

## Configuration (config.yaml)
```yaml
viewsense:
  api_url: "https://your-api-url"
  api_key: "your-key"

cameras:
  - id: "cam-uuid"
    stream_url: "rtsp://..."
    ...
```

## Maintenance
- **Check Status:** `curl http://localhost:8080/status`
- **Restart Camera:** `curl -X POST http://localhost:8080/cameras/{id}/restart`
- **Reload Config:** `curl -X POST http://localhost:8080/config/reload`
- **Logs:** `sudo journalctl -u viewsense-yolo -f`
