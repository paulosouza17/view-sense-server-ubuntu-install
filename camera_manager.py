import yaml
import logging
import asyncio
from typing import Dict
from api_client import APIClient
from detector import CameraDetector
from roi_sync import ROISyncManager

logger = logging.getLogger(__name__)

class CameraManager:
    def __init__(self, config_path: str = "config.yaml"):
        # ... existing init ...
        self.config_path = config_path
        self.cameras: Dict[str, CameraDetector] = {}
        self.config = self._load_config()
        self.api_client = None
        self.roi_manager = None # Store ROI Manager

    def _load_config(self):
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def start(self):
        # Initialize API Client
        self.api_client = APIClient(self.config)
        
        # Initialize ROI Sync Manager
        self.roi_manager = ROISyncManager(self.api_client, self.config)
        self.roi_manager.on_roi_change(self.on_roi_updated)

        # Start Camera Detectors
        for cam_conf in self.config['cameras']:
            self._start_camera(cam_conf)
            
        # Update active camera count for heartbeat
        if self.api_client:
            self.api_client.active_cameras = len(self.cameras)
            # Start ROI Sync Task
            asyncio.create_task(self.roi_manager.start())

    def on_roi_updated(self, camera_id: str, rois: list, camera_config: dict):
        """Callback from ROISyncManager when ROIs change."""
        if camera_id in self.cameras:
            detector = self.cameras[camera_id]
            detector.update_settings(rois, camera_config)
            logger.info(f"Updated settings for camera {camera_id} via ROI Sync")
        else:
            # New camera? Logic to add it would go here if we support dynamic camera add.
            # For now, we only update existing ones.
            logger.warning(f"Received ROI update for unknown camera {camera_id}")

    def _start_camera(self, cam_conf):
        cam_id = cam_conf['id']
        if cam_id in self.cameras:
            logger.warning(f"Camera {cam_id} already running.")
            return

        detector = CameraDetector(cam_conf, self.api_client)
        detector.start()
        self.cameras[cam_id] = detector
        logger.info(f"Started camera {cam_id}")

    def stop_camera(self, cam_id: str):
        if cam_id in self.cameras:
            self.cameras[cam_id].stop()
            del self.cameras[cam_id]
            logger.info(f"Stopped camera {cam_id}")

    def restart_camera(self, cam_id: str):
        logger.info(f"Restarting camera {cam_id}...")
        cam_conf = next((c for c in self.config['cameras'] if c['id'] == cam_id), None)
        if not cam_conf:
            logger.error(f"Config not found for camera {cam_id}")
            return False
            
        self.stop_camera(cam_id)
        self._start_camera(cam_conf)
        return True

    def reload_config(self):
        logger.info("Reloading configuration...")
        new_config = self._load_config()
        # Simple strategy: stop all and restart all.
        # Smarter: diff config.
        # For now, just logging that we reloaded config in memory, 
        # implementing full hot-reload might be complex due to state.
        self.config = new_config
        # Note: This doesn't apply changes to running cameras automatically in this simple implementation
        return "Config reloaded (restart required for active cameras)"

    def get_all_status(self):
        return [cam.get_status() for cam in self.cameras.values()]

    async def shutdown(self):
        logger.info("Shutting down manager...")
        for cam_id in list(self.cameras.keys()):
            self.stop_camera(cam_id)
        if self.api_client:
            await self.api_client.close()
