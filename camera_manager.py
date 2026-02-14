import yaml
import logging
import asyncio
from typing import Dict
from api_client import APIClient
from detector import CameraDetector

logger = logging.getLogger(__name__)

class CameraManager:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.cameras: Dict[str, CameraDetector] = {}
        self.config = self._load_config()
        self.api_client = None

    def _load_config(self):
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def start(self):
        # Initialize API Client
        api_config = self.config['viewsense']
        # We need the event loop for the async API client
        loop = asyncio.get_event_loop()
        
        # New: Get anon_key for Supabase
        anon_key = api_config.get('anon_key', '')
        
        self.api_client = APIClient(
            api_url=api_config['api_url'],
            api_key=api_config['api_key'],
            anon_key=anon_key,
            batch_size=api_config.get('batch_size', 20)
        )
        self.api_client.loop = loop # Inject loop for threadsafe calls

        # Start Camera Detectors
        for cam_conf in self.config['cameras']:
            self._start_camera(cam_conf)

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
