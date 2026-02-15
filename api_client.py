import asyncio
import logging
import time
import psutil
import platform
from typing import Dict, List, Any
import httpx

# Configure logging
logger = logging.getLogger(__name__)

class APIClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        viewsense_conf = config['viewsense']
        
        # Identity
        self.server_id = viewsense_conf.get('server_id', 'unknown_server')
        self.server_secret = viewsense_conf.get('server_secret', '')
        self.version = "1.0.0"

        # Explicit URLs from config
        self.ingest_url = viewsense_conf.get('api_url', '') # Fallback or main URL
        self.heartbeat_url = viewsense_conf.get('heartbeat_url', '')
        self.roi_sync_url = viewsense_conf.get('roi_sync_url', '')
        
        # Keys
        self.api_key = viewsense_conf.get('api_key', '')
        self.anon_key = viewsense_conf.get('anon_key', '')
        
        self.batch_size = viewsense_conf.get('batch_size', 20)
        self.queue: List[Dict[str, Any]] = []
        
        # Headers for Supabase Edge Functions
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key, 
            "apikey": self.anon_key,    # Supabase Gateway Auth
            "x-server-secret": self.server_secret # Server auth if needed via header
        }
        
        if not self.anon_key:
            logger.warning("anon_key is missing! Supabase requests will likely fail with 401.")
            
        self.client = httpx.AsyncClient(headers=self.headers, timeout=10.0)
        self.loop = asyncio.get_event_loop()
        self.running = True
        self.active_cameras = 0 # To be updated by manager

    async def add_detection(self, detection: Dict[str, Any]):
        """Adds a detection to the buffer and sends if batch size is reached."""
        self.queue.append(detection)
        if len(self.queue) >= self.batch_size:
            await self.flush()

    async def flush(self):
        """Sends all buffered detections to the API."""
        if not self.queue:
            return

        # Group detections by camera_id
        grouped_detections: Dict[str, List[Dict[str, Any]]] = {}
        for det in self.queue:
            cam_id = det.get("camera_id")
            if not cam_id:
                continue
            if cam_id not in grouped_detections:
                grouped_detections[cam_id] = []
            
            det_clean = det.copy()
            det_clean.pop("camera_id", None)
            grouped_detections[cam_id].append(det_clean)

        self.queue.clear()

        for cam_id, dets in grouped_detections.items():
            message = {
                "camera_id": cam_id,
                "detections": dets
            }
            
            try:
                await self._send_with_retry(self.ingest_url, message)
            except Exception as e:
                logger.error(f"Failed to send detections for camera {cam_id}: {e}")

    async def _send_with_retry(self, url: str, json_payload: Dict[str, Any], max_retries: int = 3):
        if not url:
             logger.error("No URL configured for this operation.")
             return

        for attempt in range(max_retries):
            try:
                response = await self.client.post(url, json=json_payload)
                response.raise_for_status()
                # logger.debug(f"Successfully sent payload to {url}")
                return
            except httpx.HTTPError as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {url}: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def send_heartbeat(self):
        if not self.heartbeat_url:
            return
            
        payload = {
            "server_id": self.server_id,
            "server_secret": self.server_secret,
            "cpu_usage": psutil.cpu_percent(interval=0.1),
            "ram_usage": psutil.virtual_memory().used // (1024 * 1024),
            "ram_total": psutil.virtual_memory().total // (1024 * 1024),
            "uptime_seconds": int(time.time() - psutil.boot_time()),
            "cameras_active": self.active_cameras,
            "hostname": platform.node(),
            "version": self.version,
        }
        
        # Try to get temperature (Linux specific)
        try:
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    if "cpu_thermal" in temps:
                        payload["cpu_temp"] = temps["cpu_thermal"][0].current
                    elif "coretemp" in temps:
                        payload["cpu_temp"] = temps["coretemp"][0].current
        except Exception:
            pass
        
        try:
            await self._send_with_retry(self.heartbeat_url, payload, max_retries=1)
            logger.info("💓 Heartbeat sent.")
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")

    async def heartbeat_loop(self):
        interval = self.config["viewsense"].get("heartbeat_interval_seconds", 15)
        logger.info(f"Starting heartbeat loop (interval={interval}s)...")
        while self.running:
            await self.send_heartbeat()
            await asyncio.sleep(interval)
            
    async def roi_sync_loop(self):
        # Deprecated: Handled by ROISyncManager now, but kept for compatibility logic if needed
        # We will disable the internal loop in main in favor of ROISyncManager
        pass

    async def sync_rois(self) -> Optional[Dict[str, Any]]:
        """Sincroniza ROIs do painel web via endpoint /roi-sync"""
        if not self.roi_sync_url or not self.server_id:
            logger.warning("ROI Sync skipped: roi_sync_url or server_id missing.")
            return None
            
        try:
            response = await self.client.get(
                self.roi_sync_url,
                params={"server_id": self.server_id}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            # logger.error(f"Sync ROIs failed: {e}") 
            # Log warning to avoid spamming error on transient network issues
            logger.warning(f"Sync ROIs failed: {e}")
            return {"error": str(e)}

    async def close(self):
        self.running = False
        await self.flush()
        await self.client.aclose()
