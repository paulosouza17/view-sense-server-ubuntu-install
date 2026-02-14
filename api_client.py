import asyncio
import logging
import time
from typing import Dict, List, Any
import httpx

# Configure logging
logger = logging.getLogger(__name__)

class APIClient:
    def __init__(self, api_url: str, api_key: str, anon_key: str = "", batch_size: int = 20):
        self.api_url = api_url
        self.api_key = api_key
        self.anon_key = anon_key # Supabase anon key
        self.batch_size = batch_size
        self.queue: List[Dict[str, Any]] = []
        
        # Headers for Supabase Edge Functions
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key, # Custom Auth
            "apikey": self.anon_key    # Supabase Gateway Auth
        }
        
        if not self.anon_key:
            logger.warning("anon_key is missing! Supabase requests will likely fail with 401.")
            
        self.client = httpx.AsyncClient(headers=self.headers, timeout=10.0)

    async def add_detection(self, detection: Dict[str, Any]):
        """Adds a detection to the buffer and sends if batch size is reached."""
        self.queue.append(detection)
        if len(self.queue) >= self.batch_size:
            await self.flush()

    async def flush(self):
        """Sends all buffered detections to the API."""
        if not self.queue:
            return

        payload = {
            # In a real scenario, we might want to group by camera_id if the API expects it,
            # but the spec says "detections": [...]. 
            # If the API expects a single camera_id per request, we'd need to group.
            # The prompt example shows:
            # {
            #   "camera_id": "uuid",
            #   "detections": [...]
            # }
            # Since we might have multiple cameras, let's group by camera_id.
        }
        
        # Group detections by camera_id
        grouped_detections: Dict[str, List[Dict[str, Any]]] = {}
        for det in self.queue:
            cam_id = det.get("camera_id")
            if not cam_id:
                continue
            if cam_id not in grouped_detections:
                grouped_detections[cam_id] = []
            
            # Remove camera_id from the detection object itself if api expects it at root
            # strictly speaking based on prompt:
            # "detections": [ { ... } ]
            # so we keep it clean.
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
                await self._send_with_retry(message)
            except Exception as e:
                logger.error(f"Failed to send detections for camera {cam_id}: {e}")
                # Optionally re-queue or drop. For now, we drop to avoid memory leaks if API is down long-term
                # but in production, a persistent queue (Redis/SQLite) is better.

    async def _send_with_retry(self, json_payload: Dict[str, Any], max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                response = await self.client.post(self.api_url, json=json_payload)
                response.raise_for_status()
                logger.debug(f"Successfully sent {len(json_payload['detections'])} detections.")
                return
            except httpx.HTTPError as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def close(self):
        await self.flush()
        await self.client.aclose()
