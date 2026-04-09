"""
config_sync.py — Automatic config synchronization with ViewSense backend.

Periodically polls server-bootstrap to detect camera additions, removals,
and configuration changes, then applies them via hot-reload (no restart needed).
"""
import asyncio
import logging
import hashlib
import json
import os
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime

import httpx

logger = logging.getLogger(__name__)


class ConfigSync:
    def __init__(self, config: Dict[str, Any], http_client: httpx.AsyncClient):
        vs = config["viewsense"]
        self.server_id = vs.get("server_id", "")
        self.api_key = vs.get("api_key", "")
        self.anon_key = vs.get("anon_key", "")

        # Build bootstrap URL from existing API URL
        base = vs.get("api_url", "")
        if "/ingest-detections" in base:
            self.bootstrap_url = base.replace("/ingest-detections", "/server-bootstrap")
        elif "/functions/v1/" in base:
            self.bootstrap_url = base.rsplit("/functions/v1/", 1)[0] + "/functions/v1/server-bootstrap"
        else:
            self.bootstrap_url = ""

        self.interval = vs.get("config_sync_interval_seconds", 120)
        self.client = http_client
        self.running = False

        # State
        self.last_sync_at: Optional[str] = None
        self.last_sync_status: str = "pending"
        self.last_error: Optional[str] = None
        self.sync_count: int = 0

        # Callback set by CameraManager
        self._on_diff: Optional[Callable] = None

    def on_diff(self, callback: Callable):
        """Register callback: callback(added, removed, updated) -> None"""
        self._on_diff = callback

    async def fetch_remote_cameras(self) -> Optional[List[Dict[str, Any]]]:
        """Fetch camera list from server-bootstrap API."""
        if not self.bootstrap_url or not self.server_id:
            logger.warning("ConfigSync: bootstrap_url or server_id missing, skipping")
            return None

        try:
            headers = {
                "x-api-key": self.api_key,
                "apikey": self.anon_key,
            }
            resp = await self.client.get(
                self.bootstrap_url,
                params={"server_id": self.server_id},
                headers=headers,
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
            cameras = data.get("cameras", [])
            
            # Update sync interval on the fly if it changed in the backend
            config_data = data.get("config", {})
            new_interval = config_data.get("config_sync_interval_seconds")
            if new_interval is not None and int(new_interval) != self.interval:
                logger.info(f"⏱️ ConfigSync: Interval changed to {new_interval}s from dashboard")
                self.interval = int(new_interval)
                
            return self._normalize_camera_urls(cameras)
        except httpx.HTTPStatusError as e:
            logger.error(f"ConfigSync: HTTP {e.response.status_code} from bootstrap: {e.response.text[:200]}")
            return None
        except Exception as e:
            logger.error(f"ConfigSync: fetch failed: {e}")
            return None

    @staticmethod
    def _normalize_camera_urls(cameras: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize all RTMP stream URLs to rtmp://127.0.0.1:55935/live/{hash}
        so that the ConfigSync fingerprint matches what detector.py will use after
        its own URL rewrite. Without this, the fingerprint sees localhost:1935 vs
        127.0.0.1:55935 as different every sync cycle and restarts cameras every 120s.
        """
        import re as _re
        for cam in cameras:
            url = cam.get("stream_url", "")
            if not url:
                continue
            # Relative path: /live/hash
            if not url.startswith(("rtsp://", "rtmp://", "http://", "https://", "rtp://")):
                path = url if url.startswith("/") else f"/{url}"
                cam["stream_url"] = f"rtmp://127.0.0.1:55935{path}"
                logger.info(
                    f"🔧 Fixed stream URL for '{cam.get('name', cam.get('id'))}': "
                    f"'{url}' → '{cam['stream_url']}'"
                )
            elif url.startswith("rtmp://"):
                # Normalize any RTMP host/port to 127.0.0.1:55935 (matches detector rewrite)
                match = _re.search(r'rtmp://[^/]+(/.+)', url)
                if match:
                    path = match.group(1)
                    cam["stream_url"] = f"rtmp://127.0.0.1:55935{path}"
        return cameras


    @staticmethod
    def _camera_fingerprint(cam: Dict[str, Any]) -> str:
        """
        Hash of camera config fields that REQUIRE a full camera restart to apply.
        Fields like fps and confidence_threshold are handled by update_settings()
        hot-reload and must NOT be here — mismatches cause restart loops when
        Supabase has different values from the local config.yaml overrides.
        """
        relevant = {
            "stream_url": cam.get("stream_url", ""),
            # Normalize model name: 'yolov8n' == 'yolov8n.pt'
            "model": cam.get("model", "yolov8n").removesuffix(".pt"),
            "classes": sorted(cam.get("classes", [0])),
            "demographics": cam.get("demographics_enabled", False),
        }
        raw = json.dumps(relevant, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    def compute_diff(
        self,
        current_ids: set,
        remote_cameras: List[Dict[str, Any]],
    ) -> Tuple[List[Dict], List[str], List[Dict]]:
        """
        Compare running cameras with remote config.
        Returns: (added, removed_ids, updated)
        """
        remote_map = {cam["id"]: cam for cam in remote_cameras}
        remote_ids = set(remote_map.keys())

        added_ids = remote_ids - current_ids
        removed_ids = current_ids - remote_ids
        common_ids = current_ids & remote_ids

        added = [remote_map[cid] for cid in added_ids]
        removed = list(removed_ids)
        updated = []

        # For common cameras, check if config changed
        for cid in common_ids:
            remote_cam = remote_map[cid]
            remote_fp = self._camera_fingerprint(remote_cam)
            # We store fingerprints in _fingerprints dict managed by sync_once
            if hasattr(self, "_fingerprints") and self._fingerprints.get(cid) != remote_fp:
                updated.append(remote_cam)

        return added, removed, updated

    async def sync_once(self, current_camera_ids: set, current_configs: Dict[str, Dict] = None) -> Dict[str, Any]:
        """
        Perform a single sync cycle.
        Returns summary dict with counts.
        """
        result = {"added": 0, "removed": 0, "updated": 0, "error": None}

        remote_cameras = await self.fetch_remote_cameras()
        if remote_cameras is None:
            result["error"] = self.last_error or "fetch failed"
            self.last_sync_status = "error"
            self.last_error = result["error"]
            return result

        # Build fingerprints for current cameras
        if not hasattr(self, "_fingerprints"):
            self._fingerprints: Dict[str, str] = {}
        if current_configs:
            for cid, cfg in current_configs.items():
                if cid not in self._fingerprints:
                    self._fingerprints[cid] = self._camera_fingerprint(cfg)

        added, removed, updated = self.compute_diff(current_camera_ids, remote_cameras)

        # Update fingerprints for remote cameras
        for cam in remote_cameras:
            self._fingerprints[cam["id"]] = self._camera_fingerprint(cam)
        # Remove fingerprints for removed cameras
        for cid in removed:
            self._fingerprints.pop(cid, None)

        result["added"] = len(added)
        result["removed"] = len(removed)
        result["updated"] = len(updated)

        if added or removed or updated:
            log_parts = []
            if added:
                log_parts.append(f"+{len(added)} câmeras")
            if removed:
                log_parts.append(f"-{len(removed)} câmeras")
            if updated:
                log_parts.append(f"~{len(updated)} alteradas")
            logger.info(f"🔄 Config Sync: {', '.join(log_parts)}")

            if self._on_diff:
                await self._on_diff(added, removed, updated)
        else:
            logger.debug("🔄 Config Sync: nenhuma mudança detectada")

        # -------------------------------------------------------------
        # Write active streams whitelist mapping for Node.js RTMP server
        # This allows the RTMP server to reject streams outside schedule
        # -------------------------------------------------------------
        try:
            import re as _re
            active_hashes = []
            for cam in remote_cameras:
                url = cam.get("stream_url", "")
                # Extract 48-char hex hash from /live/{hash}
                m = _re.search(r'/live/([a-f0-9]{40,})', url)
                if m:
                    active_hashes.append(m.group(1))
            dump_path = "/opt/viewsense/active_streams.json" if os.path.exists("/opt/viewsense") else "active_streams.json"
            with open(dump_path, "w") as f:
                json.dump(active_hashes, f)
            logger.debug(f"active_streams.json: {len(active_hashes)} hashes escritos")
        except Exception as e:
            logger.error(f"Failed to write active_streams.json: {e}")

        self.last_sync_at = datetime.utcnow().isoformat() + "Z"
        self.last_sync_status = "ok"
        self.last_error = None
        self.sync_count += 1

        return result

    async def loop(self, get_current_state: Callable):
        """
        Background sync loop.
        get_current_state() should return (set_of_camera_ids, dict_of_camera_configs)
        """
        self.running = True
        logger.info(f"🔄 ConfigSync loop started (interval={self.interval}s)")

        # Wait a bit before first sync to let cameras initialize
        await asyncio.sleep(10)

        while self.running:
            try:
                camera_ids, camera_configs = get_current_state()
                await self.sync_once(camera_ids, camera_configs)
            except Exception as e:
                logger.error(f"ConfigSync loop error: {e}")
                self.last_sync_status = "error"
                self.last_error = str(e)

            await asyncio.sleep(self.interval)

    def stop(self):
        self.running = False

    def get_status(self) -> Dict[str, Any]:
        return {
            "last_sync_at": self.last_sync_at,
            "last_sync_status": self.last_sync_status,
            "last_error": self.last_error,
            "sync_count": self.sync_count,
            "interval_seconds": self.interval,
            "bootstrap_url": self.bootstrap_url[:50] + "..." if self.bootstrap_url else None,
        }
