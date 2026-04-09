"""
demographics.py — Gender & Age estimation from person crops.
Uses InsightFace (buffalo_sc model) for lightweight face analysis.
Only outputs statistics — NO images are ever stored.
"""
import logging
import time
import threading
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Age bucket boundaries matching frontend DemographicsChart
AGE_BUCKETS = [
    (0, 17, "0-17"),
    (18, 24, "18-24"),
    (25, 34, "25-34"),
    (35, 44, "35-44"),
    (45, 54, "45-54"),
    (55, 64, "55-64"),
]


def _age_to_bucket(age: int) -> str:
    """Convert raw age estimate to age bucket string."""
    for lo, hi, label in AGE_BUCKETS:
        if lo <= age <= hi:
            return label
    return "65+"


class DemographicsAnalyzer:
    """
    Analyzes person crops for gender and approximate age.
    Thread-safe, lazy-loads model on first use.
    Processes ONLY persons (class 0 in COCO).
    Never stores face images — results are statistical only.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._model = None
        self._lock = threading.Lock()
        self._loaded = False
        self._load_failed = False

        # Rate-limiting cache: track_id → last analyzed result (3s window)
        self._cache: Dict[str, Dict] = {}
        self._cache_max = 500
        self._last_cleanup = time.time()

        # Persistent best-known demographics per track (used at crossing time).
        # Unlike _cache, these are NOT expired — they survive as long as the
        # tracker keeps the same track_id, so crossings pick up demographics
        # even when the face isn't visible in the crossing frame.
        self._track_best: Dict[str, Dict] = {}

        # Rate limiting: analyze at most every N frames per track
        # 3s balances CPU load vs. data capture before line crossing
        self._min_interval_seconds = 3.0

        # Stats
        self.total_analyzed = 0
        self.total_faces_found = 0
        self.total_no_face = 0
        self._last_stats_log = time.time()

    def _load_model(self):
        """Lazy-load InsightFace buffalo_l model (includes genderage.onnx)."""
        if self._load_failed:
            return False

        try:
            import os as _os
            from insightface.app import FaceAnalysis

            logger.info("📊 Loading demographics model (InsightFace buffalo_l)...")
            start = time.time()

            # Force model root so buffalo_sc is never used as a fallback.
            # NOTE: InsightFace appends 'models/' internally, so root must be
            # ~/.insightface (NOT ~/.insightface/models — that creates models/models/).
            model_root = _os.path.expanduser("~/.insightface")

            # buffalo_l is required — it contains genderage.onnx.
            # buffalo_sc does NOT include genderage and will silently produce no results.
            self._model = FaceAnalysis(
                name="buffalo_l",
                root=model_root,
                allowed_modules=["detection", "genderage"],
                providers=["CPUExecutionProvider"],
            )
            # det_thresh=0.3 (permissive) — security cameras often produce blurry partial faces.
            # det_size=640 — standard resolution for best accuracy on cropped person images.
            self._model.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.3)

            # Verify genderage module is actually loaded
            loaded_names = [m.taskname for m in self._model.models.values()]
            logger.info(f"📊 Loaded modules: {loaded_names}")
            if "genderage" not in loaded_names:
                raise RuntimeError(
                    "genderage module not found — buffalo_l may be incomplete. "
                    f"Found: {loaded_names}"
                )

            elapsed = time.time() - start
            logger.info(f"✅ Demographics model (buffalo_l) loaded in {elapsed:.1f}s")
            self._loaded = True
            return True

        except ImportError:
            logger.warning(
                "⚠️ InsightFace not installed. Demographics disabled. "
                "Install with: pip install insightface onnxruntime"
            )
            self._load_failed = True
            self.enabled = False
            return False
        except Exception as e:
            logger.error(f"❌ Failed to load demographics model: {e}")
            self._load_failed = True
            self.enabled = False
            return False

    def analyze(
        self, frame: np.ndarray, track_id: str, bbox: Dict[str, int]
    ) -> Optional[Dict]:
        """
        Analyze a detected person for gender and age.

        Args:
            frame: Full video frame (BGR, np.ndarray)
            track_id: Unique tracking ID for this person
            bbox: Bounding box {"x", "y", "width", "height"}

        Returns:
            Dict with "gender", "age", "age_raw", "confidence" or None
        """
        if not self.enabled:
            return None

        # Check cache — don't re-analyze same person within interval
        cached = self._cache.get(track_id)
        if cached and (time.time() - cached["_ts"]) < self._min_interval_seconds:
            return {k: v for k, v in cached.items() if not k.startswith("_")}

        with self._lock:
            if not self._loaded:
                if not self._load_model():
                    return None

        try:
            # Crop person from frame with padding
            # Add 20% spatial padding around the YOLO bounding box to ensure the
            # person's head isn't clipped out by a tightly fitted tracking box.
            h, w = frame.shape[:2]
            bw = bbox["width"]
            bh = bbox["height"]
            pad_x = int(bw * 0.2)
            pad_y = int(bh * 0.2)

            x1 = max(0, bbox["x"] - pad_x)
            y1 = max(0, bbox["y"] - pad_y)
            x2 = min(w, bbox["x"] + bw + pad_x)
            y2 = min(h, bbox["y"] + bh + pad_y)

            # Use the entire person crop instead of aggressively slicing the head off
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                return None

            # Run face detection + gender/age classification
            self.total_analyzed += 1
            faces = self._model.get(crop)

            if not faces:
                self.total_no_face += 1
                # Periodically log stats so operator can see detection rate
                now_t = time.time()
                if now_t - self._last_stats_log > 30:
                    self._last_stats_log = now_t
                    rate = (
                        f"{self.total_faces_found / self.total_analyzed * 100:.1f}%"
                        if self.total_analyzed > 0 else "N/A"
                    )
                    logger.info(
                        f"📊 Demographics stats — analyzed: {self.total_analyzed}, "
                        f"faces_found: {self.total_faces_found}, no_face: {self.total_no_face}, "
                        f"face_rate: {rate}"
                    )
                return None

            self.total_faces_found += 1

            # Use the largest (most confident) face
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

            gender_raw = int(face.gender) if hasattr(face, "gender") else -1
            age_raw = int(face.age) if hasattr(face, "age") else -1

            if gender_raw < 0 or age_raw < 0:
                return None

            result = {
                "gender": "male" if gender_raw == 1 else "female",
                "age": _age_to_bucket(age_raw),
                "age_raw": age_raw,
            }

            # Rate-limit cache (3s)
            self._cache[track_id] = {**result, "_ts": time.time()}
            # Persistent best-known (used at crossing time)
            self._track_best[track_id] = result
            self._cleanup_cache()

            return result

        except Exception as e:
            logger.error(f"Demographics analysis error for track {track_id}: {e}")
            return None

    def get_last_known(self, track_id: str) -> Optional[Dict]:
        """
        Return the last successfully detected demographic for a track,
        regardless of when it was detected.

        Use at crossing time: if analyze() returns None (face not visible),
        fall back to this so crossings still carry gender/age from earlier
        in the track's lifetime.
        """
        return self._track_best.get(track_id)

    def _cleanup_cache(self):
        """Remove stale entries from rate-limit cache and track_best."""
        now = time.time()
        if now - self._last_cleanup < 30:
            return

        self._last_cleanup = now
        stale = [
            k for k, v in self._cache.items()
            if now - v["_ts"] > 60
        ]
        for k in stale:
            del self._cache[k]
            # Also evict from track_best when the track is gone (>60s)
            self._track_best.pop(k, None)

        # Hard cap
        if len(self._cache) > self._cache_max:
            oldest = sorted(self._cache.items(), key=lambda x: x[1]["_ts"])
            for k, _ in oldest[: len(self._cache) - self._cache_max]:
                del self._cache[k]
                self._track_best.pop(k, None)


    def get_stats(self) -> Dict:
        """Return diagnostic stats."""
        return {
            "enabled": self.enabled,
            "loaded": self._loaded,
            "total_analyzed": self.total_analyzed,
            "total_faces_found": self.total_faces_found,
            "cache_size": len(self._cache),
            "face_rate": (
                f"{self.total_faces_found / self.total_analyzed * 100:.0f}%"
                if self.total_analyzed > 0
                else "N/A"
            ),
        }
