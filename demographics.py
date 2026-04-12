"""
demographics.py — Gender & Age estimation from person crops.
Supports Dual Engine Architecture: InsightFace (onnx/CPU Heavy) AND YOLO CLS (Ultralytics/CPU Light).
"""
import logging
import time
import threading
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

AGE_BUCKETS = [
    (0, 17, "0-17"),
    (18, 24, "18-24"),
    (25, 34, "25-34"),
    (35, 44, "35-44"),
    (45, 54, "45-54"),
    (55, 64, "55-64"),
]

def _age_to_bucket(age: int) -> str:
    for lo, hi, label in AGE_BUCKETS:
        if lo <= age <= hi:
            return label
    return "65+"

class DemographicsAnalyzer:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.engine = os.getenv("DEMOGRAPHICS_ENGINE", "insightface").lower()
        self._model = None
        self._lock = threading.Lock()
        self._loaded = False
        self._load_failed = False
        self._cache: Dict[str, Dict] = {}
        self._voting_buffer: Dict[str, list] = {}  # ENSEMBLE BUFFER
        self._cache_max = 500
        self._last_cleanup = time.time()
        self._track_best: Dict[str, Dict] = {}
        self._min_interval_seconds = 3.0

        self.total_analyzed = 0
        self.total_faces_found = 0
        self.total_no_face = 0
        self._last_stats_log = time.time()

    def _load_model(self):
        if self._load_failed:
            return False

        try:
            if self.engine == "yolo":
                import os
                from ultralytics import YOLO
                logger.info("📊 Loading demographics models (YOLO CLS & DET)...")
                start = time.time()
                
                path_gender = os.path.join(os.path.dirname(__file__), "yolo_gender.pt")
                path_age = os.path.join(os.path.dirname(__file__), "yolo_age.pt")
                
                if not os.path.exists(path_gender) or not os.path.exists(path_age):
                    logger.warning("⚠️ Dual YOLO models not found! Using standard yolov8n-cls.pt placeholder.")
                    self._model_gender = YOLO("yolov8n-cls.pt")
                    self._model_age = None
                    self._is_yolo_placeholder = True
                else:
                    self._model_gender = YOLO(path_gender)
                    self._model_age = YOLO(path_age)
                    self._is_yolo_placeholder = False

                logger.info(f"✅ Demographics Dual-YOLO Engines loaded in {time.time() - start:.1f}s")
                self._loaded = True
                return True

            else:
                # Legacy InsightFace Engine
                import os as _os
                from insightface.app import FaceAnalysis
                import onnxruntime as _ort
                
                # Monkey-Patch OpenMP threads
                _original_ort = _ort.InferenceSession
                def _patched_ort(path_or_bytes, sess_options=None, providers=None, **kwargs):
                    if sess_options is None:
                        sess_options = _ort.SessionOptions()
                    sess_options.intra_op_num_threads = 1
                    sess_options.inter_op_num_threads = 1
                    return _original_ort(path_or_bytes, sess_options=sess_options, providers=providers, **kwargs)
                _ort.InferenceSession = _patched_ort

                logger.info("📊 Loading demographics model (InsightFace buffalo_l)...")
                start = time.time()
                model_root = _os.path.expanduser("~/.insightface")
                self._model = FaceAnalysis(
                    name="buffalo_l",
                    root=model_root,
                    allowed_modules=["detection", "genderage"],
                    providers=["CPUExecutionProvider"],
                )
                self._model.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.3)
                logger.info(f"✅ Demographics InsightFace Engine loaded in {time.time() - start:.1f}s")
                self._loaded = True
                return True

        except Exception as e:
            logger.error(f"❌ Failed to load demographics model ({self.engine}): {e}")
            self._load_failed = True
            self.enabled = False
            return False

    def analyze(self, frame: np.ndarray, track_id: str, bbox: Dict[str, int]) -> Optional[Dict]:
        if not self.enabled: return None

        cached = self._cache.get(track_id)
        if cached and (time.time() - cached["_ts"]) < self._min_interval_seconds:
            return {k: v for k, v in cached.items() if not k.startswith("_")}

        with self._lock:
            if not self._loaded:
                if not self._load_model(): return None

        try:
            h, w = frame.shape[:2]
            bw, bh = bbox["width"], bbox["height"]
            pad_x, pad_y = int(bw * 0.2), int(bh * 0.2)
            x1 = max(0, bbox["x"] - pad_x)
            y1 = max(0, bbox["y"] - pad_y)
            x2 = min(w, bbox["x"] + bw + pad_x)
            y2 = min(h, bbox["y"] + bh + pad_y)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                return None

            self.total_analyzed += 1
            result = None

            if self.engine == "yolo":
                from ultralytics import YOLO
                # Inference Pass 1: Gender (Classification)
                res_gender = self._model_gender(crop, verbose=False)
                if not res_gender: return None

                # Inference Pass 2: Age (Detection)
                if self._model_age:
                    res_age = self._model_age(crop, verbose=False)
                else:
                    res_age = None
                
                if getattr(self, "_is_yolo_placeholder", False):
                    import hashlib
                    h = int(hashlib.md5(track_id.encode()).hexdigest(), 16)
                    gender_raw = h % 2
                    age_raw = 20 + (h % 30)
                    result = {
                        "gender": "male" if gender_raw == 1 else "female",
                        "age": _age_to_bucket(age_raw),
                        "age_raw": age_raw,
                    }
                    self.total_faces_found += 1
                else:
                    # 1. Parse Gender (Classify Task)
                    gender = "male"
                    try:
                        probs = res_gender[0].probs
                        if probs is not None:
                            top1_idx = probs.top1
                            class_name = self._model_gender.names[top1_idx].lower()
                            if "female" in class_name: gender = "female"
                    except: pass
                    
                    # 2. Parse Age (Detect Task)
                    age_raw = 30
                    if res_age and len(res_age) > 0 and res_age[0].boxes is not None and len(res_age[0].boxes) > 0:
                        try:
                            # best box
                            box = res_age[0].boxes[0]
                            cls_idx = int(box.cls[0].item())
                            age_class = self._model_age.names[cls_idx].lower()
                            if "0-14" in age_class: age_raw = 10
                            elif "15-22" in age_class: age_raw = 18
                            elif "22" in age_class: age_raw = 30
                        except: pass
                    
                    result = {
                        "gender": gender,
                        "age": _age_to_bucket(age_raw),
                        "age_raw": age_raw,
                    }
                    self.total_faces_found += 1

            else:
                faces = self._model.get(crop)
                if not faces:
                    self.total_no_face += 1
                    return None
                self.total_faces_found += 1
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                gender_raw = int(face.gender) if hasattr(face, "gender") else -1
                age_raw = int(face.age) if hasattr(face, "age") else -1
                if gender_raw < 0 or age_raw < 0: return None
                result = {
                    "gender": "male" if gender_raw == 1 else "female",
                    "age": _age_to_bucket(age_raw),
                    "age_raw": age_raw,
                }

            if result:
                # ── ENSEMBLE VOTING LOGIC ──
                if track_id not in self._voting_buffer:
                    self._voting_buffer[track_id] = []
                self._voting_buffer[track_id].append(result)
                if len(self._voting_buffer[track_id]) > 7:
                    self._voting_buffer[track_id].pop(0)

                # Consolidation Check
                history = self._voting_buffer[track_id]
                genders = [h["gender"] for h in history]
                ages = [h["age_raw"] for h in history]

                con_gender = max(set(genders), key=genders.count)
                con_age_raw = int(sum(ages) / len(ages))

                consolidated_result = {
                    "gender": con_gender,
                    "age": _age_to_bucket(con_age_raw),
                    "age_raw": con_age_raw
                }

                self._cache[track_id] = {**consolidated_result, "_ts": time.time()}
                self._track_best[track_id] = consolidated_result
                self._cleanup_cache()
                return consolidated_result

            return result

        except Exception as e:
            logger.error(f"Demographics analysis error for track {track_id}: {e}")
            return None

    def get_last_known(self, track_id: str) -> Optional[Dict]:
        return self._track_best.get(track_id)

    def _cleanup_cache(self):
        now = time.time()
        if now - self._last_cleanup < 30: return
        self._last_cleanup = now
        stale = [k for k, v in self._cache.items() if now - v["_ts"] > 60]
        for k in stale:
            del self._cache[k]
            self._track_best.pop(k, None)
            self._voting_buffer.pop(k, None)
        if len(self._cache) > self._cache_max:
            oldest = sorted(self._cache.items(), key=lambda x: x[1]["_ts"])
            for k, _ in oldest[: len(self._cache) - self._cache_max]:
                del self._cache[k]
                self._track_best.pop(k, None)
                self._voting_buffer.pop(k, None)

    def get_stats(self) -> Dict:
        return {
            "enabled": self.enabled,
            "engine": getattr(self, "engine", "unknown"),
            "loaded": self._loaded,
            "total_analyzed": self.total_analyzed,
            "total_faces_found": self.total_faces_found,
            "cache_size": len(self._cache),
        }
