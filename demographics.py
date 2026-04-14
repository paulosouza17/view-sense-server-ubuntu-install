"""
demographics.py — Gender & Age estimation from person crops.

ARCHITECTURE: Singleton + Batch Inference (Level 2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Um único DemographicsAnalyzer é criado para todo o processo (Singleton).
  • Todas as câmeras compartilham a mesma instância via get_analyzer().
  • Requisições de inferência entram numa queue e são processadas em
    batches de até BATCH_SIZE=4 por um worker thread dedicado.
  • Isso evita a multiplicação de thread pools ONNX (60+ threads → ~15).

Resultado esperado:
  CPU: 400% → ~80-100%  |  FPS: instável → 5-8 estável
"""
import logging
import queue
import threading
import time
import os
from typing import Dict, List, Optional, Tuple

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


# ─── Batch Worker ──────────────────────────────────────────────────────────────

BATCH_SIZE    = 4     # crops processados por chamada YOLO/ONNX
BATCH_TIMEOUT = 2.0   # segundos máx aguardando acumular batch

class _InferenceRequest:
    """Unidade de trabalho enfileirada por câmera."""
    def __init__(self, crop: np.ndarray, track_id: str, face_crop: np.ndarray, face_found: bool):
        self.crop       = crop
        self.track_id   = track_id
        self.face_crop  = face_crop
        self.face_found = face_found
        self.event      = threading.Event()
        self.result: Optional[Dict] = None


class DemographicsAnalyzer:
    """
    Singleton Batch Inference Engine.
    Não instanciar diretamente — use get_analyzer().
    """

    def __init__(self, enabled: bool = True):
        self.enabled       = enabled
        self.engine        = os.getenv("DEMOGRAPHICS_ENGINE", "yolo").lower()
        self._loaded       = False
        self._load_failed  = False
        self._lock         = threading.Lock()

        # Cache por track_id (evita re-análise dentro de _min_interval_seconds)
        self._cache:          Dict[str, Dict]  = {}
        self._voting_buffer:  Dict[str, list]  = {}
        self._track_best:     Dict[str, Dict]  = {}
        self._cache_max       = 500
        self._last_cleanup    = time.time()
        self._min_interval_seconds = 3.0

        # Telemetria
        self.total_analyzed    = 0
        self.total_faces_found = 0
        self.total_no_face     = 0
        self._last_stats_log   = time.time()

        # Batch worker
        self._req_queue: queue.Queue = queue.Queue(maxsize=64)
        self._worker_thread = threading.Thread(
            target=self._batch_worker_loop,
            name="demographics-batch-worker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("[demographics] Singleton + Batch worker iniciado (batch=%d, timeout=%.1fs)",
                    BATCH_SIZE, BATCH_TIMEOUT)

    # ── Model Loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> bool:
        if self._load_failed:
            return False
        try:
            start = time.time()
            if self.engine == "yolo":
                from ultralytics import YOLO as _YOLO
                base = os.path.dirname(__file__)
                path_gender = os.path.join(base, "yolo_gender.pt")
                path_age    = os.path.join(base, "yolo_age.pt")

                if not os.path.exists(path_gender) or not os.path.exists(path_age):
                    logger.warning("[demographics] YOLO gender/age .pt não encontrado — placeholder ativo")
                    self._model_gender = _YOLO("yolov8n-cls.pt")
                    self._model_age    = None
                    self._is_placeholder = True
                else:
                    self._model_gender = _YOLO(path_gender)
                    self._model_age    = _YOLO(path_age)
                    self._is_placeholder = False

                # Haar Cascade (face detection — CPU puro, sem ONNX pool)
                self._face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )

                # ONNX Age Regressor (thread único explícito)
                age_onnx = os.path.expanduser("~/.insightface/models/buffalo_l/genderage.onnx")
                if os.path.exists(age_onnx):
                    import onnxruntime as ort
                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = 1
                    opts.inter_op_num_threads = 1
                    self._model_age_ort = ort.InferenceSession(
                        age_onnx, sess_options=opts, providers=["CPUExecutionProvider"]
                    )
                    logger.info("[demographics] ONNX genderage.onnx carregado (1 thread)")
                else:
                    self._model_age_ort = None
                    logger.warning("[demographics] genderage.onnx não encontrado — age fallback ativo")

                logger.info("[demographics] ✅ YOLO Dual Engine carregado em %.1fs", time.time() - start)

            else:
                # InsightFace legacy — monkey-patch ONNX threads
                import onnxruntime as _ort
                _orig = _ort.InferenceSession
                def _patched(path, sess_options=None, providers=None, **kw):
                    if sess_options is None:
                        sess_options = _ort.SessionOptions()
                    sess_options.intra_op_num_threads = 1
                    sess_options.inter_op_num_threads = 1
                    return _orig(path, sess_options=sess_options, providers=providers, **kw)
                _ort.InferenceSession = _patched

                from insightface.app import FaceAnalysis
                model_root = os.path.expanduser("~/.insightface")
                self._model = FaceAnalysis(
                    name="buffalo_l", root=model_root,
                    allowed_modules=["detection", "genderage"],
                    providers=["CPUExecutionProvider"],
                )
                self._model.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.3)
                logger.info("[demographics] ✅ InsightFace buffalo_l carregado em %.1fs", time.time() - start)

            self._loaded = True
            return True

        except Exception as e:
            logger.error("[demographics] ❌ Falha ao carregar modelo (%s): %s", self.engine, e)
            self._load_failed = True
            self.enabled = False
            return False

    # ── Batch Worker Loop ─────────────────────────────────────────────────────

    def _batch_worker_loop(self):
        """Thread dedicada: acumula até BATCH_SIZE requisições e processa em lote."""
        logger.info("[demographics] Batch worker loop iniciado")

        while True:
            # Lazy load models if enabled dynamically later
            if self.enabled and not self._loaded and not self._load_failed:
                with self._lock:
                    if not self._loaded:
                        self._load_model()

            batch: List[_InferenceRequest] = []

            # Aguarda pelo menos 1 item
            try:
                first = self._req_queue.get(timeout=5.0)
                batch.append(first)
            except queue.Empty:
                continue

            # Acumula até BATCH_SIZE ou timeout
            deadline = time.time() + BATCH_TIMEOUT
            while len(batch) < BATCH_SIZE:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    req = self._req_queue.get(timeout=min(remaining, 0.05))
                    batch.append(req)
                except queue.Empty:
                    break

            # Processa lote
            self._process_batch(batch)

    def _process_batch(self, batch: List[_InferenceRequest]):
        """Executa inferência em batch e devolve resultados para cada requisição."""
        if not batch:
            return

        t0 = time.time()

        if self.engine == "yolo":
            self._process_batch_yolo(batch)
        else:
            self._process_batch_insightface(batch)

        elapsed = (time.time() - t0) * 1000
        logger.debug("[demographics] Batch %d crops → %.1fms (%.1fms/crop)",
                     len(batch), elapsed, elapsed / len(batch))

        # Notifica threads aguardando
        for req in batch:
            req.event.set()

    def _process_batch_yolo(self, batch: List[_InferenceRequest]):
        face_crops = [r.face_crop for r in batch]

        # Batch gender inference (uma chamada YOLO p/ N crops)
        try:
            gender_results = self._model_gender(face_crops, verbose=False)
        except Exception as e:
            logger.error("[demographics] YOLO batch gender error: %s", e)
            for req in batch:
                req.result = None
            return

        for i, req in enumerate(batch):
            try:
                if getattr(self, "_is_placeholder", False):
                    import hashlib
                    h = int(hashlib.md5(req.track_id.encode()).hexdigest(), 16)
                    result = {
                        "gender": "male" if h % 2 else "female",
                        "age":    _age_to_bucket(20 + h % 30),
                        "age_raw": 20 + h % 30,
                    }
                else:
                    # Parse gender
                    gender = "male"
                    try:
                        probs = gender_results[i].probs
                        if probs is not None and float(probs.data[0].item()) > 0.35:
                            gender = "female"
                    except Exception:
                        pass

                    # Parse age via ONNX (face detectada) ou fallback YOLO
                    age_raw = 38
                    if req.face_found and self._model_age_ort is not None:
                        try:
                            aimg = cv2.resize(req.face_crop, (96, 96))
                            blob = cv2.dnn.blobFromImage(
                                aimg, 1.0/128.0, (96, 96),
                                (127.5, 127.5, 127.5), swapRB=True
                            )
                            pred = self._model_age_ort.run(
                                None, {self._model_age_ort.get_inputs()[0].name: blob}
                            )[0][0]
                            parsed = int(round(pred[2] * 100))
                            if parsed > 2:
                                age_raw = parsed
                        except Exception:
                            pass
                    elif not req.face_found and self._model_age is not None:
                        try:
                            res_age = self._model_age([req.crop], verbose=False)
                            if res_age and res_age[0].boxes is not None and len(res_age[0].boxes) > 0:
                                best = max(res_age[0].boxes, key=lambda b: float(b.conf[0]))
                                if float(best.conf[0]) > 0.40:
                                    cls_name = self._model_age.names[int(best.cls[0].item())].lower()
                                    if "0-14" in cls_name:   age_raw = 10
                                    elif "15-22" in cls_name: age_raw = 20
                                    elif "22" in cls_name:    age_raw = 38
                        except Exception:
                            pass

                    result = {
                        "gender":  gender,
                        "age":     _age_to_bucket(age_raw),
                        "age_raw": age_raw,
                    }
                    self.total_faces_found += 1

                req.result = result
                self.total_analyzed += 1

            except Exception as e:
                logger.error("[demographics] Error processing track %s: %s", req.track_id, e)
                req.result = None

    def _process_batch_insightface(self, batch: List[_InferenceRequest]):
        for req in batch:
            try:
                faces = self._model.get(req.crop)
                if not faces:
                    self.total_no_face += 1
                    req.result = None
                    continue
                self.total_faces_found += 1
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                gender_raw = int(face.gender) if hasattr(face, "gender") else -1
                age_raw    = int(face.age)    if hasattr(face, "age")    else -1
                if gender_raw < 0 or age_raw < 0:
                    req.result = None
                    continue
                req.result = {
                    "gender":  "male" if gender_raw == 1 else "female",
                    "age":     _age_to_bucket(age_raw),
                    "age_raw": age_raw,
                }
                self.total_analyzed += 1
            except Exception as e:
                logger.error("[demographics] InsightFace error for track %s: %s", req.track_id, e)
                req.result = None

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, frame: np.ndarray, track_id: str, bbox: Dict[str, int]) -> Optional[Dict]:
        """
        Analisa demografia de uma pessoa. Enfileira no batch worker e aguarda resultado.
        Cache local evita re-análise dentro de _min_interval_seconds.
        """
        if not self.enabled:
            return None

        # Cache hit
        cached = self._cache.get(track_id)
        if cached and (time.time() - cached["_ts"]) < self._min_interval_seconds:
            return {k: v for k, v in cached.items() if not k.startswith("_")}

        if not self._loaded and not self._load_failed:
            return None  # Worker ainda carregando modelo

        # Extrair crop
        try:
            h, w = frame.shape[:2]
            bw, bh = bbox["width"], bbox["height"]
            px, py = int(bw * 0.2), int(bh * 0.2)
            x1 = max(0, bbox["x"] - px)
            y1 = max(0, bbox["y"] - py)
            x2 = min(w, bbox["x"] + bw + px)
            y2 = min(h, bbox["y"] + bh + py)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                return None
        except Exception:
            return None

        # Haar cascade (face detection — não impacta pool ONNX)
        face_found = False
        face_crop  = crop
        try:
            if hasattr(self, "_face_cascade"):
                gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                faces = self._face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
                )
                if len(faces) > 0:
                    face_found = True
                    (fx, fy, fw, fh) = max(faces, key=lambda r: r[2]*r[3])
                    fpx, fpy = int(fw*0.1), int(fh*0.1)
                    face_crop = crop[
                        max(0, fy-fpy):min(crop.shape[0], fy+fh+fpy),
                        max(0, fx-fpx):min(crop.shape[1], fx+fw+fpx)
                    ]
        except Exception:
            pass

        # Enfileirar requisição no batch worker
        req = _InferenceRequest(crop, track_id, face_crop, face_found)
        try:
            self._req_queue.put_nowait(req)
        except queue.Full:
            logger.warning("[demographics] Queue cheia — descartando análise para track %s", track_id)
            return self._track_best.get(track_id)

        # Aguardar resultado (não bloqueia mais que BATCH_TIMEOUT + margem)
        req.event.wait(timeout=BATCH_TIMEOUT + 1.0)

        if req.result is None:
            return self._track_best.get(track_id)

        # Ensemble voting
        result = req.result
        if track_id not in self._voting_buffer:
            self._voting_buffer[track_id] = []
        self._voting_buffer[track_id].append(result)
        if len(self._voting_buffer[track_id]) > 7:
            self._voting_buffer[track_id].pop(0)

        history = self._voting_buffer[track_id]
        genders  = [h["gender"]  for h in history]
        ages_raw = [h["age_raw"] for h in history]

        consolidated = {
            "gender":  max(set(genders), key=genders.count),
            "age":     _age_to_bucket(int(sum(ages_raw) / len(ages_raw))),
            "age_raw": int(sum(ages_raw) / len(ages_raw)),
        }

        self._cache[track_id]     = {**consolidated, "_ts": time.time()}
        self._track_best[track_id] = consolidated
        self._cleanup_cache()
        return consolidated

    def get_last_known(self, track_id: str) -> Optional[Dict]:
        return self._track_best.get(track_id)

    def _cleanup_cache(self):
        now = time.time()
        if now - self._last_cleanup < 30:
            return
        self._last_cleanup = now
        stale = [k for k, v in self._cache.items() if now - v["_ts"] > 60]
        for k in stale:
            self._cache.pop(k, None)
            self._track_best.pop(k, None)
            self._voting_buffer.pop(k, None)
        if len(self._cache) > self._cache_max:
            oldest = sorted(self._cache.items(), key=lambda x: x[1]["_ts"])
            for k, _ in oldest[:len(self._cache) - self._cache_max]:
                self._cache.pop(k, None)
                self._track_best.pop(k, None)
                self._voting_buffer.pop(k, None)

    def get_stats(self) -> Dict:
        return {
            "enabled":          self.enabled,
            "engine":           self.engine,
            "loaded":           self._loaded,
            "total_analyzed":   self.total_analyzed,
            "total_faces_found":self.total_faces_found,
            "cache_size":       len(self._cache),
            "queue_size":       self._req_queue.qsize(),
            "batch_size":       BATCH_SIZE,
            "architecture":     "singleton+batch",
        }


# ─── Singleton Global ──────────────────────────────────────────────────────────

_singleton: Optional[DemographicsAnalyzer] = None
_singleton_lock = threading.Lock()

def get_analyzer(enabled: bool = True) -> DemographicsAnalyzer:
    """
    Retorna a instância singleton do DemographicsAnalyzer.
    Thread-safe. Todas as câmeras compartilham a mesma instância.
    """
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = DemographicsAnalyzer(enabled=enabled)
                logger.info("[demographics] Singleton criado (pid=%d) enabled=%s", os.getpid(), enabled)
    
    # If a new camera requests demographics but the singleton was initialized as disabled, enable it now.
    if enabled and not _singleton.enabled:
        with _singleton_lock:
            if not _singleton.enabled:
                _singleton.enabled = True
                logger.info("[demographics] Singleton dynamically enabled by a new camera stream")
                # Thread to load the model so we don't block the caller
                if getattr(_singleton, "_worker_thread", None) and _singleton._worker_thread.is_alive():
                     pass # Worker will pick it up
                else:
                    _singleton._worker_thread = threading.Thread(
                        target=_singleton._batch_worker_loop,
                        name="demographics-batch-worker",
                        daemon=True,
                    )
                    _singleton._worker_thread.start()

    return _singleton
