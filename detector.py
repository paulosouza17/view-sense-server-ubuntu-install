
import time
import logging
import cv2
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime
import supervision as sv
from ultralytics import YOLO
import numpy as np
import asyncio

from api_client import APIClient
from tracker import Tracker
# from line_counter import LineCounter # Deprecated
from line_crossing import LineCrossingDetector, CountingLine

logger = logging.getLogger(__name__)

class CameraDetector(threading.Thread):
    def __init__(self, camera_config: Dict[str, Any], api_client: APIClient):
        super().__init__()
        self.config = camera_config
        self.api_client = api_client
        self.running = False
        self.camera_id = self.config['id']
        self.source = self.config['stream_url']
        # Support for local USB cameras (index 0, 1, etc.)
        if isinstance(self.source, str) and self.source.isdigit():
            self.source = int(self.source)
        
        self.model_name = self.config.get('model', 'yolov8n.pt')
        self.conf_threshold = self.config.get('confidence_threshold', 0.5)
        self.target_classes = self.config.get('classes', [0]) # Default to person
        
        logger.info(f"Initializing CameraDetector for {self.camera_id}...")
        
        # Initialize YOLO
        try:
            self.model = YOLO(self.model_name)
            logger.info(f"YOLO model '{self.model_name}' loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load YOLO model: {e}")
            raise

        # Initialize Tracker
        try:
            fps = self.config.get('fps', 30)
            self.tracker = sv.ByteTrack(frame_rate=fps)
            logger.info(f"ByteTrack initialized with frame_rate={fps}")
        except Exception as e:
            logger.critical(f"Failed to initialize ByteTrack: {e}")
            raise

        # Initialize Line Crossing Detector (New)
        self.crossing_detector = LineCrossingDetector()
        self.counting_lines: List[CountingLine] = []
        
        # State for dynamic resolution handling
        self.raw_rois: List[Dict] = []
        self.current_resolution: Optional[Tuple[int, int]] = None
        
        # Line Counters are now managed by ROI Sync & Dynamic Resolution
        # We skip local config reading here.
        self.roi_id = self.config.get('roi_id')
        
        # Telemetry
        self.fps_monitor = sv.FPSMonitor()
        self.frame_count = 0
        
        # Stream Output
        self.latest_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()

        # Annotators
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()
        self.line_zone_annotator = sv.LineZoneAnnotator()

    def run(self):
        logger.info(f"Starting detection loop for camera {self.camera_id} source={self.source}")
        self.running = True
        
        while self.running:
            try:
                cap = cv2.VideoCapture(self.source)
                if not cap.isOpened():
                    logger.error(f"Failed to open stream for {self.camera_id}. Retrying in 5s...")
                    time.sleep(5)
                    continue
                
                logger.info(f"Successfully connected to stream: {self.source}")
                
                 # Optimization for Live Streams (Low Latency)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Note: For strict timeout handling with FFMPEG backend, 
                # strictly speaking one should set os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;5000"
                # in main.py before loading cv2, but behavior varies by version.
                # The generic reconnection loop below handles 'hard' drops effectively.
                
                while self.running:
                    ret, frame = cap.read()
                    if not ret:
                        # Stream ended or Connection Lost logic
                        # Stream ended or Connection Lost logic
                        # If it's a file (mp4), we seek to 0.
                        # If it's a stream (rtsp/hls), seeking usually doesn't work or throws error.
                        
                        is_local_file = isinstance(self.source, str) and not (self.source.startswith('http') or self.source.startswith('rtsp'))
                        
                        if is_local_file:
                            logger.info(f"File end reached for {self.camera_id}. Looping...")
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            
                            # RESET TRACKING STATE ON LOOP RESTART
                            # This is critical so that objects are re-counted as new people in the next loop.
                            try:
                                fps = self.config.get('fps', 30)
                                self.tracker = sv.ByteTrack(frame_rate=fps)
                                # Preserve existing counting lines but clear crossing state
                                current_lines = self.counting_lines 
                                self.crossing_detector = LineCrossingDetector()
                                # No need to re-add lines, just pass them in update()
                                logger.info("Tracker and Crossing Detector reset for loop.")
                            except Exception as e:
                                logger.error(f"Error resetting tracker on loop: {e}")

                            ret, frame = cap.read()
                        
                        if not ret:
                            # If loop failed OR it is a stream (HLS/RTSP) that ended/broke
                            logger.warning(f"Stream {self.camera_id} ended or failed. Reconnecting in 2s...")
                            break # Break inner loop to re-initialize VideoCapture in outer loop
                    
                    self.frame_count += 1
                    self.fps_monitor.tick()
                    
                    # -------------------------------------------------------------
                    # DYNAMIC RESOLUTION CHECK
                    # Ensure counting lines match current video geometry
                    # -------------------------------------------------------------
                    height, width = frame.shape[:2]
                    if self.current_resolution != (width, height):
                        # Resolution changed or initial frame
                        self._rebuild_lines(width, height)
                    
                    # Inference
                    results = self.model(frame, verbose=False, conf=self.conf_threshold)[0]
                    detections = sv.Detections.from_ultralytics(results)
                    
                    # Filter classes
                    if self.target_classes:
                        # Ensure classes are integers
                        target_classes_int = [int(c) for c in self.target_classes]
                        detections = detections[np.isin(detections.class_id, target_classes_int)]

                    # Update Tracker
                    detections = self.tracker.update_with_detections(detections)
                    
                    # Track Active IDs for cleanup
                    active_track_ids = set()
                    
                    # Annotation Preparation
                    annotated_frame = frame.copy()
                    
                    # ... (Annotation logic omitted for brevity if unchanged, but included below for context) ...
                    # Draw Traces
                    annotated_frame = self.trace_annotator.annotate(
                        scene=annotated_frame, detections=detections
                    )
                    # Draw Boxes
                    annotated_frame = self.box_annotator.annotate(
                        scene=annotated_frame, detections=detections
                    )

                    # Process Detections & Events for API
                    if detections.tracker_id is not None:
                        
                        for i, track_id in enumerate(detections.tracker_id):
                            str_track_id = str(track_id)
                            active_track_ids.add(str_track_id)
                            
                            # Extract BBox
                            x1, y1, x2, y2 = detections.xyxy[i]
                            bbox = {
                                "x": int(x1),
                                "y": int(y1),
                                "width": int(x2 - x1),
                                "height": int(y2 - y1)
                            }
                            
                            # Check Crossings
                            crossings = self.crossing_detector.update(
                                track_id=str_track_id,
                                bbox=bbox,
                                lines=self.counting_lines
                            )
                            
                            # Prepare Payload
                            detection_payload = {
                                "camera_id": self.camera_id,
                                "timestamp": datetime.utcnow().isoformat() + "Z", # UTC ISO8601
                                "detection_class": self.model.names[detections.class_id[i]],
                                "confidence": float(detections.confidence[i]),
                                "track_id": str_track_id,
                                "bounding_box": bbox,
                                "roi_id": None,
                                "crossed_line": False,
                                "direction": None
                            }
                            
                            # If crossed, update payload
                            # Note: A single track might cross multiple lines in one frame (rare but possible).
                            # API usually expects one event per object per frame.
                            # We pick the first crossing event to report, or we need to send multiple.
                            # Assuming 1 event for now.
                            if crossings:
                                crossing = crossings[0]
                                detection_payload.update({
                                    "crossed_line": True,
                                    "roi_id": crossing['roi_id'],
                                    "direction": crossing['direction']
                                })
                                logger.info(f"CROSSING: {str_track_id} -> {crossing['direction']} on {crossing['roi_id']}")
                            
                            # Send to API Client ONLY if line crossed (Optimization for Counting)
                            if self.api_client and detection_payload["crossed_line"]:
                                if not self.api_client.loop or self.api_client.loop.is_closed():
                                    logger.error(f"API Client loop is closed or missing. Cannot send detection.")
                                    continue
                                
                                logger.info(f"Triggering API send for track {str_track_id} (Loop: {self.api_client.loop})")
                                try:
                                    future = asyncio.run_coroutine_threadsafe(
                                        self.api_client.add_detection(detection_payload),
                                        self.api_client.loop
                                    )
                                    # Optional: Add callback to log exceptions from the coroutine
                                    def handle_future_result(f):
                                        try:
                                            f.result()
                                        except Exception as e:
                                            logger.error(f"Error in scheduled API call: {e}")
                                            
                                    future.add_done_callback(handle_future_result)
                                except Exception as e:
                                    logger.error(f"Failed to schedule API call: {e}")
                    
                    # Cleanup stale tracks from crossing detector
                    self.crossing_detector.cleanup_stale_tracks(active_track_ids)

                    # Draw Lines (Visual Aid)
                    # We can use LineZoneAnnotator for visualization if we map CountingLine back to LineZone
                    # Or just draw lines manually using cv2
                    for line in self.counting_lines:
                         cv2.line(annotated_frame, 
                                  (int(line.p1[0]), int(line.p1[1])), 
                                  (int(line.p2[0]), int(line.p2[1])), 
                                  (0, 255, 0), 2)
                         # Draw normal vector to show direction
                         center = ((line.p1[0] + line.p2[0])/2, (line.p1[1] + line.p2[1])/2)
                         # normal is normalized, scale it for display
                         end_n = (center[0] + line._normal[0]*20, center[1] + line._normal[1]*20)
                         cv2.arrowedLine(annotated_frame, 
                                         (int(center[0]), int(center[1])), 
                                         (int(end_n[0]), int(end_n[1])), 
                                         (0, 0, 255), 1)

                    # Update Shared Frame safely
                    with self.lock:
                        self.latest_frame = annotated_frame
                        
                cap.release()
                
            except Exception as e:
                logger.error(f"Error in camera loop {self.camera_id}: {e}")
                time.sleep(5)
    
    def update_settings(self, rois: List[Dict[str, Any]], camera_config: Dict[str, Any]):
        """
        Updates camera settings (Counting Lines, Confidence, Classes) dynamically.
        """
        logger.info(f"Updating settings for camera {self.camera_id}")
        
        # 1. Update Detection Params
        if camera_config:
            new_conf = camera_config.get('confidence_threshold')
            if new_conf:
                self.conf_threshold = float(new_conf)
            
            new_classes = camera_config.get('enabled_classes')
            if new_classes:
                # Map class names to IDs... (same logic as before)
                target_ints = []
                name_to_id = {v: k for k, v in self.model.names.items()}
                for c in new_classes:
                    if isinstance(c, int):
                        target_ints.append(c)
                    elif isinstance(c, str):
                        if c in name_to_id:
                            target_ints.append(name_to_id[c])
                self.target_classes = target_ints
                
        # 2. Store Raw ROIs for Dynamic Resolution Handling
        # We process them only when we know the frame resolution in the run loop.
        with self.lock:
            self.raw_rois = rois
            # Force rebuild lines on next frame check
            self.current_resolution = None 

    def _rebuild_lines(self, width: int, height: int):
        """Rebuilds CountingLine objects with specific resolution."""
        if not self.raw_rois:
            self.counting_lines = []
            return

        logger.info(f"Rebuilding counting lines for resolution {width}x{height}...")
        try:
            new_lines = []
            for roi in self.raw_rois:
                # Assuming roi_sync passes clean dicts
                line = CountingLine.from_roi(roi, width, height)
                if line:
                   new_lines.append(line)
            
            self.counting_lines = new_lines
            self.current_resolution = (width, height)
            logger.info(f"Active counting lines: {len(self.counting_lines)}")
            
        except Exception as e:
            logger.error(f"Failed to rebuild lines: {e}")

    def stop(self):
        self.running = False
        # Do not join explicitly if called from signal handler to avoid deadlock, 
        # but here it's fine.
    
    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def get_status(self):
         return {
             "id": self.camera_id,
             "fps": self.fps_monitor.fps,
             "frames_processed": self.frame_count,
             "status": "running" if self.running else "stopped"
         }
