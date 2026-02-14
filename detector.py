import time
import logging
import cv2
import threading
from typing import Dict, Any, List, Optional
import supervision as sv
from ultralytics import YOLO
import numpy as np
import asyncio

from api_client import APIClient
from tracker import Tracker
from line_counter import LineCounter

logger = logging.getLogger(__name__)

class CameraDetector(threading.Thread):
    def __init__(self, camera_config: Dict[str, Any], api_client: APIClient):
        super().__init__()
        self.config = camera_config
        self.api_client = api_client
        self.running = False
        self.camera_id = self.config['id']
        self.source = self.config['stream_url']
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

        # Initialize Line Counters
        self.line_counters = []
        try:
            for line_conf in self.config.get('counting_lines', []):
                lc = LineCounter(
                    start_point=line_conf['start'],
                    end_point=line_conf['end'],
                    name=line_conf['name']
                )
                self.line_counters.append(lc)
            logger.info(f"Initialized {len(self.line_counters)} line counters.")
        except Exception as e:
            logger.error(f"Error initializing line counters: {e}")
            
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
                
                while self.running:
                    ret, frame = cap.read()
                    if not ret:
                        # Simulation Loop Logic:
                        # If reading fails, it might be end of file (for MP4s).
                        # Try to seek to beginning.
                        logger.info(f"Stream ended for {self.camera_id}. Attempting to loop...")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret:
                            logger.warning("Failed to loop video. Reconnecting stream...")
                            break
                    
                    self.frame_count += 1
                    self.fps_monitor.tick()
                    
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
                    
                    # Annotation Preparation
                    annotated_frame = frame.copy()
                    
                    # Draw Traces
                    annotated_frame = self.trace_annotator.annotate(
                        scene=annotated_frame, detections=detections
                    )
                    
                    # Draw Boxes
                    annotated_frame = self.box_annotator.annotate(
                        scene=annotated_frame, detections=detections
                    )

                    # Draw Labels
                    labels = [
                        f"#{tracker_id} {self.model.names[class_id]} {confidence:0.2f}"
                        for tracker_id, class_id, confidence
                        in zip(detections.tracker_id, detections.class_id, detections.confidence)
                    ]
                    annotated_frame = self.label_annotator.annotate(
                        scene=annotated_frame, detections=detections, labels=labels
                    )

                    # Check Lines
                    events = []
                    for lc in self.line_counters:
                        line_events = lc.trigger(detections)
                        events.extend(line_events)
                        # Draw Line
                        self.line_zone_annotator.annotate(annotated_frame, line_counter=lc.line_zone)
                    
                    # Update Shared Frame safely
                    with self.lock:
                        self.latest_frame = annotated_frame

                    # Process Detections & Events for API
                    if detections.tracker_id is not None:
                        count_crossed = 0
                        
                        for i, track_id in enumerate(detections.tracker_id):
                            crossing_info = next((e for e in events if str(e['track_id']) == str(track_id)), None)
                            
                            detection_payload = {
                                "camera_id": self.camera_id,
                                "detection_class": self.model.names[detections.class_id[i]],
                                "confidence": float(detections.confidence[i]),
                                "track_id": str(track_id),
                                "bounding_box": {
                                    "x": int(detections.xyxy[i][0]),
                                    "y": int(detections.xyxy[i][1]),
                                    "width": int(detections.xyxy[i][2] - detections.xyxy[i][0]),
                                    "height": int(detections.xyxy[i][3] - detections.xyxy[i][1])
                                },
                                "roi_id": self.roi_id,
                                "crossed_line": False
                            }
                            
                            if crossing_info:
                                detection_payload.update({
                                    "crossed_line": True,
                                    "direction": crossing_info['direction']
                                })
                                count_crossed += 1
                            
                            # Send to API Client
                            if self.api_client:
                                asyncio.run_coroutine_threadsafe(
                                    self.api_client.add_detection(detection_payload),
                                    self.api_client.loop
                                )

                cap.release()
                
            except Exception as e:
                logger.error(f"Error in camera loop {self.camera_id}: {e}")
                time.sleep(5)
    
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
