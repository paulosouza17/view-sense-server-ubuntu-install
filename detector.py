import time
import logging
import cv2
import threading
from typing import Dict, Any, List
import supervision as sv
from ultralytics import YOLO

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
        
        # Initialize YOLO
        self.model = YOLO(self.model_name)
        
        # Initialize Tracker
        self.tracker = sv.ByteTrack(frame_rate=self.config.get('fps', 30))
        
        # Initialize Line Counters
        self.line_counters = []
        for line_conf in self.config.get('counting_lines', []):
            lc = LineCounter(
                start_point=line_conf['start'],
                end_point=line_conf['end'],
                name=line_conf['name']
            )
            self.line_counters.append(lc)
            
        self.roi_id = self.config.get('roi_id')

        # Telemetry
        self.fps_monitor = sv.FPSMonitor()
        self.frame_count = 0
        self.last_frame = None

    def run(self):
        logger.info(f"Starting detection loop for camera {self.camera_id}")
        self.running = True
        
        while self.running:
            try:
                cap = cv2.VideoCapture(self.source)
                if not cap.isOpened():
                    logger.error(f"Failed to open stream for {self.camera_id}. Retrying in 5s...")
                    time.sleep(5)
                    continue
                
                logger.info(f"Connected to stream: {self.source}")
                
                while self.running and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning(f"Failed to read frame from {self.camera_id}. Reconnecting...")
                        break
                    
                    self.frame_count += 1
                    self.fps_monitor.tick()
                    self.last_frame = frame
                    
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
                    
                    # Check Lines
                    events = []
                    for lc in self.line_counters:
                        line_events = lc.trigger(detections)
                        events.extend(line_events)
                    
                    # Process Detections & Events
                    # We can send all detections or only those that crossed.
                    # The prompt implies: "Cada detecção que CRUZA a linha deve ter crossed_line=true..."
                    # But also "Envia detecções para a API REST".
                    # Usually we send periodic updates of all tracks OR only events.
                    # Prompt says: "Envia detecções para a API... Payload conforme doc... crossed_line=true"
                    # It seems we might want to send ALL active tracks, marking those that crossed.
                    
                    # Let's collect all current tracks
                    if detections.tracker_id is not None:
                        count_crossed = 0
                        
                        for i, track_id in enumerate(detections.tracker_id):
                            # Check if this track crossed any line in this frame
                            # The 'events' list contains dicts with track_id as STRING
                            
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
                                # Priority send? 
                                # We add to api client buffer
                            
                            # Decision: Do we send ALL detections every frame? 
                            # Or only crossed? 
                            # prompt says "Counts line crossings... Sends detections". 
                            # "Envio para API ... Acumular detecções em buffer"
                            # If we send 30fps * objects, it's a lot. 
                            # Usually we sample, or send events.
                            # Prompt "Batch size 20" and "send_interval_seconds 2". 
                            # If we send every frame, we act very fast. 
                            # I will send ALL detections that are tracked, but the API client handles batching.
                            # To save bandwidth/resources, maybe we only send if crossed_line or every N frames?
                            # Prompt "Detecta pessoas... Rastreia... Conta... Envia". 
                            # Let's Assume we send all tracked objects to visualization.
                            
                            # Optimization: only send if crossed OR if 1 second passed for this track?
                            # For now, simply send everything to the buffer.
                            # Exception: If multiple detections, this fills up fast. 
                            # Let's stick to sending all tracked objects.
                            
                            # Using asyncio.run_coroutine_threadsafe to add to async queue from this thread
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
        self.join()

    def get_status(self):
         return {
             "id": self.camera_id,
             "fps": self.fps_monitor.fps,
             "frames_processed": self.frame_count,
             "status": "running" if self.running else "stopped"
         }

import numpy as np
import asyncio
