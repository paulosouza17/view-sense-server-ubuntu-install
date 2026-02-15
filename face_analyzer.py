import cv2
import numpy as np
import logging
import os
import urllib.request
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class FaceAnalyzer:
    """
    Analyzes faces for Age and Gender using OpenCV DNN models.
    Lightweight approach suitable for CPU inference.
    """
    
    # Model URLs (Publicly available pre-trained models)
    MODELS_URL = "https://github.com/spmallick/learnopencv/raw/master/AgeGender/"
    
    FILES = {
        "face_proto": "opencv_face_detector.pbtxt",
        "face_model": "opencv_face_detector_uint8.pb",
        "age_proto": "age_deploy.prototxt",
        "age_model": "age_net.caffemodel",
        "gender_proto": "gender_deploy.prototxt",
        "gender_model": "gender_net.caffemodel"
    }
    
    # Categories
    AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-30)', '(38-43)', '(48-53)', '(60-100)']
    GENDERS = ['Male', 'Female']
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.face_net = None
        self.age_net = None
        self.gender_net = None
        
        # Mean values for preprocessing
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    def load_models(self):
        """Checks for models and loads them. Downloads if missing."""
        try:
             # Check and download
            for file_key, file_name in self.FILES.items():
                file_path = os.path.join(self.models_dir, file_name)
                if not os.path.exists(file_path):
                    logger.info(f"Downloading {file_name}...")
                    # URLs vary slightly for the .pb vs .caffemodel
                    # For simplicity, we might need specific reliable URLs.
                    # The learnopencv repo structure:
                    # https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_deploy.prototxt
                    # https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_net.caffemodel
                    # https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_deploy.prototxt
                    # https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_net.caffemodel
                    # https://github.com/spmallick/learnopencv/raw/master/AgeGender/opencv_face_detector.pbtxt
                    # https://github.com/spmallick/learnopencv/raw/master/AgeGender/opencv_face_detector_uint8.pb
                    
                    url = f"{self.MODELS_URL}{file_name}?raw=true"
                    # The .pb file is in a different folder in some repos but let's try direct map or handle exceptions
                    # Actually, better to use specific URLs hardcoded if generic fails.
                    
                    # Fix for opencv_face_detector files which might be in different path
                    # But for now assuming flat structure or I will fix config.
                    
                    try:
                        urllib.request.urlretrieve(url, file_path)
                    except Exception as e:
                        logger.error(f"Failed to download {file_name}: {e}")
            
            # Load Nets
            face_proto = os.path.join(self.models_dir, self.FILES["face_proto"])
            face_model = os.path.join(self.models_dir, self.FILES["face_model"])
            age_proto = os.path.join(self.models_dir, self.FILES["age_proto"])
            age_model = os.path.join(self.models_dir, self.FILES["age_model"])
            gender_proto = os.path.join(self.models_dir, self.FILES["gender_proto"])
            gender_model = os.path.join(self.models_dir, self.FILES["gender_model"])
            
            self.face_net = cv2.dnn.readNet(face_model, face_proto)
            self.age_net = cv2.dnn.readNet(age_model, age_proto)
            self.gender_net = cv2.dnn.readNet(gender_model, gender_proto)
            
            # Set preferable backend (CPU for compatibility)
            for net in [self.face_net, self.age_net, self.gender_net]:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                
            logger.info("Face Analysis models loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Error loading face models: {e}")
            return False

    def detect_and_analyze(self, frame: np.ndarray, person_bbox: Dict[str, int] = None) -> Optional[Dict[str, Any]]:
        """
        Analyzes a person's face. 
        If person_bbox is provided, crops to that area first.
        Returns: {'age': '25-30', 'gender': 'Male', 'confidence': float}
        """
        if self.face_net is None:
            if not self.load_models():
                return None
                
        img_h, img_w = frame.shape[:2]
        
        # Crop to Person BBox if available (optimization)
        roi = frame
        offset_x, offset_y = 0, 0
        
        if person_bbox:
            x = max(0, person_bbox['x'])
            y = max(0, person_bbox['y'])
            w = person_bbox['width']
            h = person_bbox['height']
            
            # Verification
            if w <= 0 or h <= 0 or x >= img_w or y >= img_h:
                return None
                
            roi = frame[y:y+h, x:x+w]
            offset_x, offset_y = x, y
            
        roi_h, roi_w = roi.shape[:2]
        if roi_h < 60 or roi_w < 60: # Too small for face analysis
            return None
            
        # 1. Detect Face in ROI
        blob = cv2.dnn.blobFromImage(roi, 1.0, (300, 300), [104, 117, 123], False, False)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        # Find biggest face
        best_face = None
        max_conf = 0.7 # Threshold
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > max_conf:
                x1 = int(detections[0, 0, i, 3] * roi_w)
                y1 = int(detections[0, 0, i, 4] * roi_h)
                x2 = int(detections[0, 0, i, 5] * roi_w)
                y2 = int(detections[0, 0, i, 6] * roi_h)
                
                # Check dimensions
                fw = x2 - x1
                fh = y2 - y1
                if fw > 0 and fh > 0:
                     if confidence > max_conf: # Pick highest confidence or biggest?
                         # Usually confidence is good enough
                         best_face = (x1, y1, x2, y2)
                         max_conf = confidence
                         
        if not best_face:
            return None
            
        fx1, fy1, fx2, fy2 = best_face
        
        # Padding for better age/gender inference
        pad_w = int((fx2 - fx1) * 0.1)
        pad_h = int((fy2 - fy1) * 0.1)
        
        face_img = roi[max(0, fy1-pad_h):min(roi_h, fy2+pad_h), max(0, fx1-pad_w):min(roi_w, fx2+pad_w)]
        if face_img.size == 0:
            return None
            
        # 2. Analyze Age & Gender
        blob_face = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
        
        # Gender
        self.gender_net.setInput(blob_face)
        gender_preds = self.gender_net.forward()
        gender = self.GENDERS[gender_preds[0].argmax()]
        
        # Age
        self.age_net.setInput(blob_face)
        age_preds = self.age_net.forward()
        age = self.AGE_BUCKETS[age_preds[0].argmax()]
        
        return {
            "age": age,
            "gender": gender,
            "face_box": {
                "x": offset_x + fx1, 
                "y": offset_y + fy1, 
                "w": fx2 - fx1, 
                "h": fy2 - fy1
            }
        }
