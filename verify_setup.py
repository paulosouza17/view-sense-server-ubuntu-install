import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

print("Verifying imports...")
try:
    import fastapi
    print("FastAPI imported.")
    import uvicorn
    print(f"Uvicorn: {uvicorn.__version__}")
    import ultralytics
    print(f"Ultralytics: {ultralytics.__version__}")
    import supervision
    print(f"Supervision: {getattr(supervision, '__version__', 'unknown')}")
    import cv2
    print(f"OpenCV: {cv2.__version__}")
    import httpx
    print(f"HTTPX: {httpx.__version__}")
    import yaml
    print("PyYAML imported.")
    import numpy
    print(f"Numpy: {numpy.__version__}")

    print("\n[TEST] Testing object instantiation to ensure compatibility...")
    
    # Test YOLO init
    print("  -> Testing ultralytics.YOLO()...")
    from ultralytics import YOLO
    try:
        # Just init, don't load heavy weights if possible, but 'yolov8n.pt' is smallest
        # We'll rely on it being present or downloaded.
        yolo = YOLO('yolov8n.pt') 
        print("     [OK] YOLO initialized.")
    except Exception as e:
        print(f"     [FAIL] YOLO init failed: {e}")
        sys.exit(1)

    # Test Supervision imports
    print("  -> Testing supervision.Detections...")
    try:
        # This was the error before: 'module' object has no attribute 'Detections'
        det = supervision.Detections.empty()
        print("     [OK] supervision.Detections found.")
    except AttributeError as e:
        print(f"     [FAIL] supervision.Detections missing! Check version. Error: {e}")
        sys.exit(1)

    print("  -> Testing supervision.ByteTrack...")
    try:
        tracker = supervision.ByteTrack()
        print("     [OK] supervision.ByteTrack initialized.")
    except Exception as e:
        print(f"     [FAIL] ByteTrack init failed: {e}")
        sys.exit(1)
        
    print("\n[SUCCESS] Deep verification passed!")
except ImportError as e:
    print(f"Import failed: {e}")
    # We expect some to fail if not installed in the user's current environment
    # But this script serves as a check for the user to run.

print("\nVerifying project modules...")
try:
    from api_client import APIClient
    print("api_client module OK.")
    from line_counter import LineCounter
    print("line_counter module OK.")
    from tracker import Tracker
    print("tracker module OK.")
    from detector import CameraDetector
    print("detector module OK.")
    from camera_manager import CameraManager
    print("camera_manager module OK.")
    from main import app
    print("main module OK.")
except Exception as e:
    print(f"Module verification failed: {e}")
    sys.exit(1)

print("\nVerifying config loading...")
try:
    cm = CameraManager("config.yaml")
    print("Config loaded successfully.")
    print(f"Cameras found: {len(cm.config['cameras'])}")
except Exception as e:
    print(f"Config verification failed: {e}")
    sys.exit(1)

print("\nVerification passed!")
