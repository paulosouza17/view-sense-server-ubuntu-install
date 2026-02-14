import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

print("Verifying imports...")
try:
    import fastapi
    print("FastAPI imported.")
    import uvicorn
    print("Uvicorn imported.")
    import ultralytics
    print("Ultralytics imported.")
    import supervision
    print("Supervision imported.")
    import cv2
    print("OpenCV imported.")
    import httpx
    print("HTTPX imported.")
    import yaml
    print("PyYAML imported.")
    import numpy
    print("Numpy imported.")
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
