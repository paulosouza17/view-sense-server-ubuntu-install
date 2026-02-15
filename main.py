from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
import logging
import asyncio
import time
import os

# FFMPEG Timeout Configuration (Crucial for unstable m3u8 streams)
# Must be set before importing cv2 or starting capture
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;10000" # 10s timeout

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import cv2
import threading

from camera_manager import CameraManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

camera_manager = CameraManager("config.yaml")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up ViewSense YOLO Server...")
    # Give the camera manager the current running loop (uvicorn's loop)
    # Actually, we can just let it get the running loop inside start() or pass it here.
    # But start() calls asyncio.get_event_loop() which might work if called here.
    
    # We'll modify camera_manager.start() to strictly use the current loop
    # or just rely on 'asyncio.get_running_loop()'
    
    # We need to run camera_manager.start() which is synchronous (starts threads)
    # but also initializes async api_client.
    
    loop = asyncio.get_running_loop()
    
    # We can't change the signature of start() easily without rewriting camera_manager.
    # camera_manager.start() uses asyncio.get_event_loop() which is deprecated in some contexts
    # but usually works. A better way:
    # We'll set the loop explicitly.
    
    # We'll set the loop explicitly.
    
    camera_manager.api_client_loop = loop 
    
    camera_manager.start()
    
    # Start Observability Loops
    if camera_manager.api_client:
        asyncio.create_task(camera_manager.api_client.heartbeat_loop())
        asyncio.create_task(camera_manager.api_client.roi_sync_loop())
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    await camera_manager.shutdown()

app = FastAPI(title="ViewSense YOLO Server", lifespan=lifespan)

@app.get("/status")
async def status():
    """Detailed status of all cameras."""
    stats = []
    for cam_id, detector in camera_manager.cameras.items():
        stats.append(detector.get_status())
    return {"cameras": stats}

@app.get("/metrics")
async def get_metrics():
    # Simple metrics for now
    return {
        "active_cameras": len(camera_manager.cameras),
        "total_detections_buffered": len(camera_manager.api_client.queue) if camera_manager.api_client else 0
    }

# --- MJPEG STREAMING LOGIC ---
def gen_frames(camera_id: str):
    """Generator for MJPEG stream."""
    detector = camera_manager.cameras.get(camera_id)
    if not detector:
        return
        
    while True:
        if not detector.running:
            break
            
        # Get latest annotated frame
        frame = detector.get_latest_frame()
        
        if frame is not None:
            try:
                # Encode to JPEG (CPU intensive step, only runs if client connected)
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                logger.error(f"Error encoding frame for stream {camera_id}: {e}")
                
        # Control framerate to save bandwidth (max 10fps for debug)
        time.sleep(0.1)

@app.get("/video_feed/{camera_id}")
async def video_feed(camera_id: str):
    """Returns an MJPEG stream for the specified camera."""
    if camera_id not in camera_manager.cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
        
    return StreamingResponse(
        gen_frames(camera_id), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

def generate_mjpeg_stream(camera_id: str):
    if camera_id not in camera_manager.cameras:
        return
    
    detector = camera_manager.cameras[camera_id]
    
    while True:
        frame = detector.get_latest_frame()
        if frame is not None:
            # Encode as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        # Limit to reasonable FPS for streaming (e.g., 20)
        time.sleep(0.05)

@app.get("/video/{camera_id}")
async def video_feed(camera_id: str):
    """
    Streams the processed video (MJPEG) for a specific camera.
    Can be used in <img> tags: <img src="/video/cam_id">
    """
    if camera_id not in camera_manager.cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
        
    return StreamingResponse(
        generate_mjpeg_stream(camera_id), 
        media_type="multipart/x-mixed-replace;boundary=frame"
    )

@app.post("/cameras/{camera_id}/restart")
async def restart_camera(camera_id: str, background_tasks: BackgroundTasks):
    # Restarting might take time, we can do it in background or await if it was async.
    # method restart_camera in manager is synchronous (stops/starts threads).
    # We can run it in threadpool.
    
    # result = await asyncio.to_thread(camera_manager.restart_camera, camera_id)
    # But it modifies shared state (dicts). Thread safety "should" be ok if dict ops are atomic-ish
    # or if we don't care too much about race conditions for this prototype.
    # Better: just call it.
    
    if camera_id not in camera_manager.cameras:
        raise HTTPException(status_code=404, detail="Camera not found")

    camera_manager.restart_camera(camera_id)
    return {"status": "restarted", "camera_id": camera_id}

@app.post("/config/reload")
async def reload_config():
    msg = camera_manager.reload_config()
    return {"message": msg}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
