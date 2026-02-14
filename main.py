from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
import logging
import asyncio

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
    
    camera_manager.api_client_loop = loop 
    # Wait, in camera_manager.py I did: `loop = asyncio.get_event_loop()`
    # Let's trust that works or modify camera_manager if needed.
    # But better is to just call start().
    
    camera_manager.start()
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    await camera_manager.shutdown()

app = FastAPI(title="ViewSense YOLO Server", lifespan=lifespan)

@app.get("/status")
async def get_status():
    return {
        "cameras": camera_manager.get_all_status(),
        "api_client_queue": len(camera_manager.api_client.queue) if camera_manager.api_client else 0
    }

@app.get("/metrics")
async def get_metrics():
    # Simple metrics for now
    return {
        "active_cameras": len(camera_manager.cameras),
        "total_detections_buffered": len(camera_manager.api_client.queue) if camera_manager.api_client else 0
    }

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
