import time
import logging
import numpy as np
import supervision as sv
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class MonitoredZone:
    """Represents a zone for dwell time monitoring."""
    pool_id: str
    name: str
    polygon: np.ndarray # Shape (N, 2)
    # Configuration
    min_dwell_time: float = 1.0 # Minimum seconds to count as specific event?
    
    # Internal State
    # {track_id: start_time}
    current_objects: Dict[str, float] = field(default_factory=dict)
    
    # Supervision Zone
    _sv_zone: sv.PolygonZone = field(init=False, repr=False)
    
    def __post_init__(self):
        # Initialize Supervision PolygonZone
        # triggering_anchors sets where on the box we check (CENTER, BOTTOM_CENTER, etc)
        self._sv_zone = sv.PolygonZone(
            polygon=self.polygon,
            triggering_anchors=(sv.Position.BOTTOM_CENTER,)
        )

class ZoneMonitor:
    """
    Manages multiple zones and tracks dwell time for objects.
    """
    def __init__(self, frame_resolution_wh: Tuple[int, int]):
        self.zones: List[MonitoredZone] = []
        self.resolution_wh = frame_resolution_wh
        
    def add_zone(self, pool_id: str, name: str, polygon_points: List[List[int]]):
        """
        Add a zone to monitor.
        polygon_points: List of [x, y]
        """
        poly = np.array(polygon_points, dtype=np.int32)
        zone = MonitoredZone(pool_id=pool_id, name=name, polygon=poly)
        self.zones.append(zone)
        logger.info(f"Added zone '{name}' with {len(polygon_points)} points.")

    def update(self, detections: sv.Detections) -> List[Dict[str, Any]]:
        """
        Update zones with new detections and return Dwell Time events.
        events: [{
            "type": "zone_exit" | "zone_entry",
            "zone_id": str,
            "track_id": str,
            "dwell_time": float (seconds),
            "timestamp": isoformat
        }]
        """
        events = []
        current_time = time.time()
        
        # We need tracker_id to track dwell time
        if detections.tracker_id is None:
            return []

        for zone in self.zones:
            # Check which detections are inside the zone
            # zone._sv_zone.trigger returns a boolean mask
            is_in_zone = zone._sv_zone.trigger(detections=detections)
            
            # IDs currently inside
            ids_in_zone = detections.tracker_id[is_in_zone]
            str_ids_in_zone = set(str(id) for id in ids_in_zone)
            
            # 1. Handle NEW Entries
            for track_id in str_ids_in_zone:
                if track_id not in zone.current_objects:
                    # New detection in zone
                    zone.current_objects[track_id] = current_time
                    logger.debug(f"Track {track_id} entered zone {zone.name}")
                    # Optional: Emit entry event? 
                    # Usually entry is less interesting than exit/dwell, but we can store it.
            
            # 2. Handle Exits (Objects in 'current_objects' but NOT in 'str_ids_in_zone')
            # BUT: We must be careful about lost tracking. 
            # Ideally, we only consider it an "Exit" if the tracker is still active but outside the zone.
            # However, if detection is lost entirely, we might assume they left or are occluded.
            # For simplicity: If they are not in the zone in this frame, they exited (or disappeared).
            
            # To be robust, we should check if the track_id exists in the current frame's detections *at all*.
            # If it exists in frame but not in zone -> Valid Exit.
            # If it doesn't exist in frame -> Lost Track (maybe still in zone, maybe not).
            
            # Let's get all track IDs present in the current frame
            all_active_track_ids = set(str(id) for id in detections.tracker_id)
            
            detected_exits = []
            for track_id_in_zone, start_time in list(zone.current_objects.items()):
                if track_id_in_zone not in str_ids_in_zone:
                    # It is NOT in the zone anymore.
                    
                    # Check if it is still in the frame (valid exit)
                    # or if it disappeared (signal drift/occlusion)
                    # For simple dwell time, we often count disappearance as exit if we don't have re-id.
                    # But let's assume if it's visible elsewhere, it definitely exited.
                    
                    duration = current_time - start_time
                    
                    if duration >= zone.min_dwell_time:
                         events.append({
                            "type": "zone_dwell",
                            "zone_id": zone.pool_id,
                            "track_id": track_id_in_zone,
                            "dwell_time": round(duration, 2),
                            "action": "exit" # or "summary"
                        })
                         logger.info(f"Dwell Event: {track_id_in_zone} stayed {duration:.1f}s in {zone.name}")
                    
                    del zone.current_objects[track_id_in_zone]

        return events
