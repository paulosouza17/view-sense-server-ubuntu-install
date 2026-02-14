from typing import Tuple, List, Dict
import numpy as np
import supervision as sv

class LineCounter:
    def __init__(self, start_point: Tuple[int, int], end_point: Tuple[int, int], name: str = "Line"):
        self.name = name
        # supervision LineZone expects points as sv.Point(x, y)
        self.start = sv.Point(*start_point)
        self.end = sv.Point(*end_point)
        self.line_zone = sv.LineZone(start=self.start, end=self.end)
        
    def trigger(self, detections: sv.Detections) -> List[Dict[str, Any]]:
        """
        Updates the line zone with new detections and returns a list of events
        for any track that crossed the line in this frame.
        """
        # line_zone.trigger returns (crossed_in, crossed_out) boolean arrays
        crossed_in, crossed_out = self.line_zone.trigger(detections)
        
        events = []
        
        # detections.tracker_id should be present
        if detections.tracker_id is None:
            return events

        for i, track_id in enumerate(detections.tracker_id):
            direction = None
            if crossed_in[i]:
                direction = "in"
            elif crossed_out[i]:
                direction = "out"
            
            if direction:
                events.append({
                    "track_id": str(track_id),
                    "crossed_line": True,
                    "direction": direction,
                    "line_name": self.name
                })
                
        return events
