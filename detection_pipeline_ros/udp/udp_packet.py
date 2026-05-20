import json
import struct
import numpy as np
from datetime import datetime


class DetectionPacket:
    
    def __init__(self, frame, detections, timestamp=None):
        """
        Args:
            frame: numpy array of image data
            detections: list of tuples (xyxy, mask, confidence, class_id, tracker_id, data)
            timestamp: float (unix timestamp) or None for current time
        """
        self.frame = frame
        self.detections = detections
        self.timestamp = timestamp or datetime.now().timestamp()
    
    def to_bytes(self):
        # Serialize detections as JSON (excluding masks which are too large for UDP)
        detections_list = []
        for xyxy, mask, confidence, class_id, tracker_id, data in self.detections:
            det = {
                'xyxy': xyxy.tolist() if isinstance(xyxy, np.ndarray) else xyxy,
                'confidence': float(confidence),
                'class_id': int(class_id),
                'tracker_id': int(tracker_id) if tracker_id is not None else None,
            }
            detections_list.append(det)
        
        packet_data = {
            'detections': detections_list,
            'num_detections': len(detections_list)
        }
        
        packet_json = json.dumps(packet_data).encode('utf-8')
        
        # Pack header: detections size (4 bytes) + timestamp (8 bytes, double)
        detections_size = len(packet_json)
        header = struct.pack('<Id', detections_size, self.timestamp)
        
        return header + packet_json
    
    @staticmethod
    def from_bytes(data):
        """Deserialize from bytes received via UDP."""
        if len(data) < 12:
            raise ValueError("Invalid packet: too short")
        
        # Unpack header (detections size + timestamp)
        detections_size, timestamp = struct.unpack('<Id', data[:12])
        
        # Extract packet JSON
        packet_json = data[12:12 + detections_size]
        
        # Decode packet
        packet_data = json.loads(packet_json.decode('utf-8'))
        detections_list = packet_data['detections']
        
        detections = []
        for det in detections_list:
            xyxy = np.array(det['xyxy']) if det.get('xyxy') else np.array([])
            detections.append((
                xyxy,
                None,  # masks removed
                det['confidence'],
                det['class_id'],
                det['tracker_id'],
                None  # data removed
            ))
        
        return DetectionPacket(None, detections, timestamp)
    
    def get_frame_info(self):
        """Get metadata about the packet."""
        return {
            'timestamp': self.timestamp,
            'num_detections': len(self.detections)
        }
