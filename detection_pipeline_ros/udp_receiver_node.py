#!/usr/bin/env python3
import json
import logging
import socket

import numpy as np
import rospy
from sensor_msgs.msg import Image

from detection_pipeline_ros.msg import Detection, DetectionArray
from detection_pipeline_ros.udp_packet import DetectionPacket


class UdpDetectionReceiver:
    def __init__(self):
        rospy.init_node('udp_detection_receiver')

        self.bind_ip = rospy.get_param('~bind_ip', '0.0.0.0')
        self.bind_port = int(rospy.get_param('~bind_port', 5005))
        self.detections_topic = rospy.get_param('~detections_topic', '/udp/detections')
        self.frame_id = rospy.get_param('~frame_id', 'udp_frame')
        self.max_packet_size = int(rospy.get_param('~max_packet_size', 65507))
        self.socket_timeout = float(rospy.get_param('~socket_timeout', 0.5))

        self.pub_detections = rospy.Publisher(self.detections_topic, DetectionArray, queue_size=1)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.bind_ip, self.bind_port))
        self.sock.settimeout(self.socket_timeout)

        rospy.loginfo(f"UDP receiver bound to {self.bind_ip}:{self.bind_port}")
        rospy.loginfo(f"Publishing detections topic: {self.detections_topic}")

    def _mask_to_fields(self, mask):
        if mask is None:
            return 0, 0, []

        mask_arr = np.array(mask, dtype=np.uint8)
        if mask_arr.size == 0:
            return 0, 0, []

        if mask_arr.ndim >= 2:
            height, width = mask_arr.shape[:2]
        else:
            height, width = mask_arr.size, 1

        return int(height), int(width), mask_arr.flatten().tolist()

    def _data_to_string(self, data):
        if data is None:
            return ""
        if isinstance(data, str):
            return data
        return json.dumps(data)

    def _to_detection_msg(self, det):
        xyxy, mask, confidence, class_id, tracker_id, data = det

        bbox = [0.0, 0.0, 0.0, 0.0]
        if xyxy is not None:
            xyxy_list = xyxy.tolist() if isinstance(xyxy, np.ndarray) else list(xyxy)
            if len(xyxy_list) >= 4:
                bbox = [float(x) for x in xyxy_list[:4]]

        mask_height, mask_width, mask_flat = self._mask_to_fields(mask)

        msg = Detection()
        msg.bbox = bbox
        msg.confidence = float(confidence) if confidence is not None else 0.0
        msg.class_id = int(class_id) if class_id is not None else -1
        msg.has_tracker_id = tracker_id is not None
        msg.tracker_id = int(tracker_id) if tracker_id is not None else -1
        msg.mask_height = mask_height
        msg.mask_width = mask_width
        msg.mask = mask_flat
        msg.data = self._data_to_string(data)
        return msg

    def spin(self):
        while not rospy.is_shutdown():
            try:
                data, _ = self.sock.recvfrom(self.max_packet_size)
            except socket.timeout:
                continue
            except Exception as e:
                rospy.logerr(f"Socket error: {e}")
                continue

            try:
                packet = DetectionPacket.from_bytes(data)
                print(f"[UDP] Received packet with {len(packet.detections)} detections at {packet.timestamp}")
            except Exception as e:
                rospy.logwarn(f"Failed to decode packet: {e}")
                continue

            stamp = rospy.Time.from_sec(packet.timestamp)

            detections_msg = DetectionArray()
            detections_msg.header.stamp = stamp
            detections_msg.header.frame_id = self.frame_id
            detections_msg.detections = [self._to_detection_msg(d) for d in packet.detections]
            self.pub_detections.publish(detections_msg)

    def shutdown_callback(self):
        try:
            self.sock.close()
        except Exception:
            pass


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    node = UdpDetectionReceiver()
    rospy.on_shutdown(node.shutdown_callback)

    try:
        node.spin()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
