#!/usr/bin/env python3
"""ROS 1 Noetic Detection Pipeline Node.

Subscribes:
  - /udp/frame: sensor_msgs/Image - Input frame published by the UDP server
  - /udp/detections: detection_pipeline_ros/DetectionArray - Detections for each frame
  - /livox/lidar: sensor_msgs/PointCloud2
  - /livox/imu: sensor_msgs/Imu

Publishes:
  - /detection/frame: sensor_msgs/Image - Republished input frame
  - /detection/segmentation_masks: sensor_msgs/Image - Binary segmentation mask
  - /detection/best_candidate: geometry_msgs/PointStamped - Best candidate attachment point
"""

import logging

import cv2
import numpy as np
import rospy
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image, PointCloud2, Imu

from detection_pipeline_ros.msg import DetectionArray
from detection_pipeline_ros.pipeline.PipelineContext import PipelineContext
from detection_pipeline_ros.postprocessing.PostProcessor import PostProcessor
from detection_pipeline_ros.postprocessing.geometry.BitmaskSkeleton import BitmaskSkeleton
from detection_pipeline_ros.postprocessing.geometry.DistanceHeatmap import DistanceHeatmap
from detection_pipeline_ros.postprocessing.masks.MaskExtraction import MaskExtraction
from detection_pipeline_ros.postprocessing.scoring.CandidateScoring import CandidateScoring
from detection_pipeline_ros.postprocessing.scoring.CandidateVisualizer import CandidateVisualizer


class DetectionNode:
    def __init__(self):
        rospy.init_node('detection_pipeline_node')

        self.frame_topic = rospy.get_param('~frame_topic', '/udp/frame')
        self.detections_topic = rospy.get_param('~detections_topic', '/udp/detections')
        self.lidar_topic = rospy.get_param('~lidar_topic', '/livox/lidar')
        self.imu_topic = rospy.get_param('~imu_topic', '/livox/imu')

        self.output_frame_topic = rospy.get_param('~output_frame_topic', '/detection/frame')
        self.segmentation_mask_topic = rospy.get_param('~segmentation_mask_topic', '/detection/segmentation_masks')
        self.best_candidate_topic = rospy.get_param('~best_candidate_topic', '/detection/best_candidate')
        self.debug_image_topic = rospy.get_param('~debug_image_topic', '')

        self.sync_queue_size = int(rospy.get_param('~sync_queue_size', 10))
        self.sync_slop = float(rospy.get_param('~sync_slop', 0.05))
        self.sensor_timeout = float(rospy.get_param('~sensor_timeout', 1.0))

        self.bridge = CvBridge()

        self.pub_frame = None
        if self.output_frame_topic:
            self.pub_frame = rospy.Publisher(self.output_frame_topic, Image, queue_size=1)
        self.pub_masks = rospy.Publisher(self.segmentation_mask_topic, Image, queue_size=1)
        self.pub_best = rospy.Publisher(self.best_candidate_topic, PointStamped, queue_size=1)
        self.pub_debug = None
        if self.debug_image_topic:
            self.pub_debug = rospy.Publisher(self.debug_image_topic, Image, queue_size=1)

        stages = [
            MaskExtraction(),
            DistanceHeatmap(),
            BitmaskSkeleton(),
            CandidateScoring(),
        ]
        if self.pub_debug:
            stages.append(CandidateVisualizer())

        self.postprocessor = PostProcessor(stages)

        self.last_lidar_stamp = None
        self.last_imu_stamp = None

        self._setup_subscribers()
        rospy.loginfo("Detection node initialized and ready")

    def _setup_subscribers(self):
        self.lidar_sub = rospy.Subscriber(self.lidar_topic, PointCloud2, self._lidar_callback, queue_size=1)
        self.imu_sub = rospy.Subscriber(self.imu_topic, Imu, self._imu_callback, queue_size=10)

        self.frame_sub = message_filters.Subscriber(self.frame_topic, Image)
        self.detections_sub = message_filters.Subscriber(self.detections_topic, DetectionArray)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.frame_sub, self.detections_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop
        )
        self.sync.registerCallback(self._process_callback)

        rospy.loginfo(f"Subscribing to frame topic: {self.frame_topic}")
        rospy.loginfo(f"Subscribing to detections topic: {self.detections_topic}")
        rospy.loginfo(f"Subscribing to lidar topic: {self.lidar_topic}")
        rospy.loginfo(f"Subscribing to imu topic: {self.imu_topic}")

    def _lidar_callback(self, msg):
        self.last_lidar_stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()

    def _imu_callback(self, msg):
        self.last_imu_stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()

    def _warn_if_sensor_stale(self, stamp):
        if self.last_lidar_stamp is None:
            rospy.logwarn_throttle(5.0, "No /livox/lidar messages received yet")
        else:
            age = (stamp - self.last_lidar_stamp).to_sec()
            if age > self.sensor_timeout:
                rospy.logwarn_throttle(5.0, f"/livox/lidar is stale by {age:.2f}s")

        if self.last_imu_stamp is None:
            rospy.logwarn_throttle(5.0, "No /livox/imu messages received yet")
        else:
            age = (stamp - self.last_imu_stamp).to_sec()
            if age > self.sensor_timeout:
                rospy.logwarn_throttle(5.0, f"/livox/imu is stale by {age:.2f}s")

    def _decode_detection(self, det_msg):
        bbox = np.array(det_msg.bbox, dtype=np.float32) if det_msg.bbox else np.zeros(4, dtype=np.float32)
        mask = None
        if det_msg.mask_height > 0 and det_msg.mask_width > 0 and det_msg.mask:
            expected_len = int(det_msg.mask_height * det_msg.mask_width)
            if len(det_msg.mask) == expected_len:
                mask = np.array(det_msg.mask, dtype=np.uint8).reshape(
                    (int(det_msg.mask_height), int(det_msg.mask_width))
                )
            else:
                rospy.logwarn_throttle(
                    5.0,
                    f"Mask size mismatch: expected {expected_len}, got {len(det_msg.mask)}"
                )

        tracker_id = det_msg.tracker_id if det_msg.has_tracker_id else None
        return bbox, mask, det_msg.confidence, det_msg.class_id, tracker_id, det_msg.data

    def _publish_debug_image(self, ctx, header):
        if not self.pub_debug:
            return

        debug_img = ctx.debug.get("branch_score_image")
        if debug_img is None:
            return

        if debug_img.ndim == 2:
            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)

        try:
            msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
            msg.header = header
            self.pub_debug.publish(msg)
        except CvBridgeError as e:
            rospy.logwarn_throttle(5.0, f"Failed to publish debug image: {e}")

    def _process_callback(self, frame_msg, detections_msg):
        if self.pub_frame:
            self.pub_frame.publish(frame_msg)

        stamp = frame_msg.header.stamp if frame_msg.header.stamp != rospy.Time() else rospy.Time.now()
        self._warn_if_sensor_stale(stamp)

        try:
            frame = self.bridge.imgmsg_to_cv2(frame_msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logwarn_throttle(5.0, f"Failed to decode frame: {e}")
            return

        tracks = []
        masks = []
        for det in detections_msg.detections:
            bbox, mask, confidence, class_id, tracker_id, data = self._decode_detection(det)
            tracks.append((bbox, mask, confidence, class_id, tracker_id, data))
            if mask is not None:
                masks.append(mask)

        ctx = PipelineContext(frame=frame)
        ctx.tracks = tracks
        if masks:
            base_shape = masks[0].shape
            valid_masks = [mask for mask in masks if mask.shape == base_shape]
            if len(valid_masks) != len(masks):
                rospy.logwarn_throttle(5.0, "Skipping masks with mismatched shapes")
            if valid_masks:
                ctx.masks = np.stack(valid_masks)

        ctx = self.postprocessor.process(ctx)

        if ctx.binary_mask is not None:
            mask_msg = self.bridge.cv2_to_imgmsg(ctx.binary_mask, encoding="mono8")
            mask_msg.header = frame_msg.header
            self.pub_masks.publish(mask_msg)

        if ctx.best_candidate:
            best_msg = PointStamped()
            best_msg.header = frame_msg.header
            best_msg.point.x = float(ctx.best_candidate["x"])
            best_msg.point.y = float(ctx.best_candidate["y"])
            best_msg.point.z = float(ctx.best_candidate["score"])
            self.pub_best.publish(best_msg)

        self._publish_debug_image(ctx, frame_msg.header)

    def shutdown_callback(self):
        rospy.loginfo("Detection node shut down")


def main(args=None):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    node = DetectionNode()
    rospy.on_shutdown(node.shutdown_callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    finally:
        node.shutdown_callback()


if __name__ == '__main__':
    main()
