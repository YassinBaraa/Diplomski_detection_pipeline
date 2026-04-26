#!/usr/bin/env python
"""ROS 1 Noetic Detection Pipeline Node

Publishes:
  - /detection/frame: sensor_msgs/Image - Input frame from video/camera
  - /detection/segmentation_masks: sensor_msgs/Image - Binary segmentation masks
  - /detection/best_candidate: geometry_msgs/PointStamped - Best candidate attachment point
"""

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import logging

from detection_pipeline_ros.sources.MP4Source import MP4Source
from detection_pipeline_ros.sources.CameraSource import CameraSource
from detection_pipeline_ros.detectors.YOLOBranchSeg import YOLOBranchSeg
from detection_pipeline_ros.trackers.ByteTrack import ByteTrack
from detection_pipeline_ros.postprocessing.PostProcessor import PostProcessor
from detection_pipeline_ros.postprocessing.masks.MaskExtraction import MaskExtraction
from detection_pipeline_ros.postprocessing.geometry.DistanceHeatmap import DistanceHeatmap
from detection_pipeline_ros.postprocessing.geometry.BitmaskSkeleton import BitmaskSkeleton
from detection_pipeline_ros.postprocessing.scoring.CandidateScoring import CandidateScoring
from detection_pipeline_ros.postprocessing.scoring.CandidateVisualizer import CandidateVisualizer
from detection_pipeline_ros.pipeline.DetectionPipeline import DetectionPipeline
from detection_pipeline_ros.config import ConfigManager

logger = logging.getLogger(__name__)


class DetectionNode:
    def __init__(self):
        rospy.init_node('detection_pipeline_node')

        # Get parameters from ROS parameter server
        config_path = rospy.get_param('~config_path', '')
        source_type = rospy.get_param('~source_type', 'mp4')
        video_path = rospy.get_param('~video_path', '')
        model_path = rospy.get_param('~model_path', '')
        conf_threshold = rospy.get_param('~confidence_threshold', 0.5)

        # Load configuration
        try:
            if config_path:
                self.config = ConfigManager(config_path)
            else:
                self.config = ConfigManager()
            rospy.loginfo(f"Configuration loaded: {self.config}")
        except Exception as e:
            rospy.logerr(f"Failed to load configuration: {e}")
            raise

        # Override config with ROS parameters if provided
        if source_type:
            self.config.config["source"]["type"] = source_type
        if video_path:
            self.config.config["source"]["video_path"] = video_path
        if model_path:
            self.config.config["detector"]["model_path"] = model_path
        if conf_threshold:
            self.config.config["detector"]["confidence_threshold"] = conf_threshold

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Create publishers for the three topics
        self.pub_frame = rospy.Publisher('/detection/frame', Image, queue_size=1)
        self.pub_masks = rospy.Publisher('/detection/segmentation_masks', Image, queue_size=1)
        self.pub_candidate = rospy.Publisher('/detection/best_candidate', PointStamped, queue_size=1)

        rospy.loginfo("Publishers initialized")

        # Initialize pipeline components
        try:
            # Initialize source
            source_config = self.config.get("source.type")
            if source_config == "camera":
                cam_config = self.config.get_section("camera")
                self.source = CameraSource(
                    resolution_width=cam_config.get("resolution_width", 1280),
                    resolution_height=cam_config.get("resolution_height", 720),
                    format=cam_config.get("format", "RGB888")
                )
                rospy.loginfo("Camera source initialized")
            else:  # Default to MP4
                video_path = self.config.get("source.video_path")
                self.source = MP4Source(video_path)
                rospy.loginfo(f"MP4 source initialized: {video_path}")

            # Initialize detector
            detector_config = self.config.get_section("detector")
            self.detector = YOLOBranchSeg(
                model_path=detector_config.get("model_path"),
                conf=detector_config.get("confidence_threshold", 0.5)
            )
            rospy.loginfo("YOLO detector initialized")

            # Initialize tracker
            self.tracker = ByteTrack(self.detector)
            rospy.loginfo("ByteTrack tracker initialized")

            # Initialize postprocessor
            self.postprocessor = PostProcessor([
                MaskExtraction(),
                DistanceHeatmap(),
                BitmaskSkeleton(),
                CandidateScoring(),
                CandidateVisualizer(),
            ])
            rospy.loginfo("PostProcessor initialized")

            # Create pipeline
            self.pipeline = DetectionPipeline(
                source=self.source,
                detector=self.detector,
                tracker=self.tracker,
                postprocessor=self.postprocessor
            )
            rospy.loginfo("Detection pipeline created")

        except Exception as e:
            rospy.logerr(f"Failed to initialize pipeline: {e}", exc_info=True)
            raise

        # Create timer for processing frames
        self.frame_count = 0
        self.timer = rospy.Timer(rospy.Duration(0.01), self.process_frame_callback)  # ~100Hz
        rospy.loginfo("Detection node initialized and ready")

    def process_frame_callback(self, event):
        """Process one frame from pipeline and publish results"""
        try:
            # Get next frame from pipeline generator
            ctx = next(self.pipeline.run())
            self.frame_count += 1

            # Get current timestamp
            timestamp = rospy.Time.now()

            # Publish frame
            if ctx.frame is not None:
                frame_msg = self.bridge.cv2_to_imgmsg(ctx.frame, encoding="bgr8")
                frame_msg.header.stamp = timestamp
                frame_msg.header.frame_id = "detection_frame"
                self.pub_frame.publish(frame_msg)

            # Publish segmentation masks
            if ctx.binary_mask is not None:
                mask_msg = self.bridge.cv2_to_imgmsg(ctx.binary_mask, encoding="mono8")
                mask_msg.header.stamp = timestamp
                mask_msg.header.frame_id = "segmentation_masks"
                self.pub_masks.publish(mask_msg)

            # Publish best candidate point
            if ctx.best_candidate is not None:
                candidate = ctx.best_candidate
                point_msg = PointStamped()
                point_msg.header.stamp = timestamp
                point_msg.header.frame_id = "best_candidate"
                point_msg.point.x = float(candidate.get("x", 0))
                point_msg.point.y = float(candidate.get("y", 0))
                point_msg.point.z = float(candidate.get("score", 0))  # Store score in z
                self.pub_candidate.publish(point_msg)

                # Log best candidate info
                if self.frame_count % 30 == 0:
                    rospy.loginfo(
                        f"Best candidate: ({candidate['x']}, {candidate['y']}) "
                        f"score={candidate['score']:.2f} stable={candidate.get('is_stable', False)}"
                    )

            if self.frame_count % 100 == 0:
                rospy.logdebug(f"Processed {self.frame_count} frames")

        except StopIteration:
            rospy.loginfo(f"Pipeline complete. Processed {self.frame_count} frames total")
            self.source.release()
            rospy.signal_shutdown("Pipeline complete")
        except Exception as e:
            rospy.logerr(f"Error processing frame: {e}", exc_info=True)

    def shutdown_callback(self):
        """Cleanup on shutdown"""
        try:
            self.source.release()
        except:
            pass
        rospy.loginfo("Detection node shut down")


def main(args=None):
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    node = DetectionNode()

    # Register shutdown hook
    rospy.on_shutdown(node.shutdown_callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    finally:
        node.shutdown_callback()


if __name__ == '__main__':
    main()
