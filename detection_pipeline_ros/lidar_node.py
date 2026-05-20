#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs.point_cloud2 as pc2
import numpy as np

import pcl


class CylinderDetector:
    def __init__(self):
        self.sub = rospy.Subscriber("/livox/lidar", PointCloud2, self.lidar_cb, queue_size=1)
        self.pub = rospy.Publisher("lidar/cylinders", MarkerArray, queue_size=1)
        self.marker_id = 0

    def lidar_cb(self, msg):

        pts = np.array(
            [[p[0], p[1], p[2]] for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)],
            dtype=np.float32
        )

        if len(pts) < 50:
            return

        cloud = pcl.PointCloud()
        cloud.from_array(pts)

        sor = cloud.make_voxel_grid_filter()
        sor.set_leaf_size(0.05, 0.05, 0.05)
        cloud = sor.filter()

        seg = cloud.make_segmenter_normals(ksearch=30)
        seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_CYLINDER)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_normal_distance_weight(0.1)
        seg.set_max_iterations(10000)
        seg.set_distance_threshold(0.05)
        seg.set_radius_limits(0.20, 0.5)

        inliers, coeff = seg.segment()

        if len(inliers) == 0:
            return

        rospy.loginfo(
            "Cylinder: x=%.3f y=%.3f z=%.3f ax=%.3f ay=%.3f az=%.3f r=%.3f",
            coeff[0], coeff[1], coeff[2], coeff[3], coeff[4], coeff[5], coeff[6]
        )

        # Create and publish cylinder marker
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = msg.header.frame_id
        marker.header.stamp = msg.header.stamp
        marker.ns = "cylinders"
        marker.id = self.marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        # Position: center of cylinder
        marker.pose.position.x = coeff[0]
        marker.pose.position.y = coeff[1]
        marker.pose.position.z = coeff[2]

        # Orientation: axis direction
        # coeff[3:6] is the axis direction, compute quaternion from it
        ax, ay, az = coeff[3], coeff[4], coeff[5]
        axis = np.array([ax, ay, az])
        axis = axis / np.linalg.norm(axis)  # normalize

        # Default z-axis for cylinder
        z_axis = np.array([0, 0, 1])
        
        # Compute rotation from z_axis to cylinder axis
        rotation_axis = np.cross(z_axis, axis)
        if np.linalg.norm(rotation_axis) > 1e-6:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.arccos(np.dot(z_axis, axis))
            
            # Convert axis-angle to quaternion
            half_angle = angle / 2
            sin_half = np.sin(half_angle)
            marker.pose.orientation.x = rotation_axis[0] * sin_half
            marker.pose.orientation.y = rotation_axis[1] * sin_half
            marker.pose.orientation.z = rotation_axis[2] * sin_half
            marker.pose.orientation.w = np.cos(half_angle)
        else:
            marker.pose.orientation.w = 1.0

        # Scale: radius and height
        marker.scale.x = coeff[6] * 2  # diameter
        marker.scale.y = coeff[6] * 2  # diameter
        marker.scale.z = 1.0  # height (arbitrary, can adjust)

        # Color: red
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8

        marker_array.markers.append(marker)
        self.pub.publish(marker_array)
        self.marker_id += 1


if __name__ == "__main__":
    rospy.init_node("lidar_node")
    CylinderDetector()
    rospy.spin()
