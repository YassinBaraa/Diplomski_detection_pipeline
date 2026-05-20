# Detection Pipeline ROS 1 Noetic Package

A ROS 1 Noetic implementation of the UAV branch detection pipeline for autonomous landing systems.

## Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/detection/frame` | `sensor_msgs/Image` | Republished BGR frame from `/udp/frame` |
| `/detection/segmentation_masks` | `sensor_msgs/Image` | Binary segmentation mask (mono8) |
| `/detection/best_candidate` | `geometry_msgs/PointStamped` | Best attachment point candidate |

**Best Candidate Point Details** (in `PointStamped.point`):
- `x`: Pixel x-coordinate
- `y`: Pixel y-coordinate
- `z`: Score value (attachment point quality metric)

## Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/udp/frame` | `sensor_msgs/Image` | Frame published by the UDP receiver/server |
| `/udp/detections` | `detection_pipeline_ros/DetectionArray` | Detections for each frame |
| `/livox/lidar` | `sensor_msgs/PointCloud2` | Livox lidar point cloud |
| `/livox/imu` | `sensor_msgs/Imu` | Livox IMU data |

## Building the Package

```bash
# Navigate to ROS workspace
cd ~/catkin_ws/src

# Clone/link the package (adjust path as needed)
ln -s /home/baraa/Desktop/Diplomski/Diplomski_rad/detection_pipeline_ros .

# Return to workspace root and build
cd ~/catkin_ws
catkin_make
# or use catkin build if installed:
# catkin build detection_pipeline_ros
source devel/setup.bash
```

## Installation of Python Dependencies

The package uses these external dependencies:
- `opencv-python` >= 4.5.0
- `ultralytics` >= 8.0.0
- `supervision` >= 0.3.0
- `numpy` >= 1.21.0
- `PyYAML` >= 5.4

Install them in your ROS container:

```bash
pip install opencv-python ultralytics supervision numpy PyYAML
```

## PCL (required for `lidar_node.py`)

The LiDAR node depends on the Python PCL bindings (`import pcl`). Install the Ubuntu package:

```bash
sudo apt-get update
sudo apt-get install -y python3-pcl
```

## Configuration

The detection node consumes `/udp/frame` and `/udp/detections`. The legacy YAML config for local video/model sources is not used by the current node.

## Running the Node

### Method 1: Using Launch File

```bash
roslaunch detection_pipeline_ros detection_pipeline.launch
```

### Method 2: Direct Execution

```bash
rosrun detection_pipeline_ros detection_node.py
```

### Method 3: With Parameter Overrides

```bash
rosrun detection_pipeline_ros detection_node.py \
  _frame_topic:=/udp/frame \
  _detections_topic:=/udp/detections \
  _lidar_topic:=/livox/lidar \
  _imu_topic:=/livox/imu
```

## UDP Receiver Node

Listens for UDP `DetectionPacket` frames and republishes them as ROS topics.

### Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/udp/frame` | `sensor_msgs/Image` | JPEG-decoded BGR frame |
| `/udp/detections` | `detection_pipeline_ros/DetectionArray` | Decoded detection list |

**Detection fields** (`detection_pipeline_ros/Detection`):
`bbox [x1,y1,x2,y2]`, `confidence`, `class_id`, `has_tracker_id`, `tracker_id`,
`mask_height`, `mask_width`, `mask` (flattened `uint8`), `data` (JSON string).

### Run

```bash
roslaunch detection_pipeline_ros udp_receiver.launch
```

## Monitoring Output

In another terminal, subscribe to topics:

```bash
# Watch frame output
rostopic echo /detection/frame

# Watch best candidate points
rostopic echo /detection/best_candidate

# Check message rate
rostopic hz /detection/frame
```

## Architecture

The pipeline consists of:
1. **Source**: MP4 file or camera input
2. **Detection**: YOLO instance segmentation
3. **Tracking**: ByteTrack multi-object tracking
4. **Post-processing**:
   - Mask extraction (binary segmentation)
   - Distance heatmap (centerline detection)
   - Skeleton extraction (medial axis)
   - Candidate scoring (geometric + temporal stability)
   - Visualization (debug output)

All stages run sequentially in a single ROS 1 node.

## Troubleshooting

**cv2.imshow() errors**: The pipeline doesn't display windows in ROS (removed from MaskExtraction). Use ROS visualization tools instead.

**Model not found**: Ensure model path is correct relative to current working directory or provide absolute path.

**Import errors**: Check that all python dependencies are installed and package is sourced:
```bash
source ~/catkin_ws/devel/setup.bash
```

**Slow processing**: Monitor GPU usage; ensure YOLO model is appropriate for hardware.

## Future Improvements

- Multi-node architecture (detection, tracking, post-processing as separate nodes)
- Configurable message queuing and frame dropping
- ROS service for model switching
- Performance metrics publisher
- Integration with ROS 1 camera drivers
