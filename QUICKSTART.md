# Quick Start Guide - Detection Pipeline ROS 1 Noetic

## Prerequisites
- ROS 1 Noetic
- Python 2.7 or 3.x (Noetic supports both)
- Your existing container with a ROS 1 workspace
- catkin build tools

## Step 1: Add Package to Container

```bash
# Navigate to ROS workspace src directory
cd ~/catkin_ws/src

# Clone or link the package
git clone <repo> detection_pipeline_ros
# OR
ln -s /path/to/detection_pipeline_ros .
```

## Step 2: Install Python Dependencies

```bash
pip install -r src/detection_pipeline_ros/requirements.txt
```

**Required packages:**
- opencv-python (image processing)
- ultralytics (YOLO)
- supervision (tracking)
- numpy
- PyYAML (config)

## Step 3: Build Package

```bash
cd ~/catkin_ws
catkin_make
# or if using catkin build:
catkin build detection_pipeline_ros
source devel/setup.bash
```

## Step 4: Run the Pipeline

### Option A: Using Launch File
```bash
roslaunch detection_pipeline_ros detection_pipeline.launch
```

### Option B: Direct Execution
```bash
rosrun detection_pipeline_ros detection_node.py \
  _frame_topic:=/udp/frame \
  _detections_topic:=/udp/detections \
  _lidar_topic:=/livox/lidar \
  _imu_topic:=/livox/imu
```

## Step 5: Monitor Output

Open new terminal(s):

```bash
# Start ROS master if not already running
roscore

# List available topics
rostopic list

# View frame data
rostopic echo /detection/frame

# View segmentation masks
rostopic echo /detection/segmentation_masks

# View best candidate points
rostopic echo /detection/best_candidate

# Check frame rate
rostopic hz /detection/frame
```

## Topics Published

| Topic | Type | Frequency |
|-------|------|-----------|
| `/detection/frame` | sensor_msgs/Image | ~30-60 Hz (depends on video) |
| `/detection/segmentation_masks` | sensor_msgs/Image | Same as frame |
| `/detection/best_candidate` | geometry_msgs/PointStamped | Same as frame |

## Topics Subscribed

| Topic | Type | Frequency |
|-------|------|-----------|
| `/udp/frame` | sensor_msgs/Image | Depends on server |
| `/udp/detections` | detection_pipeline_ros/DetectionArray | Depends on server |
| `/livox/lidar` | sensor_msgs/PointCloud2 | Sensor rate |
| `/livox/imu` | sensor_msgs/Imu | Sensor rate |

## Common Issues

**Problem**: "No module named 'detection_pipeline_ros'"
```bash
cd ~/catkin_ws
source devel/setup.bash
```

**Problem**: Low FPS
- Check GPU availability: `nvidia-smi`
- Use smaller YOLO model variant
- Reduce input frame resolution

**Problem**: cv2.imshow() not working
- This is expected in ROS (no display server)
- Use ROS visualization tools instead

## Running in Docker

```bash
cd detection_pipeline_ros

# Build image
docker build -f Dockerfile.example -t detection_pipeline_ros:latest .

# Run container
docker run -it --gpus all \
  -v /path/to/videos:/videos \
  -v /path/to/models:/models \
  detection_pipeline_ros:latest \
  roslaunch detection_pipeline_ros detection_pipeline.launch \
  video_path:=/videos/example.mp4 \
  model_path:=/models/best_small.pt
```

## What's Different from Original?

✅ **Advantages:**
- ROS 1 Noetic integration with publishers
- Timestamped messages with ROS Time
- Easy monitoring with ROS tools (rostopic, rosbag)
- Network-transparent (can be on different machines)
- Better logging and diagnostics

⚠️ **Changes:**
- All imports are absolute (e.g., `from detection_pipeline_ros.module import X`)
- Removed `cv2.imshow()` (use ROS visualization)
- Frame processing is asynchronous (timer-based)
- Configuration can be overridden via ROS parameters (private params with `_` prefix)

## Next Steps

1. Subscribe to topics from your ROS node
2. Use `best_candidate` data for your autonomous landing system
3. Optionally extend with ROS Services or additional Publishers
4. Deploy to embedded systems with ROS 1 support (RPi running Noetic, etc.)

## Support

Refer to original detection_pipeline documentation for algorithm details.
See README.md for full documentation.
