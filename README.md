# Detection Pipeline ROS 1 Noetic Package

A ROS 1 Noetic implementation of the UAV branch detection pipeline for autonomous landing systems.

## Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/detection/frame` | `sensor_msgs/Image` | Input BGR frame (640x480 or source resolution) |
| `/detection/segmentation_masks` | `sensor_msgs/Image` | Binary segmentation mask (mono8) |
| `/detection/best_candidate` | `geometry_msgs/PointStamped` | Best attachment point candidate |

**Best Candidate Point Details** (in `PointStamped.point`):
- `x`: Pixel x-coordinate
- `y`: Pixel y-coordinate
- `z`: Score value (attachment point quality metric)

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

## Configuration

Configuration is defined in `detection_pipeline_ros/config/default_config.yaml` with these sections:

```yaml
source:
  type: "mp4"              # "mp4" or "camera"
  video_path: "..."        # Path to video file

detector:
  model_path: "..."        # Path to YOLO .pt model
  confidence_threshold: 0.5

visualization:
  enabled: true
  resize_width: 640
  resize_height: 384
```

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
  _source_type:=mp4 \
  _video_path:=/path/to/video.mp4 \
  _model_path:=/path/to/model.pt \
  _confidence_threshold:=0.5
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

## Model Files

Place your YOLO segmentation model in the `model/` directory:

```bash
cp /path/to/best_small.pt ./model/
```

The model path in config should be relative or absolute. For ROS deployments, use:
```yaml
detector:
  model_path: "${ROS_PACKAGE_PATH}/detection_pipeline_ros/model/best_small.pt"
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


