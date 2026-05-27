# Detection Pipeline for UAV Branch Hook Attachment

## Overview

This project implements a real-time computer vision pipeline for automated branch detection and hook attachment point estimation for UAV autonomous landing systems. It combines YOLO instance segmentation, multi-object tracking, and geometric analysis to identify suitable attachment points on tree branches.

**Project Type**: Diplomski rad (Master's Thesis)
**Status**: Beta / Research Prototype
**Target Platform**: Embedded systems (Raspberry Pi, UAV edge devices)

---

## System Architecture

### Pipeline Stages

The system processes video frames through a series of specialized stages:

```
Input Frame
    ↓
┌─────────────────────────────────────────────┐
│ Detection Layer (YOLOBranchSeg/HailoSeg)    │
│ - YOLO or Hailo instance segmentation       │
│ - Outputs: bboxes, masks, confidence scores │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ Tracking Layer (ByteTrack)                  │
│ - Multi-object tracking across frames       │
│ - Assigns persistent track IDs              │
└─────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────┐
│ Post-Processing Pipeline                        │
├──────────────────────────────────────────────────┤
│ 1. MaskExtraction                              │
│    - Merge segmentation masks → binary image   │
├──────────────────────────────────────────────────┤
│ 2. DistanceHeatmap                             │
│    - Euclidean distance transform              │
│    - Identifies branch centerlines             │
├──────────────────────────────────────────────────┤
│ 3. BitmaskSkeleton                             │
│    - Morphological skeletonization             │
│    - 1-pixel-wide medial axis                  │
├──────────────────────────────────────────────────┤
│ 4. CandidateScoring (See Limitations)      │
│    - Score candidate attachment points        │
│    - Geometric + temporal stability scoring    │
│    - Apply temporal filtering                  │
├──────────────────────────────────────────────────┤
│ 5. CandidateVisualizer                         │
│    - Render results for visualization          │
└──────────────────────────────────────────────────┘
    ↓
Output: best_candidate (position, score, stability status)
```

### Directory Structure

```
detection_pipeline/
├── main.py                              # Entry point
├── README.md                            # This file
├── config/
│   ├── __init__.py                      # ConfigManager class
│   └── default_config.yaml              # Configuration parameters
├── pipeline/                            # Core orchestration
│   ├── DetectionPipeline.py            # Main pipeline executor
│   ├── PipelineContext.py              # Data container
│   └── Stage.py                        # Abstract stage base class
├── sources/                             # Frame input abstraction
│   ├── FrameSource.py                  # Abstract interface
│   ├── MP4Source.py                    # Video file input
│   ├── example.mp4                     # Test video
│   ├── CameraSource.py                 # Live camera (Raspberry Pi)
│   └── NiclaSource.py                  # Nicla Vision input (serial)
├── detectors/                           # Object detection layer
│   ├── Detector.py                     # Abstract interface
│   ├── YOLOBranchSeg.py                # YOLO segmentation detector
│   └── HailoSegDetector.py             # Hailo segmentation detector
├── trackers/                            # Object tracking layer
│   ├── Tracker.py                      # Abstract interface
│   └── ByteTrack.py                    # ByteTrack implementation
├── postprocessing/                      # Multi-stage processing
│   ├── PostProcessor.py                # Stage orchestrator
│   ├── masks/
│   │   └── MaskExtraction.py           # Binary mask generation
│   ├── geometry/
│   │   ├── DistanceHeatmap.py          # Distance transform
│   │   ├── BitmaskSkeleton.py          # Skeletonization
│   │   └── LineFit.py                  # Line fitting (disabled)
│   └── scoring/
│       ├── CandidateScoring.py         # Candidate scoring
│       └── CandidateVisualizer.py      # Visualization
└── model/
    └── best.pt                         # Pre-trained YOLO model
```

---

## Installation

### Requirements

- Python 3.8 or higher
- CUDA 11.8+ (optional, for GPU acceleration)

### Dependencies

- **ultralytics** (≥8.0) - YOLOv8 detection
- **hailo_platform** - Hailo runtime (required for Hailo models)
- **supervision** (≥0.16) - ByteTrack tracking
- **opencv-python** (≥4.5) - Computer vision
- **numpy** (≥1.20) - Numerical operations
- **scikit-image** (≥0.19) - Skeletonization
- **pyyaml** (≥6.0) - Configuration parsing
- **picamera2** (optional) - Raspberry Pi camera support

## Configuration

### Configuration System

All parameters are configurable via `config/default_config.yaml`. The system supports:

1. **YAML configuration file** - Main parameter source
2. **Environment variable overrides** - Runtime parameter adjustment
3. **Dot-notation access** - Nested parameter retrieval


### Environment Variable Overrides

Set environment variables before running to override config values:

```bash
# Override source
export PIPELINE_SOURCE_TYPE="nicla"
export PIPELINE_VIDEO_PATH="/path/to/video.mp4"
export PIPELINE_NICLA_SERIAL_PORT="/dev/ttyACM0"

# Override detector
export PIPELINE_DETECTOR_TYPE="hailo_seg"
# Override detector confidence
export PIPELINE_CONF_THRESHOLD="0.3"

# Disable visualization (headless mode)
export PIPELINE_VISUALIZATION_ENABLED="false"

# Then run
python3 main.py
```

---

## Usage

### Basic Usage

```bash
# Run with default configuration
python3 main.py

# Stop: Press 'q' or Ctrl+C
```

### Custom Configuration

```bash
# Use custom config file
python main.py --config path/to/config.yaml
```

### Headless Mode (No Display)

```bash
# Run without visualization (useful for servers/UAVs)
PIPELINE_VISUALIZATION_ENABLED=false python main.py
```

### With Custom Video

```bash
export PIPELINE_VIDEO_PATH="path/to/video.mp4"
python main.py
```

---

## Algorithm Details

### 1. YOLO Detection (YOLOBranchSeg)

**Purpose**: Detect branches and generate segmentation masks

**Process**:
- Pre-trained YOLOv8 Segment model processes each frame
- Outputs: bounding boxes, instance masks, confidence scores
- Configurable confidence threshold (default: 0.5)

**Output**: Boxes (x₁, y₁, x₂, y₂), masks, confidence values

### 1b. Hailo Segmentation (HailoSegDetector)

**Purpose**: Run YOLOv8-style segmentation on Hailo hardware using HEF models

**Process**:
- Hailo runtime executes segmentation model
- Outputs: bounding boxes, instance masks, confidence scores

### 2. Multi-Object Tracking (ByteTrack)

**Purpose**: Maintain persistent object identity across frames

**Process**:
- ByteTrack algorithm associates detections across frames
- Assigns unique track IDs to branches
- Handles track initialization and termination

**Output**: List of (bbox, mask, confidence, class_id, track_id, data)

### 3. Binary Mask Extraction (MaskExtraction)

**Purpose**: Combine instance masks into single binary branch image

**Process**:
- Max-pool all instance masks
- Apply threshold (default: 0.5)
- Output: Single binary mask representing all branches

**Output**: Binary image (0=background, 255=branch)

### 4. Distance Heatmap (DistanceHeatmap)

**Purpose**: Compute distance from each pixel to nearest non-branch pixel

**Process**:
- Euclidean distance transform on binary mask
- Identifies branch centerlines (maximum distance)
- Normalized to [0, 1] range

**Output**: Distance heatmap (float32)

### 5. Skeleton Generation (BitmaskSkeleton)

**Purpose**: Reduce branches to 1-pixel-wide medial axis

**Process**:
- Morphological skeletonization using scikit-image
- Preserves branch connectivity
- Essential for identifying junction and endpoint locations

**Output**: Skeleton image (0=non-skeleton, 255=skeleton)

### 6. Candidate Scoring (CandidateScoring) **INCOMPLETE & UNRELIABLE**

** CRITICAL LIMITATION**: This component is not validated against ground truth and should NOT be used for production UAV deployment without extensive validation.

**Purpose**: Score candidate hook attachment points on branch skeleton

**Scoring Formula**:

For each point on skeleton within track bounding box:

```
geom_score = w_boundary × d_boundary + w_junction × d_junction + w_endpoint × d_endpoint

track_bonus = w_age × age + w_hit_streak × hit_streak + w_conf × conf_avg + w_consistency × consistency

total_score = geom_score + track_bonus
```

**Geometric Factors**:
- **d_boundary** - Distance to branch center (prefer center points)
- **d_junction** - Distance to branch junctions (prefer away from junctions)
- **d_endpoint** - Distance to branch endpoints (prefer away from endpoints)

**Temporal Factors**:
- **age** - How long track has existed
- **hit_streak** - Consecutive frames track was detected
- **conf_avg** - Average confidence from detector
- **consistency** - Variance in recent scores

**Default Weights**:
```
w_boundary = 1.0    # Strongly prefer branch centers
w_junction = 0.3    # Weakly penalize junctions
w_endpoint = 0.2    # Weakly penalize endpoints
w_age = 0.05        # Weak temporal bonus
w_hit_streak = 0.1  # Moderate temporal bonus
w_conf = 0.2        # Consider detector confidence
w_consistency = 0.1 # Weak stability bonus
```

**Limitations**:
- Weights are empirically tuned without ground truth validation
- No formal justification for weight ratios
- Spatial clustering uses O(n²) algorithm (not scalable)
- Temporal locking mechanism is heuristic-based
- May produce incorrect attachment points
- Not validated for safety-critical UAV operations

### 7. Temporal Stability Filtering (CandidateScoring)

**Purpose**: Prevent rapid point jitter by locking to stable locations

**Process**:
1. Maintain buffer of last 10 frame locations
2. Identify clusters of points within 15-pixel radius
3. Lock candidate when cluster appears in 5+ consecutive frames
4. Validate locked point still exists on skeleton
5. Release lock if point drifts or track disappears

**Output**: best_candidate with `is_stable` flag

### 8. Visualization (CandidateVisualizer)

**Display Elements**:
- **Skeleton** (dark gray) - Branch medial axis
- **Candidate points** (blue/green) - Scoring candidates
- **Best candidate** (red circle) - Top-scoring point
- **Statistics** - Frame counts, stability progress

**Color Coding**:
- 🔵 Blue: Unconfirmed candidates
- 🟢 Green: Confirmed candidates (≥3 frames)
- 🔴 Red: Unstable best candidate
- 🟢 Green: Stable best candidate (locked)

---

## Output Format

The pipeline yields `PipelineContext` objects with:

### best_candidate Dictionary

```python
{
    "track_id": int,                    # Unique track identifier
    "x": int,                           # Pixel x-coordinate
    "y": int,                           # Pixel y-coordinate
    "score": float,                     # Combined scoring value
    "geom_score": float,                # Geometric component
    "track_bonus": float,               # Temporal component
    "distance_to_boundary": float,      # Distance to branch center
    "distance_to_junction": float,      # Distance to nearest junction
    "distance_to_endpoint": float,      # Distance to nearest endpoint
    "track_age": int,                   # Frames track has existed
    "hit_streak": int,                  # Consecutive frames detected
    "confidence_avg": float,            # Average detector confidence
    "consistency_score": float,         # Score variance metric
    "confirmed": bool,                  # Hit streak ≥ min_confirm_frames
    "is_stable": bool,                  # Locked to stable location?
    "bbox": (x1, y1, x2, y2)           # Track bounding box
}
```

### candidates List

List of `best_per_track` - one candidate per tracked branch

### source_metadata (optional)

When using `NiclaSource`, `PipelineContext.source_metadata` includes:

```python
{
    "distance_mm": int
}
```

---

## References

1. **YOLOv8**: Ultralytics YOLOv8 Documentation
   - https://docs.ultralytics.com

2. **ByteTrack**: Multi-Object Tracking Framework
   - GitHub: https://github.com/ifzhang/ByteTrack
   - Paper: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"

3. **Supervision**: Detection & Tracking Toolkit
   - https://supervision.roboflow.com

4. **Image Processing**:
   - Distance Transform: https://docs.opencv.org/master/d3/db0/classcv_1_1DistanceTransformLabelTypes.html
   - Skeletonization: https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html
