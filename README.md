# Detection Pipeline

Real-time computer vision pipeline for tree branch detection and perch point estimation. Part of the UAV branch perching master's thesis system.

## What It Does

Processes video frames to find the best point on a branch where the UAV can attach a hook. After a 20-second warmup it locks a `final_point` and captures a reference frame — this is handed off to the IBVS pipeline for model-free tracking.

The pipeline is a generator: `main()` yields a dict each frame consumed by `ibvs/`.

---

## Pipeline Stages

```
Frame
  ↓
YOLOBranchSeg / HailoSegDetector   — instance segmentation
  ↓
ByteTrack                          — multi-object tracking across frames
  ↓
MaskExtraction                     — merge instance masks → binary image
  ↓
DistanceHeatmap                    — euclidean distance transform
  ↓
BitmaskSkeleton                    — 1-pixel medial axis
  ↓
CandidateScoring                   — score skeleton points, pick best
  ↓
WarmupFinalPoint                   — 20s warmup → lock final_point
  ↓
CandidateVisualizer                — draw result on frame
  ↓
yield dict
```

---

## Directory Structure

```
detection_pipeline/
├── main.py                          # Entry point — generator, yields per-frame dict
├── config/
│   ├── __init__.py                  # ConfigManager (YAML + env var overrides)
│   └── default_config.yaml
├── pipeline/
│   ├── DetectionPipeline.py         # Runs stages in sequence
│   ├── PipelineContext.py           # Per-frame data container
│   └── Stage.py                     # Abstract stage base class
├── sources/
│   ├── FrameSource.py               # Abstract interface
│   ├── MP4Source.py                 # Video file
│   ├── CameraSource.py              # RPi camera
│   └── NiclaSource.py              # Nicla Vision (camera + ToF over serial)
├── detectors/
│   ├── Detector.py                  # Abstract interface
│   ├── YOLOBranchSeg.py             # YOLOv8 segmentation (CPU/GPU)
│   └── HailoSegDetector.py          # Hailo-8 accelerated inference
├── trackers/
│   ├── Tracker.py
│   └── ByteTrack.py
├── postprocessing/
│   ├── PostProcessor.py
│   ├── masks/MaskExtraction.py
│   ├── geometry/
│   │   ├── DistanceHeatmap.py
│   │   ├── BitmaskSkeleton.py
│   │   └── LineFit.py
│   └── scoring/
│       ├── CandidateScoring.py
│       ├── WarmupFinalPoint.py      # Warmup logic + final_point locking
│       └── CandidateVisualizer.py
└── model/
    └── best_small.pt                # Pre-trained YOLO weights (place here)
```

---

## Yield Output (per frame)

```python
{
    "frame": np.ndarray,          # BGR image
    "final_point": dict | None,   # {"x": int, "y": int} — locked after 20s warmup
    "best_candidate": dict | None,# {"x": int, "y": int, "score": float, ...}
    "reference_frame": np.ndarray | None,  # frame captured when final_point locked
    "distance_mm": int | None,    # ToF reading from NiclaSource
    "frame_w": int,
    "frame_h": int,
}
```

`final_point` is `None` during warmup. After warmup it stays locked until the pipeline ends.

---

## Installation

```bash
pip install ultralytics supervision opencv-python numpy scikit-image pyyaml
# For Hailo:
pip install hailo_platform
# For RPi camera:
pip install picamera2
```

## Usage

```bash
# Standalone (opens window)
python main.py

# Headless
PIPELINE_VISUALIZATION_ENABLED=false python main.py

# Custom video
PIPELINE_VIDEO_PATH=path/to/video.mp4 python main.py

# Nicla Vision source (serial)
PIPELINE_SOURCE_TYPE=nicla PIPELINE_NICLA_SERIAL_PORT=/dev/ttyACM0 python main.py

# Hailo detector
PIPELINE_DETECTOR_TYPE=hailo_seg python main.py
```

## Configuration

Edit `config/default_config.yaml` or override with `PIPELINE_*` environment variables.

Key parameters:

| Key | Default | Description |
|-----|---------|-------------|
| `source.type` | `mp4` | `mp4`, `camera`, `nicla` |
| `detector.type` | `yolo_seg` | `yolo_seg`, `hailo_seg` |
| `detector.confidence_threshold` | `0.5` | Detection confidence cutoff |
| `warmup.duration_seconds` | `20` | Warmup before final_point locks |
| `visualization.enabled` | `true` | Show OpenCV window |
