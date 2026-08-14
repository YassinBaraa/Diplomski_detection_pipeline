# Detection Pipeline

Real-time computer vision pipeline for tree branch detection and perch point estimation. Part of the UAV branch perching master's thesis system.

## What It Does

Processes video frames to find the best point on a branch where the UAV can attach a hook. After a warmup period it clusters observed candidate points and locks a `final_point` with a reference frame snapshot — this is handed off to the IBVS pipeline for model-free KLT tracking.

The pipeline is a generator: `DetectionPipeline.run()` yields a `PipelineContext` each frame.

---

## Pipeline Stages

```
Frame (DSJSource / NiclaSource / MP4Source / CameraSource)
  ↓
HailoSegDetector / YOLOBranchSeg   — instance segmentation (.hef or .pt)
  ↓
ByteTrack                          — multi-object tracking across frames
  ↓
MaskExtraction                     — merge instance masks → binary image
  ↓
DistanceHeatmap                    — euclidean distance transform
  ↓
BitmaskSkeleton                    — 1-pixel medial axis
  ↓
CandidateScoring                   — score skeleton points, pick best candidate
  ↓
WarmupFinalPoint                   — collect candidates over N frames → cluster → lock final_point
  ↓
CandidateVisualizer                — annotate frame
  ↓
yield PipelineContext
```

---

## Directory Structure

```
detection_pipeline/
├── main.py
├── test_record.py
├── config/
│   └── __init__.py
├── pipeline/
│   ├── DetectionPipeline.py         # Orchestrates source → detector → tracker → postprocessor
│   ├── PipelineContext.py           # Per-frame data container
│   └── Stage.py                     # Abstract stage base class
├── sources/
│   ├── FrameSource.py
│   ├── MP4Source.py
│   ├── CameraSource.py
│   ├── DSJSource.py                 # DSJ-3079-HE: USB UVC camera (default)
│   └── NiclaSource.py               # Nicla Vision: OV5647 camera + VL53L1X ToF over USB serial
├── detectors/
│   ├── Detector.py
│   ├── HailoSegDetector.py          # Hailo-8 accelerated YOLOv8-seg inference (.hef)
│   ├── YOLOBranchSeg.py             # YOLOv8-seg via ultralytics (.pt)
│   └── YOLOBranchDet.py             # YOLOv8 detection-only variant
├── trackers/
│   ├── Tracker.py
│   └── ByteTrack.py                 # supervision ByteTrack wrapper
├── postprocessing/
│   ├── PostProcessor.py
│   ├── masks/MaskExtraction.py
│   ├── geometry/
│   │   ├── DistanceHeatmap.py
│   │   ├── BitmaskSkeleton.py
│   │   └── LineFit.py
│   └── scoring/
│       ├── CandidateScoring.py
│       ├── WarmupFinalPoint.py      # Warmup → hierarchical clustering → final_point lock
│       └── CandidateVisualizer.py
└── model/
    └── yolov8_segmentation.hef      # Hailo-compiled model (place here)
```

---

## DSJSource

Default camera. The DSJ-3079-HE is a standard UVC (USB Video Class) device
(Realtek UC60 bridge chip, USB ID `0bda:5805`) — driven entirely by the
kernel's `uvcvideo` driver via V4L2/OpenCV, no vendor SDK needed. Captures
MJPG at 1280×720@30fps by default (native modes up to 3264×2448@15fps).

It resolves its device path via udev's `/dev/v4l/by-id/*DSJ_UC60_Video*`
symlink rather than a hardcoded `/dev/videoN` index, since USB enumeration
order isn't guaranteed to survive a reboot/replug. Falls back to device index
`0` if no by-id match is found (e.g. a differently-branded UC60 clone).

It exposes two `/dev/videoN` nodes — one actual capture node, one
metadata-only (`v4l2-ctl -d <dev> --list-formats-ext` shows which is which;
the capture node lists MJPG/YUYV formats, the metadata node lists none).
`DSJSource` always opens the capture node.

## NiclaSource

Reads JPEG frames and ToF distance from an Arduino Nicla Vision over USB serial (`/dev/ttyACM0`, 115200 baud).

The Nicla runs a MicroPython script that streams framed packets:
```
IMG:<length>\n<jpeg bytes>DIST:<mm>\n
```

**Auto-mount issue**: Linux (udisks2) auto-mounts the Nicla flash as a USB drive, which pauses MicroPython and stops the stream. Install this udev rule to prevent it:

```
# /etc/udev/rules.d/99-nicla-vision.rules
ACTION=="add", SUBSYSTEM=="block", ATTRS{idVendor}=="2341", ATTRS{idProduct}=="055f", ENV{UDISKS_AUTO}="0", ENV{UDISKS_IGNORE}="1"
```

```bash
sudo udevadm control --reload-rules
# Replug the Nicla
```

`NiclaSource` detects a silent Nicla (no `IMG:` within 8 s) and automatically sends `pyb.hard_reset()` over the REPL, up to 2 attempts before raising `RuntimeError`.

## HailoSegDetector

Runs YOLOv8-seg on Hailo-8 via `hailo_platform.VDevice`. Decodes raw output tensors: DFL box regression, sigmoid class scores, and mask coefficients. Post-processing (NMS) runs on CPU. Requires `hailort >= 4.23`.

---

## PipelineContext Fields

| Field | Type | Description |
|-------|------|-------------|
| `frame` | `np.ndarray` | BGR image |
| `source_metadata` | `dict \| None` | source-specific extra data (currently unused; the Nicla still frames each JPEG with a ToF reading over serial, but it is discarded, not surfaced here) |
| `detections` | `np.ndarray [N,4]` | xyxy bounding boxes |
| `masks` | `np.ndarray [N,H,W]` | binary segmentation masks |
| `confidences` | `np.ndarray [N]` | detection scores |
| `tracks` | `list` | active ByteTrack tracks |
| `best_candidate` | `dict \| None` | `{"x", "y", "score", ...}` |
| `final_point` | `dict \| None` | `{"x", "y"}` locked after warmup |
| `reference_frame` | `np.ndarray \| None` | frame snapshot when final_point locked |
| `debug` | `dict` | per-stage diagnostics |

`final_point` is `None` during warmup. After warmup it is re-emitted every frame unchanged.

---

## Installation

```bash
pip install ultralytics supervision opencv-python numpy scipy scikit-image pyserial

# Hailo-8
pip install hailo_platform   # also requires hailort system package (4.23+)

# RPi camera
pip install picamera2
```

## Running

```bash
python main.py          # standalone with display (DSJ + Hailo, falls back to MP4 + YOLO)
python test_record.py   # record output to video
```

## Platform

- **Inference**: Hailo-8 AI HAT on Raspberry Pi 5 (hailort 4.23)
- **Camera**: DSJ-3079-HE USB UVC camera (default); Arduino Nicla Vision (OV5647 + VL53L1X ToF) also supported
- **Fallback**: `.pt` YOLOv8 model via `YOLOBranchSeg` (no Hailo required)
