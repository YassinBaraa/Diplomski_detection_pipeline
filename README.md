example
https://docs.ultralytics.com/guides/raspberry-pi/#inference-with-camera

https://www.geeksforgeeks.org/python/python-opencv-distancetransform-function/



initial plan for detection pipeline
    yolo seg segments branches ---add a accuracy treshold
    get bitmap of segemntations
    generate dstance heatmap
    fit line to heatmap points (see hough transform)
    pick best point with weight function

    
detection_pipeline/
├── main.py
├── pipeline/
│   ├── DetectionPipeline.py
│   ├── PipelineContext.py
│   └── Stage.py
├── sources/
│   ├── FrameSource.py
│   ├── CameraSource.py
│   └── MP4Source.py
├── detectors/
│   ├── Detector.py
│   └── YOLOBranchSeg.py
├── postprocessing/
│   ├── PostProcessor.py
│   ├── filtering/
│   │   └── DetectionFilterStage.py
│   ├── masks/
│   │   └── MaskExtractionStage.py
│   ├── geometry/
│   │   ├── DistanceTransformStage.py
│   │   ├── HoughLineStage.py
│   │   └── CandidatePointStage.py
│   ├── scoring/
│   │   ├── BranchScoringStage.py
│   │   └── BestBranchStage.py
│   └── debug/
│       ├── MaskOverlayStage.py
│       └── LineOverlayStage.py
├── model/
│   └── best.pt
└── utils/
    ├── types.py
    └── image_ops.py


PipelineContext
    frame: np.ndarray
    detections: Any = None
    masks: Any = None
    binary_mask: Optional[np.ndarray] = None
    heatmap: Optional[np.ndarray] = None
    lines: Any = None
    candidates: Any = None
    best_candidate: Any = None
    debug: dict = field(default_factory=dict)