example
https://docs.ultralytics.com/guides/raspberry-pi/#inference-with-camera

https://www.geeksforgeeks.org/python/python-opencv-distancetransform-function/

https://scikit-image.org/docs/0.25.x/auto_examples/edges/plot_skeleton.html

https://docs.opencv.org/3.4/js_contour_features_fitLine.html


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
│   ├── masks/
│   │   └── MaskExtraction.py
│   ├── geometry/
│   │   ├── DistanceHeatmap.py
│   │   ├── LineFit.py
│   │   └── BitmaskSkeleton.py
│   └── scoring/
│       └── CandidateScoring.py
└── model/
    └── best.pt