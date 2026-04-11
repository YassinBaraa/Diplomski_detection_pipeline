from sources.CameraSource import CameraSource
from detectors.YOLOBranchSeg import YOLOBranchSeg
from postprocessing.PostProcessor import PostProcessor
from postprocessing.masks.MaskExtraction import MaskExtraction
from postprocessing.heatmap.HeatmapStage import HeatmapStage
from postprocessing.lines.LineDetectionStage import LineDetectionStage
from pipeline.DetectionPipeline import DetectionPipeline

def main():
    source = CameraSource()
    detector = YOLOBranchSeg()

    postprocessor = PostProcessor([
        MaskExtraction(),
        HeatmapStage(),
        LineDetectionStage(),
    ])

    pipeline = DetectionPipeline(
        source=source,
        detector=detector,
        postprocessor=postprocessor
    )

    for ctx in pipeline.run():
        print(ctx.best_candidate)

if __name__ == "__main__":
    main()