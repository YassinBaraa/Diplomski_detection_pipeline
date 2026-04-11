#from sources.CameraSource import CameraSource
from sources.MP4Source import MP4Source
from detectors.YOLOBranchSeg import YOLOBranchSeg
from postprocessing.PostProcessor import PostProcessor
from postprocessing.masks.MaskExtraction import MaskExtraction
#from postprocessing.geometry.DistanceHeatmap import DistanceHeatmap
from postprocessing.geometry.BitmaskSkeleton import BitmaskSkeleton
#from postprocessing.lines.LineDetectionStage import LineDetectionStage
from pipeline.DetectionPipeline import DetectionPipeline

def main():
    #source = CameraSource()
    source = MP4Source("sources/example.mp4")

    detector = YOLOBranchSeg()

    postprocessor = PostProcessor([
        MaskExtraction(),
        #DistanceHeatmap(),
        BitmaskSkeleton(),
        #LineDetectionStage(),
    ])

    pipeline = DetectionPipeline(
        source=source,
        detector=detector,
        postprocessor=postprocessor
    )

    #pipeline.run() yields context objects
    for ctx in pipeline.run():
        print(ctx.best_candidate)

if __name__ == "__main__":
    main()