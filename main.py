#from sources.CameraSource import CameraSource
from sources.MP4Source import MP4Source
from detectors.YOLOBranchSeg import YOLOBranchSeg
from postprocessing.PostProcessor import PostProcessor
from postprocessing.masks.MaskExtraction import MaskExtraction
from postprocessing.geometry.DistanceHeatmap import DistanceHeatmap
from postprocessing.geometry.BitmaskSkeleton import BitmaskSkeleton
#from postprocessing.geometry.LineFit import LineFit
from postprocessing.scoring.CandidateScoring import CandidateScoring
from pipeline.DetectionPipeline import DetectionPipeline

import cv2

def main():
    #source = CameraSource()
    source = MP4Source("sources/example.mp4")

    detector = YOLOBranchSeg()

    postprocessor = PostProcessor([
        MaskExtraction(),
        DistanceHeatmap(),
        BitmaskSkeleton(),
        #LineFit(),
        CandidateScoring(),
    ])

    pipeline = DetectionPipeline(
        source=source,
        detector=detector,
        postprocessor=postprocessor
    )

    #pipeline.run() yields context objects
    for ctx in pipeline.run():
        #print(ctx.best_candidate)
        cv2.imshow("Branch Score", ctx.debug["branch_score_image"])
        cv2.waitKey(1)

if __name__ == "__main__":
    main()