from pathlib import Path

#from sources.CameraSource import CameraSource
from sources.MP4Source import MP4Source
from sources.NiclaSource import NiclaSource

from detectors.HailoSegDetector import HailoSegDetector
from detectors.YOLOBranchSeg import YOLOBranchSeg

from trackers.ByteTrack import ByteTrack

from postprocessing.PostProcessor import PostProcessor
from postprocessing.masks.MaskExtraction import MaskExtraction
from postprocessing.geometry.DistanceHeatmap import DistanceHeatmap
from postprocessing.geometry.BitmaskSkeleton import BitmaskSkeleton
from postprocessing.scoring.CandidateScoring import CandidateScoring
from postprocessing.scoring.WarmupFinalPoint import WarmupFinalPoint
from postprocessing.scoring.CandidateVisualizer import CandidateVisualizer

from pipeline.DetectionPipeline import DetectionPipeline

import supervision as sv
import cv2
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    _here = Path(__file__).resolve().parent

    logger.info("Starting Detection Pipeline")

    try:
        source = NiclaSource()
        detector = HailoSegDetector(hef_path=str(_here / "model" / "yolov8_segmentation.hef"), conf=0.7)
        logger.info("Using Nicla source with Hailo detector")
        print("Source: Nicla + Hailo (.hef)")
    except Exception as e:
        logger.warning(f"Nicla/Hailo unavailable ({e}), falling back to MP4 + YOLO")
        print(f"Source: MP4 + YOLO (.pt) [Nicla unavailable: {e}]")
        source = MP4Source(str(_here / "sources" / "example.mp4"))
        detector = YOLOBranchSeg(model_path=str(_here / "model" / "best_small.pt"), conf=0.7)

    visualization_enabled = True
    window_name = "Detection Pipeline"

    tracker = ByteTrack(detector)

    postprocessor = PostProcessor([
        MaskExtraction(),
        DistanceHeatmap(),
        BitmaskSkeleton(),
        CandidateScoring(),
        WarmupFinalPoint(),
        CandidateVisualizer(),
    ])

    pipeline = DetectionPipeline(source=source, detector=detector, tracker=tracker, postprocessor=postprocessor)
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator()

    frame_count = 0
    try:
        for ctx in pipeline.run():
            frame_count += 1

            if visualization_enabled:
                if "branch_score_image" in ctx.debug:
                    display_image = ctx.debug["branch_score_image"]
                else:
                    display_image = ctx.frame.copy()
                    sv_det = getattr(ctx, "sv_detections", None)
                    if sv_det is not None and len(sv_det) > 0:
                        display_image = mask_annotator.annotate(display_image, sv_det)
                        display_image = box_annotator.annotate(display_image, sv_det)
                        if sv_det.confidence is not None:
                            labels = [f"branch {c:.2f}" for c in sv_det.confidence]
                            display_image = label_annotator.annotate(display_image, sv_det, labels)

                cv2.imshow(window_name, display_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User quit")
                    break

            yield {
                "frame": ctx.frame,
                "final_point": ctx.final_point,
                "best_candidate": ctx.best_candidate,
                "reference_frame": ctx.reference_frame,
                "distance_mm": (ctx.source_metadata or {}).get("distance_mm"),
                "frame_w": ctx.frame.shape[1],
                "frame_h": ctx.frame.shape[0],
            }

            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} frames")

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
    finally:
        source.release()
        cv2.destroyAllWindows()
        logger.info(f"Pipeline complete. Processed {frame_count} frames total")


if __name__ == "__main__":
    for _ in main():
        pass
