from pathlib import Path

#from sources.CameraSource import CameraSource
from sources.MP4Source import MP4Source
#from sources.NiclaSource import NiclaSource

#from detectors.HailoSegDetector import HailoSegDetector
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

    config_dict = {
        "source": {"type": "mp4", "video_path": str(_here / "sources" / "example.mp4")},
        "detector": {"type": "yolo_seg", "model_path": str(_here / "model" / "best_small.pt"), "confidence_threshold": 0.7},
        "visualization": {"enabled": True, "display_window_name": "Detection Pipeline"},
        "postprocessing_enabled": True,
    }
    # config_dict = {
    #     "source": {"type": "nicla"},
    #     "detector": {"type": "hailo_seg", "model_path": "model/yolov8_segmentation.hef", "confidence_threshold": 0.7},
    #     "visualization": {"enabled": True, "display_window_name": "Detection Pipeline"},
    #     "postprocessing_enabled": True,
    # }

    logger.info("Starting Detection Pipeline")

    source_type = config_dict["source"]["type"]
    if source_type == "mp4":
        source = MP4Source(config_dict["source"]["video_path"])
    elif source_type == "nicla":
        source = NiclaSource()
    else:
        raise ValueError(f"Unsupported source type: {source_type}")

    detector_conf = config_dict["detector"]["confidence_threshold"]
    detector_type = config_dict["detector"]["type"]
    detector_path = config_dict["detector"]["model_path"]

    if detector_type == "yolo_seg":
        detector = YOLOBranchSeg(model_path=detector_path, conf=detector_conf)
    elif detector_type == "hailo_seg":
        detector = HailoSegDetector(hef_path=detector_path, conf=detector_conf)
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}")

    tracker = ByteTrack(detector)

    postprocessor = PostProcessor([
        MaskExtraction(),
        DistanceHeatmap(),
        BitmaskSkeleton(),
        CandidateScoring(),
        WarmupFinalPoint(),
        CandidateVisualizer(),
    ]) if config_dict.get("postprocessing_enabled", True) else None

    pipeline = DetectionPipeline(source=source, detector=detector, tracker=tracker, postprocessor=postprocessor)

    visualization_enabled = config_dict["visualization"]["enabled"]
    window_name = config_dict["visualization"]["display_window_name"]
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
