from sources.CameraSource import CameraSource
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
from postprocessing.scoring.CandidateVisualizer import CandidateVisualizer

from pipeline.DetectionPipeline import DetectionPipeline
from config import ConfigManager

import supervision as sv
import cv2
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

box_annotator   = sv.BoxAnnotator()
mask_annotator  = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator()


def annotate_frame(frame, ctx):
    """Draw boxes and masks using ctx.sv_detections when postprocessor is disabled."""
    display = frame.copy()

    sv_detections = getattr(ctx, 'sv_detections', None)
    if sv_detections is None or len(sv_detections) == 0:
        return display

    display = mask_annotator.annotate(display, sv_detections)
    display = box_annotator.annotate(display, sv_detections)

    if sv_detections.confidence is not None:
        labels = [f"branch {c:.2f}" for c in sv_detections.confidence]
        display = label_annotator.annotate(display, sv_detections, labels)

    return display


def main():
    try:
        config = ConfigManager()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    logger.info("Starting Detection Pipeline")

    source_type = config.get("source.type", "mp4")
    if source_type == "mp4":
        source = MP4Source(config.get("source.video_path"))
    elif source_type == "camera":
        source = CameraSource()
    elif source_type == "nicla":
        source = NiclaSource()
    else:
        raise ValueError(f"Unsupported source type: {source_type}")

    detector_type       = config.get("detector.type", "yolo_seg")
    detector_model_path = config.get("detector.model_path")
    detector_conf       = config.get("detector.confidence_threshold")

    if detector_type == "hailo_seg":
        detector = HailoSegDetector(hef_path=detector_model_path, conf=detector_conf)
    elif detector_type == "yolo_seg":
        detector = YOLOBranchSeg(model_path=detector_model_path, conf=detector_conf)
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}")

    tracker = ByteTrack(detector)

    postprocessor = PostProcessor([
        MaskExtraction(),
        DistanceHeatmap(),
        BitmaskSkeleton(),
        CandidateScoring(),
        CandidateVisualizer(),
    ])
    # postprocessor = None

    pipeline = DetectionPipeline(
        source=source,
        detector=detector,
        tracker=tracker,
        postprocessor=postprocessor
    )

    frame_count = 0
    visualization_enabled = config.get("visualization.enabled", True)

    try:
        for ctx in pipeline.run():
            frame_count += 1

            if visualization_enabled:
                if "branch_score_image" in ctx.debug:
                    display_image = ctx.debug["branch_score_image"]
                else:
                    display_image = annotate_frame(ctx.frame, ctx)

                cv2.imshow(
                    config.get("visualization.display_window_name", "Detection Pipeline"),
                    display_image
                )
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("User quit")
                    break

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
    main()