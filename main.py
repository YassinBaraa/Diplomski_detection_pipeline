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
    
    # config_dict = {
    #     "source": {"type": "mp4", "video_path": "sources/example.mp4"},
    #     "detector": {"type": "yolo_seg", "model_path": "model/best_small.pt", "confidence_threshold": 0.5},
    #     "visualization": {"enabled": True, "display_window_name": "Detection Pipeline"},
    #     "postprocessing_enabled": False,
    # }
    
    try:
        config = ConfigManager()
        config_dict = {
            "source": {"type": "mp4", "video_path": "sources/example.mp4"},
            "detector": {"type": "yolo_seg", "model_path": "model/best_small.pt", "confidence_threshold": 0.5},
            "visualization": {"enabled": True, "display_window_name": "Detection Pipeline"},
            "postprocessing_enabled": True,
        }
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return
    
    logger.info("Starting Detection Pipeline")
    
    # Initialize source
    source_type = config_dict["source"]["type"]
    if source_type == "mp4":
        source = MP4Source(config_dict["source"]["video_path"])
        logger.info(f" MP4Source initialized: {config_dict['source']['video_path']}")
    elif source_type == "camera":
        source = CameraSource()
        logger.info(" CameraSource initialized")
    elif source_type == "nicla":
        source = NiclaSource()
        logger.info(" NiclaSource initialized")
    else:
        raise ValueError(f"Unsupported source type: {source_type}")
    
    # Initialize detector
    detector_type = config_dict["detector"]["type"]
    detector_model_path = config_dict["detector"].get("model_path", "model/best.pt")
    detector_conf = config_dict["detector"].get("confidence_threshold", 0.5)
    
    if detector_type == "yolo_seg":
        detector = YOLOBranchSeg(model_path=detector_model_path, conf=detector_conf)
        logger.info(f" YOLOBranchSeg initialized: {detector_model_path} (conf={detector_conf})")
    elif detector_type == "hailo_seg":
        detector = HailoSegDetector(hef_path=detector_model_path, conf=detector_conf)
        logger.info(f" HailoSegDetector initialized: {detector_model_path} (conf={detector_conf})")
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}")
    
    # Initialize tracker
    tracker = ByteTrack(detector)
    logger.info(" ByteTrack initialized")
    
    # Initialize postprocessor (optional)
    if config_dict.get("postprocessing_enabled", True):
        postprocessor = PostProcessor([
            MaskExtraction(), # makes bitmask from segmentation output
            DistanceHeatmap(), # calculates bitmask from segmentation mask
            BitmaskSkeleton(), # calculates skeleton from bitmask
            CandidateScoring(), # scoring candidates based on postprocess and weight function
            CandidateVisualizer(),
        ])
        logger.info(" PostProcessor initialized with full pipeline")
    else:
        postprocessor = None
        logger.info("⊘ PostProcessor disabled - raw detections only")

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