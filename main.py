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
#from postprocessing.geometry.LineFit import LineFit
from postprocessing.scoring.CandidateScoring import CandidateScoring
from postprocessing.scoring.CandidateVisualizer import CandidateVisualizer

from pipeline.DetectionPipeline import DetectionPipeline
from config import ConfigManager

import cv2
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # Load configuration
    try:
        config = ConfigManager()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    logger.info("Starting Detection Pipeline")

    # Initialize source
    source_type = config.get("source.type", "mp4")
    if source_type == "mp4":
        source = MP4Source(config.get("source.video_path"))
    elif source_type == "camera":
        source = CameraSource()
    elif source_type == "nicla":
        source = NiclaSource()
    else:
        raise ValueError(f"Unsupported source type: {source_type}")

    # Initialize detector
    detector_type = config.get("detector.type", "yolo_seg")
    detector_model_path = config.get("detector.model_path")
    detector_conf = config.get("detector.confidence_threshold")

    if detector_type == "hailo_seg":
        detector = HailoSegDetector(
            hef_path=detector_model_path,
            conf=detector_conf,
        )
    elif detector_type == "yolo_seg":
        detector = YOLOBranchSeg(
            model_path=detector_model_path,
            conf=detector_conf,
        )
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}")

    # Initialize tracker
    tracker = ByteTrack(detector)

    # Initialize postprocessor stages (disabled for testing)
    # postprocessor = PostProcessor([
    #     MaskExtraction(),
    #     DistanceHeatmap(),
    #     BitmaskSkeleton(),
    #     #LineFit(),
    #     CandidateScoring(),
    #     CandidateVisualizer(),
    # ])
    postprocessor = None

    # Create and run pipeline
    pipeline = DetectionPipeline(
        source=source,
        detector=detector,
        tracker=tracker,
        postprocessor=postprocessor
    )

    # Process frames
    frame_count = 0
    visualization_enabled = config.get("visualization.enabled", True)

    try:
        for ctx in pipeline.run():
            frame_count += 1

            # Display visualization if enabled
            if visualization_enabled:
                # If no postprocessing visualization, show the frame with detections
                if "branch_score_image" in ctx.debug:
                    display_image = ctx.debug["branch_score_image"]
                else:
                    display_image = ctx.frame
                
                cv2.imshow(
                    config.get("visualization.display_window_name", "Detection Pipeline"),
                    display_image
                )
                if cv2.waitKey(1) & 0xFF == ord('q'):
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
        #source.stop_recording()  
        cv2.destroyAllWindows()
        logger.info(f"Pipeline complete. Processed {frame_count} frames total")



if __name__ == "__main__":
    main()
