#from sources.CameraSource import CameraSource
from sources.MP4Source import MP4Source
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
    source = MP4Source(config.get("source.video_path"))
    source.start_recording()  # Start recording if using CameraSource

    # Initialize detector
    detector = YOLOBranchSeg(
        model_path=config.get("detector.model_path"),
        conf=config.get("detector.confidence_threshold")
    )

    # Initialize tracker
    tracker = ByteTrack(detector)

    # Initialize postprocessor stages
    postprocessor = PostProcessor([
        MaskExtraction(),
        DistanceHeatmap(),
        BitmaskSkeleton(),
        #LineFit(),
        CandidateScoring(),
        CandidateVisualizer(),
    ])

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
            if visualization_enabled and "branch_score_image" in ctx.debug:
                cv2.imshow(
                    config.get("visualization.display_window_name"),
                    ctx.debug["branch_score_image"]
                )
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User quit")
                    break

            if frame_count % 30 == 0:
                logger.debug(f"Processed {frame_count} frames")

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
    finally:
        source.release()
        source.stop_recording()  # Stop recording if using CameraSource
        cv2.destroyAllWindows()
        logger.info(f"Pipeline complete. Processed {frame_count} frames total")



if __name__ == "__main__":
    main()