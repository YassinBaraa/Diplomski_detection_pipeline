from detection_pipeline_ros.pipeline.PipelineContext import PipelineContext
import logging

logger = logging.getLogger(__name__)

class DetectionPipeline:
    def __init__(self, source, detector, tracker, postprocessor):
        self.source = source
        self.detector = detector
        self.tracker = tracker
        self.postprocessor = postprocessor

    def run(self):
        while True:
            ret, frame = self.source.read()
            if not ret or frame is None:
                break

            ctx = PipelineContext(frame=frame)
            try:
                ctx.results, ctx.detections, ctx.masks, ctx.confidences = self.detector.predict(frame)
                ctx.tracks = self.tracker.track(frame, ctx.results)
                ctx = self.postprocessor.process(ctx)
            except Exception as e:
                logger.error(f"Pipeline error processing frame: {e}", exc_info=True)
                ctx.candidates = []
                ctx.best_candidate = None

            yield ctx
