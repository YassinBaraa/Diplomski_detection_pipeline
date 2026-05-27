from pipeline.PipelineContext import PipelineContext
import logging

logger = logging.getLogger(__name__)

class DetectionPipeline:
    def __init__(self, source, detector, tracker, postprocessor=None):
        self.source = source
        self.detector = detector
        self.tracker = tracker
        self.postprocessor = postprocessor

    def run(self):
        while True:
            read_result = self.source.read()
            if not isinstance(read_result, tuple):
                raise ValueError("FrameSource.read() must return a tuple")
            if len(read_result) == 2:
                ret, frame = read_result
                source_metadata = None
            elif len(read_result) == 3:
                ret, frame, source_metadata = read_result
            else:
                raise ValueError("FrameSource.read() must return 2 or 3 values")

            if not ret or frame is None:
                break

            ctx = PipelineContext(frame=frame, source_metadata=source_metadata)
            try:
                prediction = self.detector.predict(frame)
                if isinstance(prediction, tuple):
                    if len(prediction) == 4:
                        ctx.results, ctx.detections, ctx.masks, ctx.confidences = prediction
                        ctx.class_ids = None
                    elif len(prediction) == 5:
                        ctx.results, ctx.detections, ctx.masks, ctx.confidences, ctx.class_ids = prediction
                    else:
                        raise ValueError("Detector.predict() must return 4 or 5 values")
                else:
                    ctx.results = prediction
                    ctx.detections = None
                    ctx.masks = None
                    ctx.confidences = None
                    ctx.class_ids = None

                ctx.tracks = self.tracker.track(
                    frame,
                    results=ctx.results,
                    bboxes=ctx.detections,
                    masks=ctx.masks,
                    confidences=ctx.confidences,
                    class_ids=ctx.class_ids,
                )
                if self.postprocessor is not None:
                    ctx = self.postprocessor.process(ctx)
            except Exception as e:
                logger.error(f"Pipeline error processing frame: {e}", exc_info=True)
                ctx.candidates = []
                ctx.best_candidate = None

            yield ctx
