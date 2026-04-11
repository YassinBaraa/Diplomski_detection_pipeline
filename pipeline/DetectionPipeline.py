from pipeline.PipelineContext import PipelineContext

class DetectionPipeline:
    def __init__(self, source, detector, postprocessor):
        self.source = source
        self.detector = detector
        self.postprocessor = postprocessor

    def run(self):
        while True:
            ret, frame = self.source.read()
            if not ret:
                break

            ctx = PipelineContext(frame=frame)
            ctx.detections, ctx.masks, ctx.confidences = self.detector.predict(frame)
            ctx = self.postprocessor.process(ctx)

            yield ctx