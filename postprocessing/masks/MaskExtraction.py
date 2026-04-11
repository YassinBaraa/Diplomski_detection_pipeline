from pipeline.Stage import Stage

class MaskExtraction(Stage):
    def run(self, ctx):
        if ctx.detections is None or ctx.detections.masks is None:
            ctx.binary_mask = None
            return ctx

        ctx.binary_mask = ctx.detections.masks
        return ctx