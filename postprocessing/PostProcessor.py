from postprocessing.masks.MaskExtraction import MaskExtraction

class PostProcessor:
    def __init__(self, stages):
        self.stages = stages

    def process(self, ctx):
        for stage in self.stages:
            #Go through each stage and update the context
            ctx = stage.run(ctx)
        return ctx