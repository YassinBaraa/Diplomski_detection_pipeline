import numpy as np
import cv2
from pipeline.Stage import Stage

BITMASK_THRESHOLD = 0.5

class MaskExtraction(Stage):
    def run(self, ctx):
        if ctx.masks is None or len(ctx.masks) == 0:
            ctx.binary_mask = None
            return ctx

        ctx.binary_mask = np.max(
            (ctx.masks > BITMASK_THRESHOLD).astype(np.uint8) * 255,
            axis=0
        )

        #cv2.imshow("Binary Mask", ctx.binary_mask)
        #cv2.waitKey(100)

        return ctx