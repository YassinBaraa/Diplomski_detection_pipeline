import numpy as np
import cv2
import logging
from detection_pipeline_ros.pipeline.Stage import Stage

logger = logging.getLogger(__name__)
BITMASK_THRESHOLD = 0.5

class MaskExtraction(Stage):
    def run(self, ctx):
        if ctx.masks is None or len(ctx.masks) == 0:
            return ctx

        # Validate mask dimensions match frame
        if ctx.frame is not None:
            mask_h, mask_w = ctx.masks[0].shape[:2] if len(ctx.masks[0].shape) >= 2 else ctx.masks[0].shape
            frame_h, frame_w = ctx.frame.shape[:2]
            if (mask_h, mask_w) != (frame_h, frame_w):
                logger.warning(f"Mask size mismatch: frame ({frame_h}x{frame_w}) vs masks ({mask_h}x{mask_w})")

        ctx.binary_mask = np.max(
            (ctx.masks > BITMASK_THRESHOLD).astype(np.uint8) * 255,
            axis=0
        )

        return ctx
