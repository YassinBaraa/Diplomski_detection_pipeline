import numpy as np
import cv2
import logging
from skimage.morphology import skeletonize

from pipeline.Stage import Stage

logger = logging.getLogger(__name__)

class BitmaskSkeleton(Stage):
    def run(self, ctx):
        if ctx.binary_mask is None:
            ctx.skeleton = None
            return ctx

        # Validate mask matches frame size
        if ctx.frame is not None:
            mask_h, mask_w = ctx.binary_mask.shape[:2]
            frame_h, frame_w = ctx.frame.shape[:2]
            if (mask_h, mask_w) != (frame_h, frame_w):
                logger.warning(f"Binary mask size mismatch: frame ({frame_h}x{frame_w}) vs mask ({mask_h}x{mask_w})")

        # Convert 0/255 to 0/1 for skeletonize
        normalized_mask = ctx.binary_mask // 255

        # Skeletonize (produces 0/1 boolean)
        skeleton = skeletonize(normalized_mask)

        # Convert to OpenCV binary mask (0/255 uint8)
        ctx.skeleton = skeleton.astype(np.uint8) * 255

        return ctx