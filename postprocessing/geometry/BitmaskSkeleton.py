import numpy as np
import cv2
from skimage.morphology import skeletonize
from pipeline.Stage import Stage

class BitmaskSkeleton(Stage):
    def run(self, ctx):
        if ctx.binary_mask is None:
            ctx.skeleton = None
            return ctx

        # Convert 0/255 to 0/1 for skeletonize
        normalized_mask = ctx.binary_mask // 255
        
        # Skeletonize (produces 0/1 boolean)
        skeleton = skeletonize(normalized_mask)
        
        # Convert to OpenCV binary mask (0/255 uint8)
        ctx.skeleton = skeleton.astype(np.uint8) * 255

        cv2.imshow("Skeleton", ctx.skeleton)
        cv2.waitKey(100)
        
        return ctx