import numpy as np
import cv2
from detection_pipeline_ros.pipeline.Stage import Stage

class DistanceHeatmap(Stage):
    def run(self, ctx):
        if ctx.binary_mask is None:
            return ctx

        #Determine the distance transform.
        dist = cv2.distanceTransform(ctx.binary_mask, cv2.DIST_L2, 5)

        # Make the distance transform normal.
        ctx.distance_heatmap = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

        #cv2.imshow('Distance Transform', ctx.distance_heatmap)
        #cv2.waitKey(100)

        return ctx
