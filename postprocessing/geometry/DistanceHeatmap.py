import numpy as np
import cv2
from pipeline.Stage import Stage

class DistanceHeatmap(Stage):
    def run(self, ctx):
        if ctx.binary_mask is None:
            return ctx

        #Determine the distance transform.
        dist = cv2.distanceTransform(ctx.binary_mask, cv2.DIST_L2, 5)

        # Keep the raw maximum (px) so a normalised value can be turned back into
        # a real half-thickness in pixels.
        ctx.debug["heatmap_max_px"] = float(dist.max())

        # Make the distance transform normal.
        ctx.distance_heatmap = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

        #cv2.imshow('Distance Transform', ctx.distance_heatmap)
        #cv2.waitKey(100)

        return ctx