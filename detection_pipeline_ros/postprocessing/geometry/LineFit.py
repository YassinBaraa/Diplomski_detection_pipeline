#DOES NOT WORK WELL, REMOVED FROM PIPELINE

import cv2
import numpy as np
from detection_pipeline_ros.pipeline.Stage import Stage


MINIMUM_SKELETON_POINTS = 10


class LineFit(Stage):
    def run(self, ctx):
        if ctx.skeleton is None:
            ctx.lines = []
            return ctx

        ys, xs = np.where(ctx.skeleton > 0)

        if len(xs) < MINIMUM_SKELETON_POINTS:
            ctx.lines = []
            return ctx

        # OpenCV expects points in (x, y), not (row, col)
        points = np.column_stack((xs, ys)).astype(np.float32)

        vx, vy, x0, y0 = cv2.fitLine(
            points,
            cv2.DIST_L2,
            0,
            0.01,
            0.01
        ).reshape(-1)

        h, w = ctx.skeleton.shape[:2]

        # Build long line in parametric form:
        # p(t) = (x0, y0) + t * (vx, vy)
        intersections = []

        eps = 1e-6

        # Intersect with x = 0
        if abs(vx) > eps:
            t = (0 - x0) / vx
            y = y0 + t * vy
            if 0 <= y < h:
                intersections.append((0, int(round(y))))

        # Intersect with x = w - 1
        if abs(vx) > eps:
            t = ((w - 1) - x0) / vx
            y = y0 + t * vy
            if 0 <= y < h:
                intersections.append((w - 1, int(round(y))))

        # Intersect with y = 0
        if abs(vy) > eps:
            t = (0 - y0) / vy
            x = x0 + t * vx
            if 0 <= x < w:
                intersections.append((int(round(x)), 0))

        # Intersect with y = h - 1
        if abs(vy) > eps:
            t = ((h - 1) - y0) / vy
            x = x0 + t * vx
            if 0 <= x < w:
                intersections.append((int(round(x)), h - 1))

        # Remove duplicates
        intersections = list(dict.fromkeys(intersections))

        if len(intersections) >= 2:
            p1, p2 = intersections[0], intersections[1]
            ctx.lines = [(p1[0], p1[1], p2[0], p2[1])]

            if "line_fit_image" not in ctx.debug:
                ctx.debug["line_fit_image"] = ctx.frame.copy()

            cv2.line(
                ctx.debug["line_fit_image"],
                (int(p1[0]), int(p1[1])),
                (int(p2[0]), int(p2[1])),
                (0, 255, 0),
                2
            )
        else:
            ctx.lines = []



        return ctx
