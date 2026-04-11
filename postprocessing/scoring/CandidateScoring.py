import cv2
import numpy as np
from pipeline.Stage import Stage


MIN_DISTANCE_TO_BOUNDARY = 3.0
JUNCTION_NEIGHBOR_THRESHOLD = 3

W_BOUNDARY = 1.0
W_JUNCTION = 0.7
W_BRANCH = 0.5


class CandidateScoring(Stage):
    def run(self, ctx):
        if ctx.binary_mask is None or ctx.skeleton is None:
            ctx.candidates = []
            ctx.best_candidate = None
            return ctx

        mask = (ctx.binary_mask > 0).astype(np.uint8)
        skeleton = (ctx.skeleton > 0).astype(np.uint8)

        if mask.sum() == 0 or skeleton.sum() == 0:
            ctx.candidates = []
            ctx.best_candidate = None
            return ctx

        # Distance to nearest boundary
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        ctx.distance_heatmap = dist

        # Junction / endpoint detection (assumes you have a working _neighbor_count)
        neighbor_count = self._neighbor_count(skeleton)
        junctions = ((skeleton > 0) & (neighbor_count >= JUNCTION_NEIGHBOR_THRESHOLD)).astype(np.uint8)
        endpoints = ((skeleton > 0) & (neighbor_count == 1)).astype(np.uint8)

        # Distance to junctions and endpoints
        junction_penalty_src = np.where(junctions > 0, 0, 1).astype(np.uint8)
        junction_dist = cv2.distanceTransform(junction_penalty_src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        endpoint_penalty_src = np.where(endpoints > 0, 0, 1).astype(np.uint8)
        endpoint_dist = cv2.distanceTransform(endpoint_penalty_src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        ys, xs = np.where(skeleton > 0)

        if len(xs) == 0:
            ctx.candidates = []
            ctx.best_candidate = None
            return ctx

        candidates = []
        best_candidate = None
        best_score = -np.inf

        for x, y in zip(xs, ys):
            d_boundary = float(dist[y, x])
            d_junction = float(junction_dist[y, x])
            d_endpoint = float(endpoint_dist[y, x])

            if d_boundary < MIN_DISTANCE_TO_BOUNDARY:
                continue

            score = (
                W_BOUNDARY * d_boundary +
                W_JUNCTION * d_junction +
                W_BRANCH * d_endpoint
            )

            candidate = {
                "x": int(x),
                "y": int(y),
                "score": float(score),
                "distance_to_boundary": d_boundary,
                "distance_to_junction": d_junction,
                "distance_to_endpoint": d_endpoint,
                "is_junction": bool(junctions[y, x]),
                "is_endpoint": bool(endpoints[y, x]),
            }
            candidates.append(candidate)

            if score > best_score:
                best_score = score
                best_candidate = candidate

        ctx.candidates = candidates
        ctx.best_candidate = best_candidate

        # === ONLY VISUALIZATION ADDED FROM HERE ===
        vis = cv2.resize(ctx.frame.copy(), (640, 384))
        if vis.ndim == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        # Skeleton: green
        if ctx.skeleton is not None:
            ys_sk, xs_sk = np.where(ctx.skeleton > 0)
            for x_sk, y_sk in zip(xs_sk, ys_sk):
                cv2.circle(vis, (int(x_sk), int(y_sk)), 1, (0, 255, 0), -1)

        # Junctions: yellow
        if junctions is not None:
            ys_j, xs_j = np.where(junctions > 0)
            for x_j, y_j in zip(xs_j, ys_j):
                cv2.circle(vis, (int(x_j), int(y_j)), 3, (0, 255, 255), -1)

        # Endpoints: cyan
        if endpoints is not None:
            ys_e, xs_e = np.where(endpoints > 0)
            for x_e, y_e in zip(xs_e, ys_e):
                cv2.circle(vis, (int(x_e), int(y_e)), 3, (255, 255, 0), -1)

        # Best candidate: red + blue circle (clearance)
        if ctx.best_candidate is not None:
            x = ctx.best_candidate["x"]
            y = ctx.best_candidate["y"]
            r = int(round(ctx.best_candidate["distance_to_boundary"]))

            cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
            cv2.circle(vis, (x, y), r, (255, 0, 0), 2)

        ctx.debug["branch_score_image"] = vis
        # ==========================================

        return ctx


    def _neighbor_count(self, skeleton):
        s = (skeleton > 0).astype(np.uint8)
        p = np.pad(s, 1, mode="constant", constant_values=0)

        neighbor_count = (
            p[:-2, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] + p[2:, 1:-1]
        )

        return neighbor_count