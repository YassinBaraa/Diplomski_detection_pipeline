import numpy as np
from pipeline.Stage import Stage
from scipy.cluster.hierarchy import fclusterdata


WARMUP_FRAMES = 50  # frames to observe before locking final point (~5s at 10fps)
CLUSTER_DISTANCE_THRESHOLD = 30  # pixels


class WarmupFinalPoint(Stage):

    def __init__(self):
        self.frame_count = 0
        self.warmup_complete = False
        self.candidate_history = []  # List of (x, y) tuples
        self._final_point = None     # Persists across frames after warmup
        self._reference_frame = None # Frame snapshot when final_point was locked

    def run(self, ctx):
        self.frame_count += 1

        # Collect best candidate during warmup
        if not self.warmup_complete and self.frame_count <= WARMUP_FRAMES:
            if ctx.best_candidate is not None:
                x = ctx.best_candidate["x"]
                y = ctx.best_candidate["y"]
                self.candidate_history.append((x, y))

            ctx.debug["warmup_progress"] = f"{self.frame_count}/{WARMUP_FRAMES} frames"
            ctx.final_point = None

        # Process warmup results after frame count reached
        elif not self.warmup_complete and self.frame_count > WARMUP_FRAMES:
            self.warmup_complete = True

            if len(self.candidate_history) == 1:
                # Only one observation — use it directly
                center_x, center_y = int(self.candidate_history[0][0]), int(self.candidate_history[0][1])
                self._final_point = {
                    "x": center_x,
                    "y": center_y,
                    "occurrences": 1,
                    "total_observations": 1,
                }
                self._reference_frame = ctx.frame.copy()
            elif len(self.candidate_history) > 1:
                points = np.array(self.candidate_history)

                clusters = fclusterdata(
                    points,
                    t=CLUSTER_DISTANCE_THRESHOLD,
                    criterion='distance',
                    method='complete'
                )

                unique_clusters, counts = np.unique(clusters, return_counts=True)
                best_cluster_idx = unique_clusters[np.argmax(counts)]
                best_cluster_points = points[clusters == best_cluster_idx]

                center_x = int(np.mean(best_cluster_points[:, 0]))
                center_y = int(np.mean(best_cluster_points[:, 1]))

                self._final_point = {
                    "x": center_x,
                    "y": center_y,
                    "occurrences": int(np.max(counts)),
                    "total_observations": len(self.candidate_history),
                }
                self._reference_frame = ctx.frame.copy()

            ctx.debug["warmup_progress"] = "Complete"

        # Re-apply persisted final_point every frame after warmup
        if self.warmup_complete:
            ctx.final_point = self._final_point
            ctx.reference_frame = self._reference_frame
            ctx.debug["warmup_progress"] = "Complete"
        
        return ctx
