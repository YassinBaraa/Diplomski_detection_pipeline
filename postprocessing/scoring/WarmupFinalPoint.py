import numpy as np
import time
from pipeline.Stage import Stage
from scipy.cluster.hierarchy import fclusterdata


WARMUP_DURATION_SEC = 20
CLUSTER_DISTANCE_THRESHOLD = 30  # pixels


class WarmupFinalPoint(Stage):
    
    def __init__(self):
        self.start_time = None
        self.warmup_complete = False
        self.candidate_history = []  # List of (x, y) tuples
    
    def run(self, ctx):
        # Initialize start time on first frame
        if self.start_time is None:
            self.start_time = time.time()
        
        elapsed = time.time() - self.start_time
        
        # Collect best candidate during warmup
        if not self.warmup_complete and elapsed < WARMUP_DURATION_SEC:
            if ctx.best_candidate is not None:
                x = ctx.best_candidate["x"]
                y = ctx.best_candidate["y"]
                self.candidate_history.append((x, y))
            
            ctx.debug["warmup_progress"] = f"{elapsed:.1f}/{WARMUP_DURATION_SEC}s"
            ctx.final_point = None
            
        # Process warmup results after duration
        elif not self.warmup_complete and elapsed >= WARMUP_DURATION_SEC:
            self.warmup_complete = True
            
            if len(self.candidate_history) > 0:
                # Cluster nearby points
                points = np.array(self.candidate_history)
                
                # Use hierarchical clustering to group nearby points
                clusters = fclusterdata(
                    points,
                    t=CLUSTER_DISTANCE_THRESHOLD,
                    criterion='distance',
                    method='complete'
                )
                
                # Find cluster with most hits
                unique_clusters, counts = np.unique(clusters, return_counts=True)
                best_cluster_idx = unique_clusters[np.argmax(counts)]
                best_cluster_points = points[clusters == best_cluster_idx]
                
                # Calculate cluster center
                center_x = int(np.mean(best_cluster_points[:, 0]))
                center_y = int(np.mean(best_cluster_points[:, 1]))
                
                ctx.final_point = {
                    "x": center_x,
                    "y": center_y,
                    "occurrences": int(np.max(counts)),
                    "total_observations": len(self.candidate_history),
                }

            else:
                ctx.final_point = None
            
            ctx.debug["warmup_progress"] = "Complete"
        
        # After warmup, keep final_point set
        if self.warmup_complete and ctx.final_point is not None:
            ctx.debug["warmup_progress"] = "Complete"
        
        return ctx
