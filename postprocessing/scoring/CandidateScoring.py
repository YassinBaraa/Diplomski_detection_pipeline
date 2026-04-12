import cv2
import numpy as np
from pipeline.Stage import Stage

MIN_DISTANCE_TO_BOUNDARY = 0.5
JUNCTION_NEIGHBOR_THRESHOLD = 3

W_BOUNDARY = 1.0
W_JUNCTION = 0.3
W_ENDPOINT = 0.2

W_TRACK_AGE = 0.05
W_HIT_STREAK = 0.1
W_CONF = 0.2
W_CONSISTENCY = 0.1

MIN_CONFIRM_FRAMES = 3
STABILITY_BUFFER_SIZE = 10  # Keep last 10 frames
STABILITY_THRESHOLD = 5  # Point must appear in 5+ frames within clustering distance
SPATIAL_CLUSTER_RADIUS = 15  # pixels - points within this are considered same location
TRACK_CLEANUP_FRAMES = 30  # Delete track if not seen for this many frames


class CandidateScoring(Stage):
    def __init__(self):
        super().__init__()
        self.track_memory = {}

    def run(self, ctx):
        if ctx.binary_mask is None or ctx.skeleton is None or ctx.distance_heatmap is None or ctx.tracks is None:
            ctx.candidates = []
            ctx.best_candidate = None
            return ctx

        mask = (ctx.binary_mask > 0).astype(np.uint8)
        skeleton = (ctx.skeleton > 0).astype(np.uint8)
        dist = ctx.distance_heatmap.astype(np.float32)

        if mask.sum() == 0 or skeleton.sum() == 0:
            ctx.candidates = []
            ctx.best_candidate = None
            return ctx

        neighbor_count = self._neighbor_count(skeleton)
        junctions = ((skeleton > 0) & (neighbor_count >= JUNCTION_NEIGHBOR_THRESHOLD)).astype(np.uint8)
        endpoints = ((skeleton > 0) & (neighbor_count == 1)).astype(np.uint8)

        junction_penalty_src = np.where(junctions > 0, 0, 1).astype(np.uint8)
        junction_dist = cv2.distanceTransform(junction_penalty_src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        endpoint_penalty_src = np.where(endpoints > 0, 0, 1).astype(np.uint8)
        endpoint_dist = cv2.distanceTransform(endpoint_penalty_src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        all_candidates = []
        best_per_track = []
        seen_track_ids = set()

        debug_skipped_no_skeleton = 0
        debug_skipped_boundary = 0

        for xyxy, mask_i, confidence, class_id, tracker_id, data in ctx.tracks:
            if tracker_id is None:
                continue

            track_id = int(tracker_id)
            seen_track_ids.add(track_id)

            x1, y1, x2, y2 = map(int, xyxy)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(mask.shape[1], x2)
            y2 = min(mask.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            if track_id not in self.track_memory:
                self.track_memory[track_id] = {
                    "age": 0,
                    "hit_streak": 0,
                    "conf_history": [],
                    "score_history": [],
                    "location_history": [],  # Buffer of (x, y, score) tuples
                    "locked_point": None,  # Locked stable point
                    "frames_locked": 0,
                }

            mem = self.track_memory[track_id]
            mem["age"] += 1
            mem["hit_streak"] += 1
            mem["conf_history"].append(float(confidence) if confidence is not None else 0.0)
            mem["conf_history"] = mem["conf_history"][-10:]
            mem["frames_since_last_seen"] = 0

            conf_avg = float(np.mean(mem["conf_history"])) if mem["conf_history"] else 0.0

            # Compute consistency: inverse of coefficient of variation of recent scores
            # Higher consistency score = more stable branch
            if len(mem["score_history"]) >= 3:
                score_mean = np.mean(mem["score_history"])
                score_std = np.std(mem["score_history"])
                # CV = std/mean, consistency = 1/(1+CV) so lower variance = higher consistency
                consistency_score = 1.0 / (1.0 + (score_std / (abs(score_mean) + 1e-6)))
            else:
                consistency_score = 0.5  # Default for new tracks

            ys, xs = np.where(skeleton[y1:y2, x1:x2] > 0)
            if len(xs) == 0:
                debug_skipped_no_skeleton += 1
                continue

            best_track_candidate = None
            best_track_score = -np.inf

            for x_local, y_local in zip(xs, ys):
                x = x_local + x1
                y = y_local + y1

                d_boundary = float(dist[y, x])
                d_junction = float(junction_dist[y, x])
                d_endpoint = float(endpoint_dist[y, x])

                if d_boundary < MIN_DISTANCE_TO_BOUNDARY:
                    debug_skipped_boundary += 1
                    continue

                geom_score = (
                    W_BOUNDARY * d_boundary +
                    W_JUNCTION * d_junction +
                    W_ENDPOINT * d_endpoint
                )

                track_bonus = (
                    W_TRACK_AGE * mem["age"] +
                    W_HIT_STREAK * mem["hit_streak"] +
                    W_CONF * conf_avg +
                    W_CONSISTENCY * consistency_score
                )

                total_score = geom_score + track_bonus

                candidate = {
                    "track_id": track_id,
                    "x": int(x),
                    "y": int(y),
                    "score": float(total_score),
                    "geom_score": float(geom_score),
                    "track_bonus": float(track_bonus),
                    "distance_to_boundary": d_boundary,
                    "distance_to_junction": d_junction,
                    "distance_to_endpoint": d_endpoint,
                    "track_age": int(mem["age"]),
                    "hit_streak": int(mem["hit_streak"]),
                    "confidence_avg": float(conf_avg),
                    "consistency_score": float(consistency_score),
                    "confirmed": bool(mem["hit_streak"] >= MIN_CONFIRM_FRAMES),
                    "bbox": (x1, y1, x2, y2),
                }

                all_candidates.append(candidate)
                mem["score_history"].append(float(total_score))
                mem["score_history"] = mem["score_history"][-10:]

                if total_score > best_track_score:
                    best_track_score = total_score
                    best_track_candidate = candidate

            if best_track_candidate is not None:
                best_per_track.append(best_track_candidate)
                # Add to location history for stability filtering
                mem["location_history"].append((best_track_candidate["x"], best_track_candidate["y"], best_track_candidate["score"]))
                mem["location_history"] = mem["location_history"][-STABILITY_BUFFER_SIZE:]

        for track_id in list(self.track_memory.keys()):
            if track_id not in seen_track_ids:
                self.track_memory[track_id]["hit_streak"] = 0
                self.track_memory[track_id]["frames_since_last_seen"] = self.track_memory[track_id].get("frames_since_last_seen", 0) + 1
                # Unlock point if track disappears for multiple frames
                if self.track_memory[track_id]["frames_since_last_seen"] > 2:
                    self.track_memory[track_id]["locked_point"] = None
                    self.track_memory[track_id]["frames_locked"] = 0
                # Delete track if not seen for TRACK_CLEANUP_FRAMES frames
                if self.track_memory[track_id]["frames_since_last_seen"] > TRACK_CLEANUP_FRAMES:
                    del self.track_memory[track_id]

        confirmed_candidates = [c for c in best_per_track if c["confirmed"]]

        if confirmed_candidates:
            best_candidate = max(confirmed_candidates, key=lambda c: c["score"])
        elif best_per_track:
            best_candidate = max(best_per_track, key=lambda c: c["score"])
        else:
            best_candidate = None

        # Apply temporal stability filtering
        # Only lock in a point if it's been consistent across recent frames
        if best_candidate is not None:
            best_track_id = best_candidate["track_id"]
            if best_track_id in self.track_memory:
                mem = self.track_memory[best_track_id]

                # Validate locked point still exists on skeleton
                if mem["locked_point"] is not None:
                    locked_x, locked_y = mem["locked_point"]
                    # Check if locked point is still on skeleton
                    if 0 <= locked_y < skeleton.shape[0] and 0 <= locked_x < skeleton.shape[1]:
                        if skeleton[locked_y, locked_x] > 0:
                            # Locked point is still valid
                            best_candidate["x"] = locked_x
                            best_candidate["y"] = locked_y
                            best_candidate["is_stable"] = True
                            mem["frames_locked"] += 1
                        else:
                            # Point drifted off skeleton, release lock
                            mem["locked_point"] = None
                            mem["frames_locked"] = 0
                            best_candidate["is_stable"] = False
                    else:
                        # Point out of bounds, release lock
                        mem["locked_point"] = None
                        mem["frames_locked"] = 0
                        best_candidate["is_stable"] = False

                # Try to lock a new point if we don't have one
                elif len(mem["location_history"]) >= STABILITY_THRESHOLD:
                    stable_point = self._find_stable_location(mem["location_history"])
                    if stable_point is not None:
                        stable_x, stable_y = stable_point
                        # Double-check new lock point exists on skeleton
                        if 0 <= stable_y < skeleton.shape[0] and 0 <= stable_x < skeleton.shape[1]:
                            if skeleton[stable_y, stable_x] > 0:
                                # Lock to this point
                                mem["locked_point"] = stable_point
                                mem["frames_locked"] = 1
                                best_candidate["x"] = stable_x
                                best_candidate["y"] = stable_y
                                best_candidate["is_stable"] = True
                            else:
                                best_candidate["is_stable"] = False
                        else:
                            best_candidate["is_stable"] = False
                    else:
                        best_candidate["is_stable"] = False
                else:
                    best_candidate["is_stable"] = False

        # Show best candidates per track instead of all candidate points
        ctx.candidates = best_per_track
        ctx.best_candidate = best_candidate

        ctx.debug["num_candidates"] = len(best_per_track)
        ctx.debug["num_tracks"] = len(seen_track_ids)
        ctx.debug["num_confirmed_tracks"] = len(confirmed_candidates)
        ctx.debug["skipped_no_skeleton"] = debug_skipped_no_skeleton
        ctx.debug["skipped_boundary"] = debug_skipped_boundary

        # Add stability buffer progress for debugging
        if best_candidate is not None:
            best_track_id = best_candidate["track_id"]
            if best_track_id in self.track_memory:
                buffer_progress = min(len(self.track_memory[best_track_id]["location_history"]), STABILITY_THRESHOLD)
                ctx.debug["stability_buffer_progress"] = buffer_progress
        else:
            ctx.debug["stability_buffer_progress"] = 0

        return ctx

    def _neighbor_count(self, skeleton):
        s = (skeleton > 0).astype(np.uint8)
        p = np.pad(s, 1, mode="constant", constant_values=0)
        neighbor_count = (
            p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
            p[1:-1, :-2]               + p[1:-1, 2:] +
            p[2:, :-2]  + p[2:, 1:-1]  + p[2:, 2:]
        )
        return neighbor_count

    def _find_stable_location(self, location_history):
        """Find the most stable (most common) location cluster in history.

        Returns the median position of the largest cluster, or None if no stable cluster found.
        """
        if len(location_history) < STABILITY_THRESHOLD:
            return None

        locations = np.array([(x, y) for x, y, _ in location_history])

        # Find clusters using simple greedy approach
        clusters = []
        used = set()

        for i, loc in enumerate(locations):
            if i in used:
                continue

            # Start new cluster
            cluster = [loc]
            used.add(i)

            # Add all nearby points to cluster
            for j, other_loc in enumerate(locations):
                if j not in used:
                    dist = np.sqrt((loc[0] - other_loc[0])**2 + (loc[1] - other_loc[1])**2)
                    if dist <= SPATIAL_CLUSTER_RADIUS:
                        cluster.append(other_loc)
                        used.add(j)

            clusters.append(cluster)

        # Find largest cluster
        if not clusters:
            return None

        largest_cluster = max(clusters, key=len)

        # Only accept if cluster has sufficient size
        if len(largest_cluster) < STABILITY_THRESHOLD:
            return None

        # Return median position of cluster
        cluster_array = np.array(largest_cluster)
        median_x = int(np.median(cluster_array[:, 0]))
        median_y = int(np.median(cluster_array[:, 1]))

        return (median_x, median_y)