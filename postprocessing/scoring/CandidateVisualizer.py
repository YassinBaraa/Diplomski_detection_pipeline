import cv2
import numpy as np
import logging
from pipeline.Stage import Stage

logger = logging.getLogger(__name__)


class CandidateVisualizer(Stage):
    def run(self, ctx):
        
        if ctx.frame is None:
            return ctx

        try:
            vis = cv2.resize(ctx.frame.copy(), (640, 384))
            if vis.ndim == 2:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

            # Calculate scale factors from ORIGINAL frame to resized display
            frame_h, frame_w = ctx.frame.shape[:2]
            vis_h, vis_w = vis.shape[:2]
            scale_x = vis_w / frame_w if frame_w > 0 else 1.0
            scale_y = vis_h / frame_h if frame_h > 0 else 1.0

            # Draw binary mask as semi-transparent green overlay
            if ctx.binary_mask is not None:
                mask_display = cv2.resize(ctx.binary_mask, (vis_w, vis_h), interpolation=cv2.INTER_NEAREST)
                green_overlay = np.zeros_like(vis)
                green_overlay[mask_display > 0] = [0, 180, 0]
                vis = cv2.addWeighted(vis, 1.0, green_overlay, 0.35, 0)

            # Validate that skeleton matches frame dimensions
            display_skeleton = ctx.skeleton
            if ctx.skeleton is not None:
                skel_h, skel_w = ctx.skeleton.shape[:2]
                if (skel_h, skel_w) != (frame_h, frame_w):
                    logger.warning(f"Skeleton size mismatch: frame ({frame_h}x{frame_w}) vs skeleton ({skel_h}x{skel_w})")
                    # Resize skeleton copy for display only, don't modify ctx.skeleton
                    display_skeleton = cv2.resize(ctx.skeleton, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

            # Draw skeleton for debugging
            if display_skeleton is not None:
                ys_sk, xs_sk = np.where(display_skeleton > 0)
                if len(xs_sk) > 0:
                    sample_rate = max(1, len(xs_sk) // 5000)
                    for i in range(0, len(xs_sk), sample_rate):
                        x_sk, y_sk = xs_sk[i], ys_sk[i]
                        x_scaled = int(x_sk * scale_x)
                        y_scaled = int(y_sk * scale_y)
                        if 0 <= x_scaled < vis_w and 0 <= y_scaled < vis_h:
                            vis[y_scaled, x_scaled] = [100, 100, 100]  # Dark gray for skeleton
            if ctx.candidates:
                for cand in ctx.candidates:
                    try:
                        x = int(cand["x"] * scale_x)
                        y = int(cand["y"] * scale_y)

                        # Clamp to visualization bounds
                        x = max(0, min(x, vis_w - 1))
                        y = max(0, min(y, vis_h - 1))

                        score = cand["score"]
                        cv2.circle(vis, (x, y), 5, (255, 0, 0), 2)
                        cv2.putText(vis, f"{score:.1f}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                    except KeyError as e:
                        logger.warning(f"Candidate missing field: {e}")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid candidate data: {e}")

            # Draw best candidate
            if ctx.best_candidate:
                try:
                    
                    x = int(ctx.best_candidate["x"] * scale_x)
                    y = int(ctx.best_candidate["y"] * scale_y)
                    x = max(0, min(x, vis_w - 1))
                    y = max(0, min(y, vis_h - 1))

                    r = max(1, int(ctx.best_candidate.get("distance_to_boundary", 5) * scale_x))
                    cv2.circle(vis, (x, y), 8, (0, 0, 255), 3)
                    cv2.circle(vis, (x, y), r, (0, 255, 255), 2)

                    text = f"Best: {ctx.best_candidate['score']:.2f}"
                    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                except KeyError as e:
                    logger.warning(f"Best candidate missing field: {e}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid best candidate data: {e}")

            # Draw final point from warmup
            if ctx.final_point is not None:
                try:
                    x = int(ctx.final_point["x"] * scale_x)
                    y = int(ctx.final_point["y"] * scale_y)
                    x = max(0, min(x, vis_w - 1))
                    y = max(0, min(y, vis_h - 1))
                    
                    # Draw larger circle for final point
                    cv2.circle(vis, (x, y), 12, (255, 0, 255), 3)  # Magenta for final point
                    occurrences = ctx.final_point.get("occurrences", 0)
                    total = ctx.final_point.get("total_observations", 0)
                    text = f"FINAL: ({x},{y}) [{occurrences}/{total}]"
                    cv2.putText(vis, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Error drawing final point: {e}")

            # Draw statistics
            try:
                y_pos = 60
                num_cands = ctx.debug.get("num_candidates", 0)
                num_tracks = ctx.debug.get("num_tracks", 0)
                num_confirmed = ctx.debug.get("num_confirmed_tracks", 0)
                skipped_skeleton = ctx.debug.get("skipped_no_skeleton", 0)
                skipped_boundary = ctx.debug.get("skipped_boundary", 0)
                warmup_prog = ctx.debug.get("warmup_progress", "")

                cv2.putText(vis, f"Candidates: {num_cands}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(vis, f"Tracks: {num_tracks}", (10, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(vis, f"Confirmed: {num_confirmed}", (10, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(vis, f"Skipped (skeleton): {skipped_skeleton}", (10, y_pos + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 200), 1)
                cv2.putText(vis, f"Skipped (boundary): {skipped_boundary}", (10, y_pos + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 200), 1)
                if warmup_prog:
                    cv2.putText(vis, f"Warmup: {warmup_prog}", (10, y_pos + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 100), 1)
            except (ValueError, TypeError) as e:
                logger.warning(f"Error drawing statistics: {e}")

            ctx.debug["branch_score_image"] = vis

        except Exception as e:
            logger.error(f"CandidateVisualizer critical error: {e}", exc_info=True)
            ctx.debug["branch_score_image"] = ctx.frame

        return ctx