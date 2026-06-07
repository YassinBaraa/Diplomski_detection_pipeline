import sys
import cv2
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from sources.MP4Source import MP4Source
from detectors.YOLOBranchSeg import YOLOBranchSeg
from trackers.ByteTrack import ByteTrack
from postprocessing.PostProcessor import PostProcessor
from postprocessing.masks.MaskExtraction import MaskExtraction
from postprocessing.geometry.DistanceHeatmap import DistanceHeatmap
from postprocessing.geometry.BitmaskSkeleton import BitmaskSkeleton
from postprocessing.scoring.CandidateScoring import CandidateScoring
from postprocessing.scoring.WarmupFinalPoint import WarmupFinalPoint
from postprocessing.scoring.CandidateVisualizer import CandidateVisualizer
from pipeline.DetectionPipeline import DetectionPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUT_DIR     = _HERE / "detection_test"
EXAMPLE_MP4 = _HERE / "sources" / "example.mp4"
YOLO_PATH   = str(_HERE / "model" / "best_small.pt")
CONF        = 0.4
VIS_W       = 640
VIS_H       = 384
MP4_FPS     = 12
NICLA_FPS   = 10


def make_output_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUT_DIR}")


def add_label(frame, text):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    cv2.putText(frame, text, (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, ts, (frame.shape[1] - 145, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    return frame


def create_writer(path, w, h, fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(str(path), fourcc, fps, (w, h))


def vis_stage1_mask(ctx):
    if ctx.binary_mask is None:
        return np.zeros((VIS_H, VIS_W, 3), dtype=np.uint8)
    vis = cv2.cvtColor(ctx.binary_mask, cv2.COLOR_GRAY2BGR)
    return vis


def vis_stage2_heatmap(ctx):
    if ctx.distance_heatmap is None:
        return np.zeros((VIS_H, VIS_W, 3), dtype=np.uint8)
    heat_uint8 = (ctx.distance_heatmap * 255).astype(np.uint8)
    vis = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    return vis


def vis_stage3_skeleton(ctx):
    if ctx.skeleton is None:
        return np.zeros((VIS_H, VIS_W, 3), dtype=np.uint8)
    vis = np.zeros((*ctx.skeleton.shape[:2], 3), dtype=np.uint8)
    vis[ctx.skeleton > 0] = [255, 255, 255]
    return vis


def vis_stage4_candidates(ctx):
    h = ctx.frame.shape[0] if ctx.frame is not None else VIS_H
    w = ctx.frame.shape[1] if ctx.frame is not None else VIS_W
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    if ctx.candidates:
        for cand in ctx.candidates:
            try:
                cv2.circle(vis, (int(cand["x"]), int(cand["y"])), 5, (255, 0, 0), 2)
                cv2.putText(vis, f"{cand['score']:.2f}", (int(cand["x"]) - 10, int(cand["y"]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            except (KeyError, ValueError, TypeError):
                pass

    if ctx.best_candidate:
        try:
            bx, by = int(ctx.best_candidate["x"]), int(ctx.best_candidate["y"])
            r = max(1, int(ctx.best_candidate.get("distance_to_boundary", 5)))
            cv2.circle(vis, (bx, by), 8, (0, 0, 255), 3)
            cv2.circle(vis, (bx, by), r, (0, 255, 255), 2)
            cv2.putText(vis, f"Best: {ctx.best_candidate['score']:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        except (KeyError, ValueError, TypeError):
            pass

    return vis


def vis_stage5_full(ctx):
    if "branch_score_image" in ctx.debug:
        return ctx.debug["branch_score_image"].copy()
    return ctx.frame.copy() if ctx.frame is not None else np.zeros((VIS_H, VIS_W, 3), dtype=np.uint8)


def run_mp4_stage(idx, name, stages, vis_fn, detector):
    out_path = OUT_DIR / f"mp4_stage_{idx}_{name}.mp4"
    source = MP4Source(str(EXAMPLE_MP4))
    tracker = ByteTrack(detector)
    postprocessor = PostProcessor(stages)
    pipeline = DetectionPipeline(source, detector, tracker, postprocessor)
    writer = create_writer(out_path, VIS_W, VIS_H, MP4_FPS)
    frame_count = 0
    try:
        for ctx in pipeline.run():
            vis = vis_fn(ctx)
            vis = cv2.resize(vis, (VIS_W, VIS_H))
            add_label(vis, f"Stage {idx}: {name}")
            writer.write(vis)
            frame_count += 1
    finally:
        writer.release()
        source.release()
    logger.info(f"Stage {idx} ({name}): {frame_count} frames saved → {out_path}")


def run_nicla_phase(detector):
    out_path = OUT_DIR / "nicla_full_pipeline.mp4"
    source = NiclaSource()
    tracker = ByteTrack(detector)
    postprocessor = PostProcessor([
        MaskExtraction(), DistanceHeatmap(), BitmaskSkeleton(),
        CandidateScoring(), WarmupFinalPoint(), CandidateVisualizer(),
    ])
    pipeline = DetectionPipeline(source, detector, tracker, postprocessor)
    writer = None
    frame_count = 0
    try:
        for ctx in pipeline.run():
            vis = ctx.debug.get("branch_score_image", ctx.frame).copy()
            add_label(vis, "Nicla Full Pipeline")
            if writer is None:
                h, w = vis.shape[:2]
                writer = create_writer(out_path, w, h, NICLA_FPS)
            writer.write(vis)
            frame_count += 1
    except KeyboardInterrupt:
        logger.info("Nicla phase stopped by user (Ctrl+C)")
    finally:
        if writer is not None:
            writer.release()
        source.release()
    logger.info(f"Nicla phase: {frame_count} frames saved → {out_path}")


def main():
    make_output_dir()

    STAGE_CONFIGS = [
        (1, "MaskExtraction",   [MaskExtraction()],                                          vis_stage1_mask),
        (2, "DistanceHeatmap",  [MaskExtraction(), DistanceHeatmap()],                       vis_stage2_heatmap),
        (3, "BitmaskSkeleton",  [MaskExtraction(), DistanceHeatmap(), BitmaskSkeleton()],    vis_stage3_skeleton),
        (4, "CandidateScoring", [MaskExtraction(), DistanceHeatmap(), BitmaskSkeleton(),
                                  CandidateScoring()],                                        vis_stage4_candidates),
        (5, "FullPipeline",     [MaskExtraction(), DistanceHeatmap(), BitmaskSkeleton(),
                                  CandidateScoring(), WarmupFinalPoint(), CandidateVisualizer()], vis_stage5_full),
    ]

    detector = YOLOBranchSeg(YOLO_PATH, conf=CONF)
    for idx, name, stages, vis_fn in STAGE_CONFIGS:
        logger.info(f"--- Stage {idx}: {name} ---")
        run_mp4_stage(idx, name, stages, vis_fn, detector)

    logger.info("All recordings complete.")


if __name__ == "__main__":
    main()
