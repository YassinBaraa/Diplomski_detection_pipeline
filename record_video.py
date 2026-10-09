"""Record raw frames from the configured source to an MP4 for offline testing.

Usage:
    python record_video.py                     # record until 'q' / Ctrl+C
    python record_video.py --duration 30       # stop after 30 s
    python record_video.py --no-preview        # headless (e.g. over SSH)
    python record_video.py --config my.yaml    # other config file

The source type comes from `source.type` in the config (PIPELINE_SOURCE_TYPE
env var overrides it). Output goes to recordings/<source>_<timestamp>.mp4 and
can be played back through the pipeline with source.type: "mp4".
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from config import ConfigManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUT_DIR = _HERE / "recordings"
# Frames buffered before opening the writer, so the file's fps matches the
# rate the source actually delivers instead of the nominal one.
FPS_PROBE_FRAMES = 30


def make_source(cfg):
    """Build the frame source named by source.type. Imports are lazy so a
    missing optional dependency (picamera2, serial) only matters if used."""
    source_type = cfg.get("source.type")
    if source_type == "dsj":
        from sources.DSJSource import DSJSource
        return DSJSource(
            width=cfg.get("dsj.resolution_width", 1280),
            height=cfg.get("dsj.resolution_height", 720),
            fps=cfg.get("dsj.fps", 30),
        )
    if source_type == "mp4":
        from sources.MP4Source import MP4Source
        return MP4Source(str(_HERE / cfg.get("source.video_path")))
    if source_type == "camera":
        from sources.CameraSource import CameraSource
        return CameraSource()
    if source_type == "nicla":
        from sources.NiclaSource import NiclaSource
        return NiclaSource(port=cfg.get("source.nicla_serial_port", "/dev/ttyACM0"))
    raise ValueError(f"Unknown source.type {source_type!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to YAML config (default: config/default_config.yaml)")
    parser.add_argument("--duration", type=float, default=None, help="Stop after this many seconds")
    parser.add_argument("--no-preview", action="store_true", help="Don't open a preview window")
    parser.add_argument("--output", default=None, help="Output .mp4 path (default: recordings/<source>_<timestamp>.mp4)")
    args = parser.parse_args()

    cfg = ConfigManager(args.config)
    source_type = cfg.get("source.type")
    source = make_source(cfg)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = OUT_DIR / f"{source_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    preview = not args.no_preview and bool(os.environ.get("DISPLAY"))
    if not args.no_preview and not preview:
        logger.info("No DISPLAY found, recording without preview")

    logger.info(f"Recording from {source_type!r} -> {out_path}  (stop with 'q' or Ctrl+C)")

    # A file source is read as fast as possible, so measuring would give a
    # meaningless rate -- use the file's own fps. Live sources get measured.
    fixed_fps = source.cap.get(cv2.CAP_PROP_FPS) if source_type == "mp4" else None
    probe_frames = 1 if fixed_fps else FPS_PROBE_FRAMES

    writer = None
    probe = []
    frame_count = 0
    fps = None
    start = time.monotonic()
    try:
        while True:
            # Sources disagree on arity (MP4Source: ok, frame; DSJSource: ok, frame, extra)
            ok, frame = source.read()[:2]
            if not ok:
                logger.info("Source ended")
                break

            if writer is None:
                probe.append(frame)
                if len(probe) >= probe_frames:
                    fps = fixed_fps or len(probe) / (time.monotonic() - start)
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    logger.info(f"Measured {fps:.1f} fps at {w}x{h}, writer opened")
                    for f in probe:
                        writer.write(f)
                    probe = []
            else:
                writer.write(frame)
            frame_count += 1

            if preview:
                cv2.imshow("Recording", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User quit")
                    break

            if args.duration is not None and time.monotonic() - start >= args.duration:
                logger.info(f"Reached {args.duration:.0f} s")
                break

            if frame_count % 100 == 0:
                logger.info(f"{frame_count} frames")
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        # Short recording that never filled the probe buffer: flush it anyway.
        if writer is None and probe:
            fps = fixed_fps or len(probe) / max(time.monotonic() - start, 1e-6)
            h, w = probe[0].shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            for f in probe:
                writer.write(f)
        if writer is not None:
            writer.release()
        source.release()
        if preview:
            cv2.destroyAllWindows()

    elapsed = time.monotonic() - start
    if frame_count:
        logger.info(f"Saved {frame_count} frames ({elapsed:.1f} s, {fps:.1f} fps) -> {out_path}")
    else:
        logger.warning("No frames captured, nothing saved")


if __name__ == "__main__":
    main()
