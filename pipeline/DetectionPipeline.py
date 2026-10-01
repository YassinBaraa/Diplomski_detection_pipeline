from pipeline.PipelineContext import PipelineContext
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


class DetectionPipeline:
    def __init__(self, source, detector, tracker, postprocessor=None):
        self.source = source
        self.detector = detector
        self.tracker = tracker
        self.postprocessor = postprocessor

    def run(self):
        frame_n = 0
        while True:
            t_read0 = time.monotonic()
            read_result = self.source.read()
            t_read1 = time.monotonic()
            if not isinstance(read_result, tuple):
                raise ValueError("FrameSource.read() must return a tuple")
            if len(read_result) == 2:
                ret, frame = read_result
                source_metadata = None
            elif len(read_result) == 3:
                ret, frame, source_metadata = read_result
            else:
                raise ValueError("FrameSource.read() must return 2 or 3 values")

            if not ret or frame is None:
                print("[DetectionPipeline] Source exhausted — stopping")
                break

            frame_n += 1
            print(f"[DetectionPipeline] Frame {frame_n}: read OK, shape={frame.shape[:2]}")

            ctx = PipelineContext(frame=frame, source_metadata=source_metadata)

            try:
                t_det0 = time.monotonic()
                prediction = self.detector.predict(frame)
                t_det1 = time.monotonic()
                if isinstance(prediction, tuple):
                    if len(prediction) == 4:
                        ctx.results, ctx.detections, ctx.masks, ctx.confidences = prediction
                        ctx.class_ids = None
                    elif len(prediction) == 5:
                        ctx.results, ctx.detections, ctx.masks, ctx.confidences, ctx.class_ids = prediction
                    else:
                        raise ValueError("Detector.predict() must return 4 or 5 values")
                else:
                    ctx.results = prediction
                    ctx.detections = None
                    ctx.masks = None
                    ctx.confidences = None
                    ctx.class_ids = None

                n_det = len(ctx.detections) if ctx.detections is not None and len(ctx.detections) > 0 else 0
                print(f"[DetectionPipeline] Frame {frame_n}: detector found {n_det} detections")

                t_track0 = time.monotonic()
                track_result = self.tracker.track(
                    frame,
                    results=ctx.results,
                    bboxes=ctx.detections,
                    masks=ctx.masks,
                    confidences=ctx.confidences,
                    class_ids=ctx.class_ids,
                )
                t_track1 = time.monotonic()

                if (
                    isinstance(track_result, tuple)
                    and len(track_result) == 2
                    and isinstance(track_result[0], list)
                ):
                    ctx.tracks, ctx.sv_detections = track_result
                else:
                    ctx.tracks = track_result
                    ctx.sv_detections = None

                # Keep only the masks of detections the tracker confirmed, so
                # low-confidence detections don't leak into the binary mask.
                if ctx.sv_detections is not None:
                    ctx.masks = ctx.sv_detections.mask

                n_tracks = len(ctx.tracks) if ctx.tracks else 0
                print(f"[DetectionPipeline] Frame {frame_n}: tracker has {n_tracks} active tracks")

                t_post0 = time.monotonic()
                if self.postprocessor is not None:
                    ctx = self.postprocessor.process(ctx)
                    fp = getattr(ctx, "final_point", None)
                    bc = getattr(ctx, "best_candidate", None)
                    print(f"[DetectionPipeline] Frame {frame_n}: postprocessor — "
                          f"final_point={fp}, best_candidate={bc}")
                t_post1 = time.monotonic()

                print(f"[DetectionPipeline] Frame {frame_n}: TIMING — "
                      f"read={1000*(t_read1-t_read0):.0f}ms "
                      f"detect={1000*(t_det1-t_det0):.0f}ms "
                      f"track={1000*(t_track1-t_track0):.0f}ms "
                      f"postprocess={1000*(t_post1-t_post0):.0f}ms "
                      f"total={1000*(t_post1-t_read0):.0f}ms")

            except Exception as e:
                logger.error(f"Pipeline error processing frame: {e}", exc_info=True)
                ctx.candidates = []
                ctx.best_candidate = None

            yield ctx