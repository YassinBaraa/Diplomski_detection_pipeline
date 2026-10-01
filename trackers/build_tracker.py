from trackers.ByteTrack import ByteTrack
from trackers.TrackCoaster import TrackCoaster


def build_tracker(detector, cfg):
    """cfg: the `tracking` section of the config."""
    tracker = ByteTrack(
        detector,
        track_activation_threshold=cfg["track_activation_threshold"],
        lost_track_buffer=cfg["lost_track_buffer"],
        minimum_consecutive_frames=cfg["minimum_consecutive_frames"],
    )
    if cfg["coast_frames"]:
        tracker = TrackCoaster(tracker, max_coast=cfg["coast_frames"])
    return tracker
