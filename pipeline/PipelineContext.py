from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np

@dataclass
class PipelineContext:
    frame: np.ndarray
    detections: Any = None
    masks: Any = None
    binary_mask: Optional[np.ndarray] = None
    distance_heatmap: Optional[np.ndarray] = None
    skeleton: Optional[np.ndarray] = None
    lines: Any = None
    candidates: Any = None
    best_candidate: Any = None
    debug: dict = field(default_factory=dict)