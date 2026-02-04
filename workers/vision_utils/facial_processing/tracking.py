import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class TrackedFace:
    # a single face being tracked over time
    track_id : int
    bbox : Tuple[int, int, int, int]

    identity: str = "Unknown"
    identity_score: float = 0.0
    latest_embedding: Optional[np.ndarray]=None
    
    frames_since_recognition: int = 0
    frames_unseen: int = 0
    is_confirmed: bool = False

    def update_pos(self, new_bbox):
        self.bbox = new_bbox
        self.frames_unseen = 0
        self.frames_since_recognition += 1


def calculate_iou(boxA, boxB):
    #given two rectangles, how much overlap
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    intersect_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = intersect_area / float(boxAArea + boxBArea - intersect_area)

    return iou

