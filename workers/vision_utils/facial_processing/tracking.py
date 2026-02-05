from dataclasses import dataclass, field
from typing import List, Tuple
from collections import deque

# import numpy as np

HISTORY_SIZE = 20

@dataclass
class TrackedFace:
    # a single face being tracked over time
    track_id: int
    bbox: Tuple[int, int, int, int]

    identity: str = "Unknown"
    is_confirmed: bool = False

    # latest_embedding: Optional[np.ndarray] = None

    #stores confidence score of current identity for the last N frames
    score_history: deque = field(default_factory = lambda: deque(maxlen=HISTORY_SIZE)) 
    score_sum: float = 0.0

    frames_since_recognition: int = 0
    frames_unseen: int = 0
    
    @property
    def average_score(self)->float:
        if not self.score_history:
            return 0.0
        return self.score_sum/len(self.score_history)

    def update_pos(self, new_bbox):
        self.bbox = new_bbox
        self.frames_unseen = 0
        self.frames_since_recognition += 1

    def update_identity_score(self, identity_score):
        self.score_history.append(identity_score)
        if len(self.score_history)==HISTORY_SIZE:
            self.score_sum-=self.score_history.popleft()
        self.score_sum+=identity_score
    
    def update_identity(self, new_identity = "Unknown", is_confirmed = False, identity_score = 0.0):
        self.identity = new_identity
        self.is_confirmed=is_confirmed
        self.score_history.clear()
        self.score_history.append(identity_score)



def calculate_iou(boxA, boxB):
    # given two rectangles, how much overlap
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    intersect_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = intersect_area / float(boxAArea + boxBArea - intersect_area)

    return iou


class FaceTracker:
    def __init__(self, iou_threshold=0.3, untracked_before_delete=10):
        self.next_obj_id = 0
        self.tracks: List[TrackedFace] = []
        self.iou_threshold = iou_threshold
        self.untracked_before_delete = untracked_before_delete

    def update(self, detected_bboxes: List[Tuple[int, int, int, int]]):
        # updates matching boxes & return list of viable faces

        if not self.tracks:
            for box in detected_bboxes:
                self._register(box)
            return self.tracks

        used_detection_indices = set()

        # loop through self track to find best match in new bboxes
        for track in self.tracks:
            best_iou = 0
            best_match_id = -1

            for d_id, bbox in enumerate(detected_bboxes):
                if d_id in used_detection_indices:
                    continue
                iou = calculate_iou(track.bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = d_id

            if best_iou > self.iou_threshold:
                track.update_pos(detected_bboxes[best_match_id])
                used_detection_indices.add(best_match_id)
            else:
                track.frames_unseen += 1

        # if unused, register as new
        for d_id, bbox in enumerate(detected_bboxes):
            if d_id not in used_detection_indices:
                self._register(bbox)

        # remove the ones without current match for too long
        self.tracks = [
            t for t in self.tracks if t.frames_unseen <= self.untracked_before_delete
        ]

        return self.tracks

    def _register(self, bbox_):
        self.tracks.append(TrackedFace(track_id=self.next_obj_id, bbox=bbox_))
        self.next_obj_id += 1
