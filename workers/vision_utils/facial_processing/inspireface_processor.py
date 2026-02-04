import inspireface as isf
from inspireface import FaceInformation

import numpy as np


class InspireFaceProcessor:
    def __init__(
        self, model_path="Megatron", confidence_threshold=0.5, download_model=False
    ):
        self.model_path = model_path
        self.session = None
        self.known_faces = {} # list for now; we can remove this and make it a db later if known faces is to grow larger

        self._initialize_model(confidence_threshold, download_model)

    def _initialize_model(self, confidence_threshold, download_model):
        # set up inspireface session
        isf.ignore_check_latest_model(download_model)
        ret = isf.launch(self.model_path)
        if not ret:
            raise RuntimeError("InspireFace launch failed; check given model/path")

        params = isf.SessionCustomParameter(
            enable_recognition=True,
            enable_face_emotion=True,
            enable_face_attribute=False,  # age & gender & whatnot
            enable_liveness=False,  # interesting parameter: to differentiate a physical picture of someone vs real person
        )
        self.session = isf.InspireFaceSession(params)
        self.session.set_detection_confidence_threshold(confidence_threshold)
        print("[Vision] InspireFace Model initialized")

    def detect_faces(self, image: np.ndarray):
        return self.session.face_detection(image)
    
    def extract_embedding(self, image: np.ndarray, face_obj: FaceInformation):
        return self.session.face_feature_extract(image, face_obj)
    
    def identify_embedding(self, embedding: np.ndarray, threshold = 0.3):
        # given embedding, compare to known faces and return best match name and according score
        best_score = 0.0
        best_match = "Unkown"

        if self.known_faces:
            for name, known_feature in self.known_faces.items():
                score = isf.feature_comparison(embedding, known_feature)
                if score > threshold and score > best_score:
                    best_score = score
                    best_match = name

        return best_match, best_score