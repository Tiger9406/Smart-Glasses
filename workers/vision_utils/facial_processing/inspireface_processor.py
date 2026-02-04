import inspireface as isf

# import numpy as np


class InspireFaceProcessor:
    def __init__(
        self, model_path="Megatron", confidence_threshold=0.5, download_model=False
    ):
        self.model_path = model_path
        self.session = None

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

    # def process_frame(self, image: np.ndarray):
    #     #given image, run detection & return features
    #     results = []
    #     faces = self.session.face_detection(image)

    #     if not faces:
    #         return results

    #     # okay so we know there are faces; what do we do with the faces
    #     # not a huge overhead so ig we just run

    #     for i, face, in enumerate(faces):
    #         x1, y1, x2, y2 = map(int, face.location)

    #         # we only have to run facial rec if previously undetected
