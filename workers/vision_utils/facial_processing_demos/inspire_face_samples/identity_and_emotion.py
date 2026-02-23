import cv2
import inspireface as isf


def main():
    isf.ignore_check_latest_model(True)
    ret = isf.launch("Megatron")
    assert ret, "Launch failure. Please ensure the resource path is correct."

    params = isf.SessionCustomParameter(
        enable_recognition=True,
        enable_face_emotion=True,
        enable_face_attribute=False,  # age & gender & whatnot
        enable_liveness=False,  # interesting parameter: to differentiate a physical picture of someone vs real person
    )

    session = isf.InspireFaceSession(params)

    session.set_detection_confidence_threshold(0.5)

    def get_face_feature(image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        faces = session.face_detection(img)
        if not faces:
            return None

        # gets the first for now; we can do multiple people later
        feature = session.face_feature_extract(img, faces[0])
        return feature

    known_feature = get_face_feature("./workers/vision_utils/photos/You.png")
    print("Collected your facial data")

    test_img = cv2.imread("./workers/vision_utils/photos/You_And_Others.png")
    faces = session.face_detection(test_img)

    print(f"Detected {len(faces)} faces")

    EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    if len(faces) > 0:
        # new pipeline for analyzing emotions
        extended_faces = session.face_pipeline(test_img, faces, params)
        for i, face in enumerate(faces):
            detected_name = "Unkown"
            current_feature = session.face_feature_extract(test_img, face)
            similarity = isf.feature_comparison(current_feature, known_feature)
            if similarity > 0.3:
                detected_name = "You"

            emotion_idx = extended_faces[i].emotion
            emotion_str = EMOTION_LABELS[emotion_idx]

            x1, y1, x2, y2 = face.location

            color = (0, 255, 0) if detected_name == "Unkown" else (0, 0, 255)
            cv2.rectangle(test_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

            label = f"{detected_name} | {emotion_str} ({similarity: .2f})"
            cv2.putText(
                test_img,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                color,
                1,
            )

    print("Press 'q' to close image window")
    while True:
        cv2.imshow("Processed Recognition Frame: ", test_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
