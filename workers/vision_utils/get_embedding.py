import cv2
from workers.vision_utils.inspireface_processor import InspireFaceProcessor
import numpy as np

if __name__ == "__main__":
    processor = InspireFaceProcessor()
    image_root = "database/sample_data/"
    person_name = "ross_geller"
    image_type = "_face.jpg"
    img_path = f"{image_root}{person_name}{image_type}"
    image = cv2.imread(img_path)

    if image is None:
        print("Could not read image")
    else:
        faces = processor.detect_faces(image)
        if len(faces) > 0:
            embedding = processor.extract_embedding(image, faces[0])
            output_file = f"{image_root}face_{person_name}.npy"
            np.save(output_file, embedding)
            
            print(f"Success! Embedding saved to {output_file}")
            print(f"Vector shape: {embedding.shape}")
        else:
            print("No faces detected in the image.")