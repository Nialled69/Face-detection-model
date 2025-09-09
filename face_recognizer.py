from deepface import DeepFace
import numpy as np

class FaceRecognizer:

    """
    A wrapper for the DeepFace library to use a specific recognition model.
    """

    def __init__(self, model_name='ArcFace', backend='tensorflow'):

        """
        Initializes the recognizer and builds the model.
        This will download the model weights the first time it's run.
        """

        self.model_name = model_name
        self.backend = backend
        
        # Building the model on initialization for loading into memory.
        # The first call is always slow since the model will be built first bozo
        
        print(f"[INFO] Loading '{self.model_name}' model...")
        dummy_image = np.zeros((112, 112, 3), dtype=np.uint8)
        DeepFace.represent(img_path=dummy_image, 
                           model_name=self.model_name, 
                           detector_backend='skip')
        print(f"[INFO] '{self.model_name}' model loaded successfully.")

    def get_embedding(self, face_image):
        
        """
        Generates a face embedding for a given face image.

        Args:
            face_image (np.ndarray): The aligned face image (BGR format).

        Returns:
            list: The 512-dimension embedding vector, or None if no face is found.
        """
        
        try:
            embedding_objs = DeepFace.represent( # passing BGR image
                img_path=face_image,
                model_name=self.model_name,
                detector_backend='skip'  # skipping detection because the face is already aligned and cropped.
            )
            
            if embedding_objs:
                return embedding_objs[0]["embedding"]
            else:
                return None

        except Exception as e:
            print(f"[ERROR] Could not generate embedding: {e}")
            return None