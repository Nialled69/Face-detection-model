import cv2
import mediapipe as mp
import numpy as np
from face_processing import FaceAligner
from face_recognizer import FaceRecognizer

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize our FaceAligner
aligner = FaceAligner(desired_face_width=112, desired_face_height=112)

# Initialize our FaceRecognizer
# This will take a moment as it loads the ArcFace model into memory
recognizer = FaceRecognizer(model_name='ArcFace')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()
print("[INFO] Starting video streaming...")

prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    aligned_face_canvas = np.zeros((112, 112, 3), dtype="uint8")

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        # 1. Align the face
        aligned_face = aligner.align(frame.copy(), face_landmarks)
        
        if aligned_face is not None and aligned_face.shape[0] > 0 and aligned_face.shape[1] > 0:
            aligned_face_canvas = aligned_face
            
            # 2. Get the face embedding
            embedding = recognizer.get_embedding(aligned_face)
            
            # For now, just print the shape to confirm it works
            if embedding is not None:
                # The embedding is a list of 512 numbers
                print(f"Embedding Shape: {np.array(embedding).shape}", end='\r')


    # Display the original frame and the aligned face
    cv2.imshow("Webcam Feed", frame)
    cv2.imshow("Aligned Face", aligned_face_canvas)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
face_mesh.close()