import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random
import insightface # NEW: For ArcFace recognition

# --- Initialization (Same as before) ---
print("Initializing MediaPipe Face Mesh...")
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

print("Opening webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# --- NEW: Load InsightFace ArcFace Model ---
# This will download the model on the first run
print("Loading InsightFace ArcFace model... (This may take a moment)")
# We initialize the full 'app' pipeline
app = insightface.app.FaceAnalysis(name='buffalo_l', 
                                   providers=['CPUExecutionProvider'])
# We tell it to prepare for CPU execution
app.prepare(ctx_id=-1)
# *** We then extract just the recognition model ***
# This allows us to bypass insightface's detection/alignment
rec_model = app.models['recognition']
print("ArcFace model loaded successfully.")

# --- Alignment Constants (Same as before)---
OUTPUT_SIZE = (112, 112)
TARGET_LANDMARKS = np.array([
    [38.2946, 51.6963],  # Left eye corner
    [73.5318, 51.5014],  # Right eye corner
    [56.0252, 77.7346]   # Mouth center
], dtype=np.float32)

# --- Liveness Constants (Same as before) ---
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291
EYE_LEFT_INNER = 362
EYE_RIGHT_INNER = 133

# --- Helper Functions (Same as before) ---
def get_2d_coords(landmarks, landmark_index, img_w, img_h):
    lm = landmarks[landmark_index]
    return np.array([lm.x * img_w, lm.y * img_h])

def get_distance(lm1, lm2):
    return math.hypot(lm1[0] - lm2[0], lm1[1] - lm2[1])

def calculate_ear(landmarks, eye_indices, img_w, img_h):
    points = np.array([[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in eye_indices])
    v1 = math.hypot(points[1][0] - points[5][0], points[1][1] - points[5][1])
    v2 = math.hypot(points[2][0] - points[4][0], points[2][1] - points[4][1])
    h = math.hypot(points[0][0] - points[3][0], points[0][1] - points[3][1])
    if h == 0: return 0.0
    return (v1 + v2) / (2.0 * h)

def calculate_smile_ratio(landmarks, img_w, img_h):
    mouth_left = get_2d_coords(landmarks, MOUTH_LEFT_CORNER, img_w, img_h)
    mouth_right = get_2d_coords(landmarks, MOUTH_RIGHT_CORNER, img_w, img_h)
    eye_left = get_2d_coords(landmarks, EYE_LEFT_INNER, img_w, img_h)
    eye_right = get_2d_coords(landmarks, EYE_RIGHT_INNER, img_w, img_h)
    mouth_width = get_distance(mouth_left, mouth_right)
    eye_width = get_distance(eye_left, eye_right)
    if eye_width == 0: return 0.0
    return mouth_width / eye_width

# --- Liveness State Machine (Same as before) ---
class LivenessChallenge:
    def __init__(self, challenge_wait_time=3.0, response_time=2.0):
        self.state = "START"
        self.challenge_wait_time = challenge_wait_time
        self.response_time = response_time
        self.start_time = time.time()
        self.action_detected = False
        self.challenge_type = random.choice(["BLINK", "SMILE"])
        self.threshold = 0.0
        
        if self.challenge_type == "BLINK":
            self.threshold = 0.20
        else: # "SMILE"
            self.threshold = 1.7
        print(f"Liveness test: {self.challenge_type} (Thresh: {self.threshold})")

    def update_state(self, ear, sr):
        current_time = time.time()
        elapsed = current_time - self.start_time

        if self.state == "START":
            if elapsed > self.challenge_wait_time:
                self.state = "CHALLENGE"
                self.start_time = current_time
            return "Get Ready..."
        
        elif self.state == "CHALLENGE":
            challenge_text = f"{self.challenge_type} NOW"
            if elapsed > self.response_time:
                self.state = "PASSED" if self.action_detected else "FAILED"
                return self.state
            
            if self.challenge_type == "BLINK":
                if ear < self.threshold: self.action_detected = True
            elif self.challenge_type == "SMILE":
                if sr > self.threshold: self.action_detected = True
            return challenge_text
        
        return self.state

liveness_test = LivenessChallenge()
embedding_printed = False # NEW: To print only once

# --- Main Loop ---
with mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        img_h, img_w = image.shape[:2]
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        image.flags.writeable = True

        aligned_face = np.zeros((OUTPUT_SIZE[0], OUTPUT_SIZE[1], 3), dtype=np.uint8)
        status_text = ""

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            
            # --- Alignment Logic (Same as before) ---
            left_eye_x = face_landmarks[EYE_LEFT_INNER].x
            right_eye_x = face_landmarks[EYE_RIGHT_INNER].x
            source_indices = [EYE_LEFT_INNER, EYE_RIGHT_INNER, 13] if left_eye_x > right_eye_x else [EYE_RIGHT_INNER, EYE_LEFT_INNER, 13]
            source_points = np.array([[face_landmarks[i].x * img_w, face_landmarks[i].y * img_h] for i in source_indices], dtype=np.float32)
            transform_matrix = cv2.getAffineTransform(source_points, TARGET_LANDMARKS)
            aligned_face = cv2.warpAffine(
                image, transform_matrix, (OUTPUT_SIZE[1], OUTPUT_SIZE[0]), flags=cv2.INTER_LINEAR
            )

            # --- Liveness Calculations (Same as before) ---
            ear = (calculate_ear(face_landmarks, LEFT_EYE_INDICES, img_w, img_h) + 
                   calculate_ear(face_landmarks, RIGHT_EYE_INDICES, img_w, img_h)) / 2.0
            sr = calculate_smile_ratio(face_landmarks, img_w, img_h)
            
            status_text = liveness_test.update_state(ear, sr)
            
            # --- NEW: Generate ArcFace Embedding ---
            if liveness_test.state == "PASSED" and not embedding_printed:
                # 1. Convert our 112x112 aligned BGR face to RGB
                aligned_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                
                # 2. Normalize from [0, 255] to [-1.0, 1.0]
                aligned_normalized = (aligned_rgb.astype(np.float32) - 127.5) / 128.0
                
                # 3. Transpose from (H, W, C) to (C, H, W)
                aligned_transposed = np.transpose(aligned_normalized, (2, 0, 1))
                
                # 4. Add batch dimension (N) -> (1, 3, 112, 112)
                input_blob = np.expand_dims(aligned_transposed, axis=0)
                
                # 5. Get the 512-dim embedding
                embedding = rec_model.session.run(rec_model.output_names, {rec_model.input_name: input_blob})[0]
                
                # 6. Flatten from (1, 512) to (512,)
                embedding = embedding.flatten()
                
                print("--- STEP 4 SUCCESS ---")
                print(f"Generated embedding of shape: {embedding.shape}")
                print(f"First 5 values: {embedding[:5]}")
                embedding_printed = True # Make sure we only print this once
            
            # --- Drawing (Same as before) ---
            mp_drawing.draw_landmarks(
                image=image, landmark_list=results.multi_face_landmarks[0],
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)
        
        else:
            status_text = "No face detected"
            liveness_test = LivenessChallenge()
            embedding_printed = False # Reset if face is lost

        # --- Display Status Text (Same as before) ---
        if liveness_test.state == "PASSED": color = (0, 255, 0)
        elif liveness_test.state == "FAILED": color = (0, 0, 255)
        else: color = (255, 255, 0)
        cv2.putText(image, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # --- Display windows ---
        cv2.imshow('Face Recognition Pipeline (Step 4)', image)
        cv2.imshow('Aligned Face (Step 2)', aligned_face)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# --- Cleanup ---
print("Cleaning up and closing...")
cap.release()
cv2.destroyAllWindows()
for i in range(5):
    cv2.waitKey(1)