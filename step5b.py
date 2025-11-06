import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random
import insightface
import os

# --- NEW: Helper function for Cosine Similarity ---
def cosine_similarity(emb1, emb2):
    """Calculates the cosine similarity between two embeddings."""
    dot_product = np.dot(emb1, emb2)
    norm_emb1 = np.linalg.norm(emb1)
    norm_emb2 = np.linalg.norm(emb2)
    return dot_product / (norm_emb1 * norm_emb2)

# --- Initialization (All same as before) ---
print("Initializing...")
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

cap = cv2.VideoCapture(0)
if not cap.isOpened(): raise IOError("Cannot open webcam")

print("Loading ArcFace model...")
app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1)
rec_model = app.models['recognition']
print("Model loaded.")

# --- NEW: Load Face Database ---
DATABASE_PATH = "face_database.npy"
if not os.path.exists(DATABASE_PATH):
    print("Error: Database 'face_database.npy' not found.")
    print("Please run the 'step5_enroll.py' script first.")
    exit()

print("Loading face database...")
database = np.load(DATABASE_PATH, allow_pickle=True).item()
print(f"Database loaded. Found {len(database)} enrolled users.")

# --- Constants (All same as before) ---
OUTPUT_SIZE = (112, 112)
TARGET_LANDMARKS = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 77.7346]
], dtype=np.float32)
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291
EYE_LEFT_INNER = 362
EYE_RIGHT_INNER = 133
VERIFICATION_THRESHOLD = 0.5 # Cosine similarity. ArcFace is high; > 0.4 is good, > 0.5 is strict.

# --- Helper Functions (All same as before) ---
# (get_2d_coords, get_distance, calculate_ear, calculate_smile_ratio)
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

# --- Liveness State Machine (Same, but we reset it) ---
class LivenessChallenge:
    def __init__(self, challenge_wait_time=3.0, response_time=2.0):
        self.state = "START"
        self.challenge_wait_time = challenge_wait_time
        self.response_time = response_time
        self.start_time = time.time()
        self.action_detected = False
        self.challenge_type = random.choice(["BLINK", "SMILE"])
        self.threshold = 0.20 if self.challenge_type == "BLINK" else 1.7
        print(f"Verification - Liveness test: {self.challenge_type} (Thresh: {self.threshold})")

    def update_state(self, ear, sr):
        current_time = time.time()
        elapsed = current_time - self.start_time
        if self.state == "START":
            if elapsed > self.challenge_wait_time:
                self.state = "CHALLENGE"; self.start_time = current_time
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
    
    def reset(self):
        # NEW: To restart the test
        print("Resetting liveness test...")
        self.__init__()

liveness_test = LivenessChallenge()
verification_state = "PENDING" # PENDING -> SUCCESS / FAILED
verification_name = ""

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
            
            # --- Alignment ---
            left_eye_x = face_landmarks[EYE_LEFT_INNER].x
            right_eye_x = face_landmarks[EYE_RIGHT_INNER].x
            source_indices = [EYE_LEFT_INNER, EYE_RIGHT_INNER, 13] if left_eye_x > right_eye_x else [EYE_RIGHT_INNER, EYE_LEFT_INNER, 13]
            source_points = np.array([[face_landmarks[i].x * img_w, face_landmarks[i].y * img_h] for i in source_indices], dtype=np.float32)
            transform_matrix = cv2.getAffineTransform(source_points, TARGET_LANDMARKS)
            aligned_face = cv2.warpAffine(
                image, transform_matrix, (OUTPUT_SIZE[1], OUTPUT_SIZE[0]), flags=cv2.INTER_LINEAR
            )

            # --- Liveness ---
            ear = (calculate_ear(face_landmarks, LEFT_EYE_INDICES, img_w, img_h) + 
                   calculate_ear(face_landmarks, RIGHT_EYE_INDICES, img_w, img_h)) / 2.0
            sr = calculate_smile_ratio(face_landmarks, img_w, img_h)
            
            # Only run liveness if verification is pending
            if verification_state == "PENDING":
                status_text = liveness_test.update_state(ear, sr)
            
            # --- Verification Logic ---
            if liveness_test.state == "PASSED":
                # Liveness passed, now do recognition
                print("Liveness passed, checking recognition...")
                
                # --- Get Live Embedding ---
                aligned_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                aligned_normalized = (aligned_rgb.astype(np.float32) - 127.5) / 128.0
                aligned_transposed = np.transpose(aligned_normalized, (2, 0, 1))
                input_blob = np.expand_dims(aligned_transposed, axis=0)
                live_embedding = rec_model.session.run(rec_model.output_names, 
                                                       {rec_model.input_name: input_blob})[0]
                live_embedding = live_embedding.flatten()
                
                # --- Compare to Database ---
                best_match_name = ""
                best_match_score = 0.0
                
                for name, saved_embedding in database.items():
                    similarity = cosine_similarity(live_embedding, saved_embedding)
                    
                    if similarity > best_match_score:
                        best_match_score = similarity
                        best_match_name = name
                
                # --- Final Decision ---
                if best_match_score > VERIFICATION_THRESHOLD:
                    verification_state = "SUCCESS"
                    verification_name = best_match_name
                    status_text = f"Welcome, {verification_name}! (Score: {best_match_score:.2f})"
                else:
                    verification_state = "FAILED"
                    status_text = f"Unknown User (Best: {best_match_score:.2f})"
                
                # We've made a decision, reset liveness for the next attempt
                liveness_test.state = "DONE" # Stop liveness from running
            
            elif liveness_test.state == "FAILED":
                status_text = "Liveness FAILED. Please try again."
                # Reset the whole process
                verification_state = "PENDING"
                liveness_test.reset()

        else:
            status_text = "No face detected"
            verification_state = "PENDING" # Reset if face is lost
            liveness_test.reset()

        # --- Display Final Status Text ---
        if verification_state == "SUCCESS":
            color = (0, 255, 0) # Green
            status_text = f"Welcome, {verification_name}!"
        elif verification_state == "FAILED":
            color = (0, 0, 255) # Red
            status_text = "Verification FAILED"
        elif liveness_test.state == "FAILED":
            color = (0, 0, 255) # Red
            status_text = "Liveness FAILED"
        else:
            color = (255, 255, 0) # Cyan
        
        cv2.putText(image, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow('Verification (Step 5)', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# --- Cleanup ---
print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
