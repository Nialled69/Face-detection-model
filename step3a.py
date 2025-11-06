import cv2
import mediapipe as mp
import numpy as np
import math
import time # NEW: To handle timers

# --- Initialization (Same as before) ---
print("Initializing MediaPipe Face Mesh...")
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

print("Opening webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# --- Alignment Constants (Same as before) ---
OUTPUT_SIZE = (112, 112)
TARGET_LANDMARKS = np.array([
    [38.2946, 51.6963],  # Left eye corner
    [73.5318, 51.5014],  # Right eye corner
    [56.0252, 77.7346]   # Mouth center
], dtype=np.float32)

# --- Liveness Constants (Same as before) ---
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

def calculate_ear(landmarks, eye_indices, img_w, img_h):
    points = np.array([
        [landmarks[i].x * img_w, landmarks[i].y * img_h] for i in eye_indices
    ])
    v1 = math.hypot(points[1][0] - points[5][0], points[1][1] - points[5][1])
    v2 = math.hypot(points[2][0] - points[4][0], points[2][1] - points[4][1])
    h = math.hypot(points[0][0] - points[3][0], points[0][1] - points[3][1])
    if h == 0: return 0.0
    ear = (v1 + v2) / (2.0 * h)
    return ear

# --- NEW: Liveness State Machine ---
class LivenessChallenge:
    def __init__(self, blink_threshold=0.20, challenge_wait_time=3.0, response_time=2.0):
        self.state = "START" # START -> CHALLENGE -> PASSED / FAILED
        self.blink_threshold = blink_threshold # Based on your 0.17 observation
        self.challenge_wait_time = challenge_wait_time # Time to get ready
        self.response_time = response_time # Time to perform the blink
        self.start_time = time.time()
        self.blink_detected = False
        print(f"Liveness test initiated. Get ready... (Threshold set to {self.blink_threshold})")

    def update_state(self, ear):
        current_time = time.time()
        elapsed = current_time - self.start_time

        if self.state == "START":
            if elapsed > self.challenge_wait_time:
                self.state = "CHALLENGE"
                self.start_time = current_time # Reset timer for the response
                print("CHALLENGE: BLINK NOW!")
            return "Get Ready..."

        elif self.state == "CHALLENGE":
            if elapsed > self.response_time:
                # Time's up! Check if they blinked.
                if self.blink_detected:
                    self.state = "PASSED"
                    print("Liveness: PASSED")
                else:
                    self.state = "FAILED"
                    print("Liveness: FAILED")
                return self.state
            
            # Check for the blink *during* the challenge window
            if ear < self.blink_threshold:
                self.blink_detected = True
                
            return "BLINK NOW"

        elif self.state == "PASSED":
            return "Liveness: SUCCESS"

        elif self.state == "FAILED":
            return "Liveness: FAILED"

# Initialize our liveness challenge
liveness_test = LivenessChallenge()

# --- Main Loop ---
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        img_h, img_w = image.shape[:2]
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        image.flags.writeable = True

        aligned_face = np.zeros((OUTPUT_SIZE[0], OUTPUT_SIZE[1], 3), dtype=np.uint8)
        status_text = ""

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # --- Alignment Logic (Same as before) ---
            left_eye_x = face_landmarks.landmark[362].x
            right_eye_x = face_landmarks.landmark[133].x
            source_indices = [362, 133, 13] if left_eye_x < right_eye_x else [133, 362, 13]
            
            source_points = np.array([
                [face_landmarks.landmark[i].x * img_w, face_landmarks.landmark[i].y * img_h]
                for i in source_indices
            ], dtype=np.float32)
            
            transform_matrix = cv2.getAffineTransform(source_points, TARGET_LANDMARKS)
            aligned_face = cv2.warpAffine(
                image, transform_matrix, (OUTPUT_SIZE[1], OUTPUT_SIZE[0]), flags=cv2.INTER_LINEAR
            )

            # --- Liveness Calculation (Same as before) ---
            left_ear = calculate_ear(face_landmarks.landmark, LEFT_EYE_INDICES, img_w, img_h)
            right_ear = calculate_ear(face_landmarks.landmark, RIGHT_EYE_INDICES, img_w, img_h)
            ear = (left_ear + right_ear) / 2.0
            
            # --- NEW: Update Liveness State ---
            # Update the state machine with the current EAR
            status_text = liveness_test.update_state(ear)
            
            # --- Drawing (Same as before) ---
            mp_drawing.draw_landmarks(
                image=image, landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)
        
        else:
            # Handle case where no face is detected
            status_text = "No face detected"
            # Reset the test if the user looks away
            liveness_test = LivenessChallenge() 

        # --- Display Status Text ---
        # Set color based on state
        if liveness_test.state == "PASSED":
            color = (0, 255, 0) # Green
        elif liveness_test.state == "FAILED":
            color = (0, 0, 255) # Red
        else:
            color = (255, 255, 0) # Cyan

        cv2.putText(image, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # --- Display windows ---
        cv2.imshow('Liveness Challenge (Step 3)', image)
        cv2.imshow('Aligned Face (Step 2)', aligned_face)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# --- Cleanup ---
print("Cleaning up and closing...")
cap.release()
cv2.destroyAllWindows()
for i in range(5):
    cv2.waitKey(1)