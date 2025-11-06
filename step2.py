import cv2
import mediapipe as mp
import numpy as np

print("Initializing MediaPipe Face Mesh :-")
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

print("Opening camera")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

OUTPUT_SIZE = (112, 112) # (height, width)


TARGET_LANDMARKS = np.array([
    [38.2946, 51.6963],  # Target for the landmark on the *left* of the screen
    [73.5318, 51.5014],  # Target for the landmark on the *right* of the screen
    [56.0252, 77.7346]   # Target for the mouth
], dtype=np.float32)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    print("Webcam opened. Press 'q' in the popup window to quit.")
    
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

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # Dynamic Orientation Check
            # Get x-coordinates of the semantic left (362) and right (133) eyes
            left_eye_x = face_landmarks.landmark[362].x
            right_eye_x = face_landmarks.landmark[133].x

            if left_eye_x < right_eye_x:
                # Mirrored image (pc webcam option )
                # Left eye (362) is on the left side of the screen.
                # Right eye (133) is on the right side of the screen.
                source_indices = [362, 133, 13]
            else:
                # Non-mirrored image (like Android default)
                # Right eye (133) is on the left side of the screen.
                # Left eye (362) is on the right side of the screen.
                source_indices = [133, 362, 13]

            source_points = np.array([
                [face_landmarks.landmark[i].x * img_w, face_landmarks.landmark[i].y * img_h]
                for i in source_indices
            ], dtype=np.float32)

            # Calculate the Affine Transformation Matrix
            transform_matrix = cv2.getAffineTransform(source_points, TARGET_LANDMARKS)

            # Applying the transformation to the original image
            aligned_face = cv2.warpAffine(
                image,
                transform_matrix,
                (OUTPUT_SIZE[1], OUTPUT_SIZE[0]), # dsize is (width, height)
                flags=cv2.INTER_LINEAR
            )

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

        cv2.imshow('MediaPipe Face Mesh (Step 1)', image)
        cv2.imshow('Aligned Face (Step 2 - Robust)', aligned_face)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

print("Cleaning up the program residuals and terminating it")
cap.release()
cv2.destroyAllWindows()
for i in range(5):
    cv2.waitKey(1)
