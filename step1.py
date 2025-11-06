import cv2
import mediapipe as mp

print("Initializing MediaPipe Face Mesh :- ")

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

print("Opening camera")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise IOError("Cannot open webcam") 

with mp_face_mesh.FaceMesh(
    max_num_faces=1,                
    refine_landmarks=True,          
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    print("Webcam opened. Press 'q' in the popup window to quit.")
    
    while cap.isOpened():
        
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(image_rgb)

        image.flags.writeable = True

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

        cv2.imshow('Face Mesh', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

print("Cleaning up the program residuals and terminating it")
cap.release()
cv2.destroyAllWindows()

for i in range(5):
    cv2.waitKey(1)
