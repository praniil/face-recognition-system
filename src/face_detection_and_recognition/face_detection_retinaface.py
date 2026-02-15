import cv2
from retinaface import RetinaFace

def face_detection_retinaface():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return
    
    # Set FPS
    cap.set(cv2.CAP_PROP_FPS, 60)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Requested FPS: 60, actual FPS: {actual_fps}")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to grab frame.")
            continue

        # Detect faces
        detections = RetinaFace.detect_faces(frame)

        # Draw bounding boxes and landmarks
        for face_id, face_data in detections.items():
            x1, y1, x2, y2 = face_data['facial_area']
            landmarks = face_data['landmarks']

            # Draw bounding box (convert floats to int)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Draw landmarks
            for key in landmarks:
                x, y = landmarks[key]
                cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
            
        cv2.imshow('Face Detection', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Pressed Q. Closing the window.")
            break

    cap.release()
    cv2.destroyAllWindows()