import cv2
import torch
from facenet_pytorch import MTCNN
import os
import shutil

file_path = "../../detected-faces-mtcnn-test-video"
#check if the folder exists
if os.path.exists(file_path):
    shutil.rmtree(file_path)
    print(f"deleted the existing folder: {file_path}")
    
os.makedirs(file_path, exist_ok=True)

def face_detection_mtcnn_test_video():
    # Set up video capture
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("../../test-videos/attendance2.mp4")
    if not cap.isOpened():
        print("Error: Could not open video stream")
        exit()
    
    # Set requested FPS
    cap.set(cv2.CAP_PROP_FPS, 60)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Requested FPS: 60, actual FPS: {actual_fps}")

    # Use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize MTCNN detector
    detector = MTCNN(keep_all=True, device=device)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to grab frame.")
            continue

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        boxes, probs, landmarks = detector.detect(frame_rgb, landmarks=True)

        # Draw results
        if boxes is not None:
            for i, (box, landmark) in enumerate(zip(boxes, landmarks)):
                x1, y1, x2, y2 = [int(b) for b in box]
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size == 0:
                    print("Warning: Empty crop, skipping")
                    continue

                #save the face crop in the target directory
                cv2.imwrite(f"{file_path}/direct_detected_face{i}.png", face_crop)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                
                # Draw facial landmarks
                for (x, y) in landmark:
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), 2)
            

        # Display
        cv2.imshow('Face Detection', frame)

        # Press 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Pressed Q. Closing the window.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_detection_mtcnn_test_video()