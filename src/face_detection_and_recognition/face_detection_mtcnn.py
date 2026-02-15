import cv2
import torch
from facenet_pytorch import MTCNN
import os
import shutil
import align_faces as af

file_path = "../../detected-faces-mtcnn"
#check if the folder exists
if os.path.exists(file_path):
    shutil.rmtree(file_path)
    print(f"deleted the existing folder: {file_path}")
    
os.makedirs(file_path, exist_ok=True)

# define text properties to display
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.4
color = (0, 255, 0)  # Green color in BGR format
thickness = 1
line_type = cv2.LINE_AA

def face_detection_mtcnn():
    # Set up video capture
    cap = cv2.VideoCapture(0)

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

    # name of the opencv window
    window_name = "Face detection window"

    while True:
        # capture the frame
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to grab frame.")
            continue

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        boxes, probs, landmarks = detector.detect(frame_rgb, landmarks=True)
        min_face_detection_size = 20

        # Draw results
        if boxes is not None:
            for i, (box, landmark) in enumerate(zip(boxes, landmarks)):
                x1, y1, x2, y2 = [int(b) for b in box]
                face_crop = frame[y1:y2, x1:x2]
                
                text = f"Confidence: {probs[0]:.8f}"
                org = (x1, y2 + 15)

                # check if face crop is empty or full
                if face_crop is not None and not face_crop.size == 0:
                    if (y2 - y1) < min_face_detection_size and (x2 - x1) < min_face_detection_size:
                        face_crop = cv2.resize(face_crop, (min_face_detection_size, min_face_detection_size))

                    cv2.imwrite(f"{file_path}/direct_detected_face{i}.png", face_crop)

                else:
                    print("Error: face_crop is empty or None. Cannot resize.")
                
                #write the confidence score
                cv2.putText(frame, text, org, font, font_scale, color, thickness, line_type)
                # draw rectange around the face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                # Draw facial landmarks
                for (x, y) in landmark:
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), 2)

        # Display
        cv2.imshow(window_name, frame)

        # Press 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Pressed Q. Closing the window.")
            break

    cap.release()
    cv2.destroyAllWindows()

    # align_face() funciton called from align_faces.py
    af.align_face()

if __name__ == "__main__":
    face_detection_mtcnn()