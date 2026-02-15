import cv2
import torch
from facenet_pytorch import MTCNN
import align_faces_realtime as afreal

# text display properties
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.4
color = (0, 255, 0)
thickness = 1
line_type = cv2.LINE_AA

# GPU / CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# MTCNN
detector = MTCNN(keep_all=True, device=device)

# video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

# cap = cv2.VideoCapture("../../test-videos/attendance101.mp4")
# if not cap.isOpened():
#     print("Error: Could not open video stream")
#     exit()

window_name = "Face detection window"

# STORE LAST NAME FOR EACH FACE 
last_names = []        # persistent storage across frames

def face_detection_mtcnn():
    global last_names

    frame_count = 0
    frame_skip_alignment = 20   # only align every 5 frames

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to grab frame.")
            continue

        frame_count += 1

        # convert BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # face detection
        boxes, probs, landmarks = detector.detect(frame_rgb, landmarks=True)
        min_face_detection_size = 5

        if boxes is not None:
            # ensure last_names has same length as number of faces 
            if len(last_names) < len(boxes):
                last_names.extend(["Unknown"] * (len(boxes) - len(last_names)))

            for i, (box, landmark) in enumerate(zip(boxes, landmarks)):
                x1, y1, x2, y2 = [int(b) for b in box]
                face_crop = frame[y1:y2, x1:x2]

                # validate crop
                if face_crop is not None and face_crop.size != 0:
                    if (y2 - y1) < min_face_detection_size or (x2 - x1) < min_face_detection_size:
                        face_crop = cv2.resize(face_crop, (min_face_detection_size, min_face_detection_size))
                else:
                    print("Error: face_crop empty")
                    continue

                # align & recognize only every N frames
                if frame_count % frame_skip_alignment == 0:
                    matched_name = afreal.align_face_realtime(window_name, frame, face_crop)
                    print("face detection: ", matched_name)
                    last_names[i] = matched_name     # update stored name
                    print(last_names)
                else:
                    matched_name = last_names[i]      # reuse last recognized name


                # draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                # draw landmarks
                for (x, y) in landmark:
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), 2)

                # draw name under face (last recognized name)
                cv2.putText(frame, matched_name, (x1, y2 + 15), font, font_scale, color, thickness, line_type)

        # display
        cv2.imshow(window_name, frame)

        # quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Pressed Q. Closing the window.")
            break

    cap.release()
    cv2.destroyAllWindows()
