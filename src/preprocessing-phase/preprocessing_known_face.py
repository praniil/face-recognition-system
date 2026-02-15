import cv2
import os
from facenet_pytorch import MTCNN

input_folder_path = "../../known_faces"
output_folder_path = "../../known_face_crop"

os.makedirs(output_folder_path, exist_ok=True)

known_faces = os.listdir(input_folder_path)

detector = MTCNN()

facenet_dim = 160

for know_face in known_faces:
    img = cv2.imread(os.path.join(input_folder_path, know_face))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes, probs, landmarks = detector.detect(img_rgb, landmarks=True)

    if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                face_crop = img[y1:y2, x1:x2]   

                if face_crop is not None and face_crop.size != 0:
                    if (y2 - y1) < facenet_dim or (x2 - x1) < facenet_dim:
                        face_crop = cv2.resize(face_crop, (facenet_dim, facenet_dim)) # facenet expects this dim
                else:
                    print("Error: face_crop empty")
                    continue

                print(os.path.join(output_folder_path, know_face))
                cv2.imwrite(os.path.join(output_folder_path, know_face), face_crop)