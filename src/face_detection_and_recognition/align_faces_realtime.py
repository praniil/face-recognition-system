import cv2
import face_alignment
import numpy as np
import face_recognition_realtime as frecog

# Load the face alignment model
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

MIN_FACE_SIZE = 160  # safe minimum size for FaceAlignment

def align_face_realtime(window_name, frame, image):
    if image is None or image.size == 0:
        return None
    
    #extract the first two indices height and width
    h, w = image.shape[:2]
    
    # Skip or resize small faces
    if h < MIN_FACE_SIZE or w < MIN_FACE_SIZE:
        # print(f"Face too small ({w}x{h}), resizing to {MIN_FACE_SIZE}x{MIN_FACE_SIZE}")
        image = cv2.resize(image, (MIN_FACE_SIZE, MIN_FACE_SIZE), interpolation=cv2.INTER_CUBIC)
    
    # Detect landmarks
    landmark_list = fa.get_landmarks(image)
    if landmark_list is None:
        print("No landmarks detected")
        return None
    
    landmarks = landmark_list[0]

    # Eye landmarks
    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)

    # Roll angle (tilt of eyes)
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    roll_angle = np.degrees(np.arctan2(dy, dx))

    # Eye midpoint
    eyes_center = (left_eye + right_eye) / 2

    # Roll alignment (2D rotation)
    Mroll = cv2.getRotationMatrix2D(tuple(eyes_center), roll_angle, scale=1)
    aligned_face = cv2.warpAffine(image, Mroll, (image.shape[1], image.shape[0]),
                                  flags=cv2.INTER_CUBIC)
    
    # Call FaceNet recognition
    matched_name = frecog.face_recognition_facenet(window_name, frame, image, aligned_face)

    return matched_name