import os
import cv2
import face_alignment
import numpy as np
import shutil
import math

MIN_FACE_SIZE = 160
def align_face():
    # Input and output folders
    # input_folder = "../../detected-faces-mtcnn"
    input_folder = "../../detected-faces-mtcnn-test-video"
    output_folder = "../../detected-aligned-faces-mtcnn"

    # Check if the folder exists
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
        print(f"Deleted the existing folder: {output_folder}")

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Initialize face alignment (2D landmarks)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

    # List all images in the input folder
    file_names = os.listdir(input_folder)

    for file_name in file_names:
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        # Read image
        image = cv2.imread(input_path)
        if image is None or image.size == 0:
            return None
        
        #extract the first two indices height and width
        h, w = image.shape[:2]

        # Skip or resize small faces
        if h < MIN_FACE_SIZE or w < MIN_FACE_SIZE:
            # print(f"Face too small ({w}x{h}), resizing to {MIN_FACE_SIZE}x{MIN_FACE_SIZE}")
            image = cv2.resize(image, (MIN_FACE_SIZE, MIN_FACE_SIZE), interpolation=cv2.INTER_CUBIC)

        # Detect landmarks
        landmarks_list = fa.get_landmarks(image)
        if landmarks_list is None:
            print(f"No face detected in: {file_name}")
            continue

        # Take the first detected face
        landmarks = landmarks_list[0]

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

        # Compute eye distance
        eye_distance = np.linalg.norm(right_eye - left_eye)

        # Nose point
        nose_point = landmarks[31]

        # 2D Vector-based yaw and pitch estimation 
        horizontal_nose_deviation = nose_point[0] - eyes_center[0]
        vertical_nose_deviation = nose_point[1] - eyes_center[1]

        yaw_angle = math.atan(horizontal_nose_deviation / eye_distance)   # radians
        pitch_angle = -math.atan(vertical_nose_deviation / eye_distance)  # radians

        # Convert to degrees
        yaw_angle_deg = np.degrees(yaw_angle)
        pitch_angle_deg = np.degrees(pitch_angle)

        # Print angles
        print(f"File: {file_name}")
        print(f"Roll: {roll_angle:.2f}°, Yaw: {yaw_angle_deg:.2f}°, Pitch: {pitch_angle_deg:.2f}°")

        # Save aligned face (roll-corrected only)
        cv2.imwrite(output_path, aligned_face)
        print(f"Aligned face saved: {output_path}")

    print("Face alignment completed!")

if __name__ == "__main__":
    align_face()
