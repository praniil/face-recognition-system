import face_recognition
import os
from pathlib import Path
import re
import numpy as np
from facenet_pytorch import InceptionResnetV1
import torch
from torchvision import transforms
from PIL import Image
import cv2
import pickle

# define text properties to display
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.4
color = (0, 255, 0)  # Green color in BGR format
thickness = 1
line_type = cv2.LINE_AA

# LOAD FACENET MODEL ONLY ONCE (for performance)
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

# Transform function (store once)
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# GET EMBEDDING FUNCTION
def get_embedding(image_np):
    """
    image_np: numpy array (H,W,3) BGR or RGB
    Returns: numpy array (512,)
    """
    img = Image.fromarray(image_np.astype('uint8')).convert('RGB')
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        emb = facenet_model(tensor)[0]  # (512,)

    return emb.detach().cpu().numpy().reshape(-1)

# ---- LOAD ALL KNOWN FACES -----------------------------
# input_folder_known_face = "../../known_face_crop"
# file_names_known_face = os.listdir(input_folder_known_face)
# known_image = []
# known_face_names = []
# known_image_encoding = []

# with open('known_face_encoding_data.pkl', 'rb') as file:
#     loaded_data = pickle.load(file)

# for file_name in file_names_known_face:
#     known_image.append(face_recognition.load_image_file(os.path.join(input_folder_known_face, file_name)))
#     path_object = Path(file_name)
#     name_only = re.sub(r'\d+', '', path_object.stem)
#     known_face_names.append(name_only)

# print(len(known_face_names))
# print(len(loaded_data))

# #unique name arrays
# unique_names = list(set(known_face_names))
# print(unique_names)
# if len(unique_names) != len(loaded_data):
#     for i in known_image:
#         # encoding = face_recognition.face_encodings(i, model="cnn")      #cnn based on ResNet-34
#         encoding = get_embedding(i)
#         known_image_encoding.append(encoding[0])  # only one face per image
#     pickle_data_known_names = {}
#     for known_names, known_encoding in zip(known_face_names, known_image_encoding):
#         pickle_data_known_names[known_names] = known_encoding
#     print(len(pickle_data_known_names))
#     with open('known_face_encoding_data.pkl', 'wb') as file:
#         pickle.dump(pickle_data_known_names, file)

# else:
#     print("in else")
#     known_face_names.clear()
#     for key, value in loaded_data.items():
#         known_face_names.append(key)
#         known_image_encoding.append(value)
#     print("known from else: ", len(known_face_names))

known_face_names = []
known_image_encoding = []

with open('known_face_encoding_data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)
    
for key, value in loaded_data.items():
    known_face_names.append(key)
    known_image_encoding.append(value)

# MAIN FACE RECOGNITION FUNCTION
def face_recognition_facenet(window_name, frame, image, aligned_face):

    # ---- COMPUTE UNKNOWN FACE EMBEDDING --------------------
    unknown_emb = get_embedding(aligned_face)  # (512,)

    # ---- DISTANCE METRICS ----------------------------------
    def euclidean_distance(known, unknown):
        return np.linalg.norm(known - unknown, axis=1)

    def cosine_similarity(known, unknown):
        known_norm = known / np.linalg.norm(known, axis=1, keepdims=True)
        unknown_norm = unknown / np.linalg.norm(unknown)
        return np.dot(known_norm, unknown_norm)

    # ---- CALCULATE DISTANCES -------------------------------
    distances = euclidean_distance(known_image_encoding, unknown_emb)
    cosine_scores = cosine_similarity(known_image_encoding, unknown_emb)

    # ---- FIND BEST MATCH -----------------------------------
    best_euclid_index = distances.argmin()
    best_cosine_index = cosine_scores.argmax()

    min_dist = distances[best_euclid_index]
    max_cos = cosine_scores[best_cosine_index]

    euclidean_threshold = 1.1
    cosine_threshold = 0.6

    # ---- DECISION (cosine is usually more reliable) --------
    if max_cos >= cosine_threshold:
        matched_name = known_face_names[best_cosine_index]
        print(f"[COSINE] Matched with {matched_name} | sim={max_cos:.3f}")
        return matched_name

    if min_dist <= euclidean_threshold:
        matched_name = known_face_names[best_euclid_index]
        print(f"[EUCLIDEAN] Matched with {matched_name} | dist={min_dist:.3f}")
        return matched_name

    print(f"No match | dist={min_dist:.3f} | sim={max_cos:.3f}")
    return ""
