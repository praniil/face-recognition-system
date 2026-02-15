import face_recognition
import os
from pathlib import Path
import re
from facenet_pytorch import InceptionResnetV1
import torch
from torchvision import transforms
from PIL import Image
import pickle

def get_embedding(image_path):
    # Transform image
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),                   # Converts to torch.Tensor
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalization
    ])
    img = Image.fromarray(image_path.astype('uint8')).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    # Get embedding
    with torch.no_grad():
        embedding = facenet_model(img_tensor)
    return embedding
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()


input_folder_known_face = "../../known_face_crop"
file_names_known_face = os.listdir(input_folder_known_face)
known_image = []
known_face_names = []
known_image_encoding = []

with open('known_face_encoding_data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

for file_name in file_names_known_face:
    known_image.append(face_recognition.load_image_file(os.path.join(input_folder_known_face, file_name)))
    path_object = Path(file_name)
    name_only = re.sub(r'\d+', '', path_object.stem)
    known_face_names.append(name_only)

print(len(known_face_names))
print(len(loaded_data))

#unique name arrays
unique_names = list(set(known_face_names))

for i in known_image:
    encoding = get_embedding(i)
    known_image_encoding.append(encoding[0])  # only one face per image

pickle_data_known_names = {}

for known_names, known_encoding in zip(known_face_names, known_image_encoding):
    pickle_data_known_names[known_names] = known_encoding

print(len(pickle_data_known_names))

with open('known_face_encoding_data.pkl', 'wb') as file:
    pickle.dump(pickle_data_known_names, file)
