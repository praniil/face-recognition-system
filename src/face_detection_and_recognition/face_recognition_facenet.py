import face_recognition
import os
from pathlib import Path
from db_connection import db_connection
import re
import numpy as np
from facenet_pytorch import InceptionResnetV1
import torch
from torchvision import transforms
from PIL import Image
import pickle

def face_recognition_facenet():
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
    # DB CONNECTION

    connection = db_connection()
    cursor = connection.cursor()

    # creating a table of students info if not exists

    # if connection is None:
    #     print(f"Cant access the database")
    # else:
    #     create_students_query = """
    #         CREATE TABLE IF NOT EXISTS students (
    #         student_id SERIAL PRIMARY KEY,
    #         name VARCHAR(100),
    #         enrollment_status VARCHAR(100)
    #         );
    #     """
    #     cursor.execute(create_students_query)

    #     create_students_face_encoding_query = """
    #         CREATE TABLE IF NOT EXISTS students_face_encoding (
    #             face_encoding_id SERIAL PRIMARY KEY,
    #             student_id SERIAL NOT NULL,
    #             face_encoding DOUBLE PRECISION[],
    #             FOREIGN KEY(student_id) REFERENCES students(student_id)
    #         );
    #     """
    #     cursor.execute(create_students_face_encoding_query)

    #     connection.commit()
    #     print("students and face encoding table created successfully.")

    # LOADING THE KNOWN FACES

    input_folder_known_face = "../../known_face_crop"
    # file_names_known_face = os.listdir(input_folder_known_face)

    # known_image = []
    known_face_names = []
    known_image_encoding = []

    with open('known_face_encoding_data.pkl', 'rb') as file:
        loaded_data = pickle.load(file)

    for key, value in loaded_data.items():
        known_face_names.append(key)
        known_image_encoding.append(value)

    # # INSERITNG THE DETAILS OF STUDENTS IN STUDENTS TABLE
    # for i, name in enumerate(unique_names):
    #     student_table = "students"
    #     students_column = "(name, enrollment_status)"
    #     enrollment_status = "enrolled"
    #     insert_query = """
    #         INSERT INTO {table} {columns}
    #         VALUES (%s, %s)
    #     """.format(table = student_table, columns = students_column)
    #     cursor.execute(insert_query, (name, enrollment_status))
    #     connection.commit()


    # # INSERTING THE FACE ENCODING IN STUDENTS FACE ENCODING TABLE
    # for name, face_encoding in zip(known_face_names, known_image_encoding):
    #     select_query = """SELECT student_id from students where name = %s"""
    #     cursor.execute(select_query, (name,))
    #     result = cursor.fetchall()
    #     print(result)

    #     table_name = "students_face_encoding"
    #     columns = "(student_id, face_encoding)"

    #     if result:
    #         for row in result:
    #             print("student, student id", name, row[0])
    #             student_id = row[0]
    #             insert_query = """
    #                 INSERT INTO {table} {column}
    #                 VALUES (%s, %s)
    #             """.format(table=table_name, column = columns)

    #             cursor.execute(insert_query, (student_id, face_encoding.tolist()))
    #             connection.commit()

        
    # LOAD THE UNKNOWN FACES
    input_folder_unknown_faces = "../../detected-aligned-faces-mtcnn"
    # input_folder_unknown_faces = "../../test-images"
    file_names_unknown_faces = os.listdir(input_folder_unknown_faces)

    unknown_image = []
    unknown_image_encoding = []

    for file_name in file_names_unknown_faces:
        unknown_image.append(face_recognition.load_image_file(os.path.join(input_folder_unknown_faces, file_name)))

    for i in unknown_image:
        # encoding = face_recognition.face_encodings(i, model="cnn")
        encoding = get_embedding(i)
        unknown_image_encoding.append(encoding[0])  # only one face per image

    # SIMILARITY COMPARISION FACE RECOGNITION

    # Convert embeddings from torch.Tensor to numpy arrays
    #converting the tensor into numpy arrays so that it is supported by compare_faces function
    known_image_encoding_np = [e.detach().cpu().numpy() for e in known_image_encoding]
    unknown_image_encoding_np = [e.detach().cpu().numpy() for e in unknown_image_encoding]


    euclidean_threshold = 1.1  # Euclidean distance threshold (for facenet 0.8 to 1.1) the embedding are not normalized in facenet. (different persons 1.0 to 1.4)
    cosine_threshold = 0.6  # 0.6 to 0.9 same person

    detected_faces = []

    def euclidean_distance(known_embeddings, unknown_embedding):
        """
        known_embeddings: list of np.ndarray (shape: 512,)
        unknown_embedding: np.ndarray (shape: 512,)
        Returns: distances (list of floats, lower = more similar)
        """
        distances = []
        for emb in known_embeddings:
            dist = np.linalg.norm(emb - unknown_embedding)
            distances.append(dist)
        return np.array(distances)


    def l2_normalize(vector):
        norm = np.linalg.norm(vector)
        return vector if norm == 0 else vector/ norm

    def calculate_cosine_similarity(known_embeddings, unknown_embedding):
        known_norm = known_embeddings / np.linalg.norm(known_embeddings, axis=1, keepdims=True)
        unknown_norm = unknown_embedding / np.linalg.norm(unknown_embedding)
        return np.dot(known_norm, unknown_norm)

    def convert_cosine_similarity_to_percentage(cosine_sim):
        cosine_sim = np.clip(cosine_sim, 0, 1)
        return cosine_sim * 100

    for idx, unknown_encoding in enumerate(unknown_image_encoding_np):
        # Compute Euclidean distances
        distances = euclidean_distance(known_image_encoding_np, unknown_encoding)

        cosine_similarity = calculate_cosine_similarity(known_image_encoding_np, unknown_encoding)
        best_cosine_match_index = cosine_similarity.argmax()
        max_cosine_similarity = cosine_similarity[best_cosine_match_index]

        # Find best match
        best_match_index = distances.argmin()  # index with min distance
        min_distance = distances[best_match_index]

        if min_distance <= euclidean_threshold:
            name = known_face_names[best_match_index]
            detected_faces.append(name)
            print(f"Unknown face '{file_names_unknown_faces[idx]}' MATCHED with known face '{name}' (Euclidean distance: {min_distance:.3f})")
        else:
            name = "Unknown Person"
            print(f"Unknown face '{file_names_unknown_faces[idx]}' did NOT match any known face (closest distance: {min_distance:.3f})")

        if max_cosine_similarity >= cosine_threshold:
            name = known_face_names[best_cosine_match_index]
            print(f"Unknown face '{file_names_unknown_faces[idx]}' MATCHED with known face '{name}' (Cosine sim: {max_cosine_similarity:.3f})")
            detected_faces.append(name)
        else:
            name = "Unknown Person"
            print(f"Unknown face '{file_names_unknown_faces[idx]}' did NOT match any known face (closest distance: {max_cosine_similarity:.3f})")
    
    unique_detected_face = list(set(detected_faces))
    print(unique_detected_face)

if __name__ == "__main__":
    face_recognition_facenet()