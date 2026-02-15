import face_recognition
import os
from pathlib import Path
from db_connection import db_connection
import re
import numpy as np

# DB CONNECTION

connection = db_connection()
cursor = connection.cursor()

# creating a table of students info if not exists

if connection is None:
    print(f"Cant access the database")
else:
    create_students_query = """
        CREATE TABLE IF NOT EXISTS students (
        student_id SERIAL PRIMARY KEY,
        name VARCHAR(100),
        enrollment_status VARCHAR(100)
        );
    """
    cursor.execute(create_students_query)

    create_students_face_encoding_query = """
        CREATE TABLE IF NOT EXISTS students_face_encoding (
            face_encoding_id SERIAL PRIMARY KEY,
            student_id SERIAL NOT NULL,
            face_encoding DOUBLE PRECISION[],
            FOREIGN KEY(student_id) REFERENCES students(student_id)
        );
    """
    cursor.execute(create_students_face_encoding_query)

    connection.commit()
    print("students and face encoding table created successfully.")

# LOADING THE KNOWN FACES

input_folder_known_face = "../../known_face_crop"
file_names_known_face = os.listdir(input_folder_known_face)

known_image = []
known_face_names = []
known_image_encoding = []

for file_name in file_names_known_face:
    known_image.append(face_recognition.load_image_file(os.path.join(input_folder_known_face, file_name)))
    path_object = Path(file_name)
    name_only = re.sub(r'\d+', '', path_object.stem)
    known_face_names.append(name_only)

for i in known_image:
    encoding = face_recognition.face_encodings(i, model="cnn")      #cnn based on ResNet-34
    known_image_encoding.append(encoding[0])  # only one face per image

#unique name arrays
unique_names = list(set(known_face_names))
print(unique_names)

# INSERITNG THE DETAILS OF STUDENTS IN STUDENTS TABLE
for i, name in enumerate(unique_names):
    student_table = "students"
    students_column = "(name, enrollment_status)"
    enrollment_status = "enrolled"
    insert_query = """
        INSERT INTO {table} {columns}
        VALUES (%s, %s)
    """.format(table = student_table, columns = students_column)
    cursor.execute(insert_query, (name, enrollment_status))
    connection.commit()


# INSERTING THE FACE ENCODING IN STUDENTS FACE ENCODING TABLE
for name, face_encoding in zip(known_face_names, known_image_encoding):
    select_query = """SELECT student_id from students where name = %s"""
    cursor.execute(select_query, (name,))
    result = cursor.fetchall()
    print(result)

    table_name = "students_face_encoding"
    columns = "(student_id, face_encoding)"

    if result:
        for row in result:
            print("student, student id", name, row[0])
            student_id = row[0]
            insert_query = """
                INSERT INTO {table} {column}
                VALUES (%s, %s)
            """.format(table=table_name, column = columns)

            cursor.execute(insert_query, (student_id, face_encoding.tolist()))
            connection.commit()

    
# LOAD THE UNKNOWN FACES

input_folder_unknown_faces = "../../detected-aligned-faces-mtcnn"
# input_folder_unknown_faces = "../../test-images"
file_names_unknown_faces = os.listdir(input_folder_unknown_faces)

unknown_image = []
unknown_image_encoding = []

for file_name in file_names_unknown_faces:
    unknown_image.append(face_recognition.load_image_file(os.path.join(input_folder_unknown_faces, file_name)))

for i in unknown_image:
    encoding = face_recognition.face_encodings(i, model="cnn")
    unknown_image_encoding.append(encoding[0])  # only one face per image

# SIMILARITY COMPARISION FACE RECOGNITION

for idx, unknown_encoding in enumerate(unknown_image_encoding):
    # Compare with known faces
    matches = face_recognition.compare_faces(known_face_encodings=known_image_encoding, 
                                             face_encoding_to_check=unknown_encoding)

    # Compute distances to all known faces/ similarity
    distances = face_recognition.face_distance(known_image_encoding, unknown_encoding)
    
    threshold = 0.6

    name = "Unknown Person"
    best_match_index = distances.argmin()  # index of closest match
    distance_value = distances[best_match_index]
    
    if matches[best_match_index] and distance_value <= threshold:
        name = known_face_names[best_match_index]
        print(f"Unknown face '{file_names_unknown_faces[idx]}' MATCHED with known face '{name}' (distance: {distance_value:.3f})")
    else:
        print(f"Unknown face '{file_names_unknown_faces[idx]}' did NOT match any known face (closest distance: {distance_value:.3f})")

