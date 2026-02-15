# Face Recognition Attendance System

A real-time face detection and recognition system built with Python, designed for automated attendance tracking. The system uses state-of-the-art deep learning models (MTCNN for detection and FaceNet for recognition) to accurately identify individuals.

## Features

- **Real-time Face Detection**: Detects faces from webcam or video files using MTCNN
- **Face Alignment**: Automatically aligns faces for better recognition accuracy
- **Dual Similarity Metrics**: Uses both Euclidean distance and Cosine similarity for robust matching
- **GPU Acceleration**: Supports CUDA for faster processing
- **Database Integration**: PostgreSQL support for storing student information and face encodings
- **High Accuracy**: FaceNet-based embeddings provide reliable face recognition
- **Preprocessing Pipeline**: Handles face cropping, alignment, and normalization

## How It Works

### 3-Phase Recognition Pipeline

1. **Detection Phase** (MTCNN)
   - Captures video from webcam or processes video files
   - Detects faces with bounding boxes and facial landmarks
   - Extracts face crops with minimum quality thresholds

2. **Alignment Phase** (face_alignment)
   - Detects 68 facial landmarks
   - Calculates roll, yaw, and pitch angles
   - Rotates faces to normalize orientation
   - Ensures consistent 160×160 pixel size

3. **Recognition Phase** (FaceNet)
   - Generates 512-dimensional embeddings using InceptionResnetV1
   - Compares unknown faces against known database
   - Uses two similarity metrics:
     - **Euclidean Distance** (threshold: ≤ 1.1)
     - **Cosine Similarity** (threshold: ≥ 0.6)
   - Returns identified individuals

For detailed technical documentation, see [FACE_RECOGNITION_EXPLAINED.md](FACE_RECOGNITION_EXPLAINED.md)

## Requirements

- Python 3.12+
- CUDA-compatible GPU (optional, for acceleration)
- Webcam (for real-time detection)
- PostgreSQL (optional, for database features)

### Key Dependencies

- `opencv-python` - Image processing and video capture
- `torch` & `torchvision` - Deep learning framework
- `facenet-pytorch` - Pre-trained FaceNet model
- `face-recognition` - Face encoding utilities
- `face-alignment` - Facial landmark detection
- `mtcnn` - Face detection
- `psycopg2` - PostgreSQL database connection
- `numpy` - Numerical operations

See [requirements.txt](requirements.txt) for complete list.

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd face-recognition-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up database** (optional)
   - Install PostgreSQL
   - Update database credentials in `db_connection.py`
   - Run the application to auto-create tables

## Usage

### 1. Register Known Faces

Place images of known individuals in the `known_face_crop/` folder:
```
known_face_crop/
├── alice1.jpg
├── alice2.jpg
├── bob1.jpg
└── charlie1.jpg
```

Run the registration script:
```bash
cd src/face_detection_and_recognition
python register_students.py
```

This generates `known_face_encoding_data.pkl` containing face embeddings.

### 2. Real-time Face Detection

Detect faces from webcam:
```bash
python face_detection_mtcnn.py
```

Or from video file:
```bash
python face_detection_mtcnn_test_video.py
```

Press `q` to quit.

### 3. Face Alignment

Align detected faces (runs automatically after detection):
```bash
python align_faces.py
```

### 4. Face Recognition

Identify aligned faces:
```bash
python face_recognition_facenet.py
```

Outputs matched names and similarity scores.

### 5. Complete Real-time Recognition

Run the full pipeline in real-time:
```bash
python face_recognition_realtime.py
```

## Project Structure

```
face-recognition-system/
├── src/
│   ├── face_detection_and_recognition/
│   │   ├── face_detection_mtcnn.py         # MTCNN face detection
│   │   ├── face_detection_mtcnn_test_video.py
│   │   ├── face_detection_realtime.py      # Real-time detection
│   │   ├── align_faces.py                  # Face alignment
│   │   ├── align_faces_realtime.py
│   │   ├── face_recognition_facenet.py     # FaceNet recognition
│   │   ├── face_recognition_realtime.py    # Real-time recognition
│   │   ├── register_students.py            # Register known faces
│   │   ├── db_connection.py                # Database utilities
│   │   └── main.py                         # Main entry point
│   └── preprocessing-phase/
│       └── preprocessing_known_face.py     # Preprocess training data
├── known_face_crop/                        # Known face images
├── detected-faces-mtcnn/                   # Detected face crops
├── detected-aligned-faces-mtcnn/           # Aligned faces
├── requirements.txt
├── pyproject.toml
├── README.md
└── FACE_RECOGNITION_EXPLAINED.md          # Technical documentation
```

## Technical Details

### Face Detection (MTCNN)
- Multi-task Cascaded Convolutional Networks
- Detects faces of various sizes and orientations
- Returns bounding boxes, confidence scores, and 5 facial landmarks
- GPU-accelerated processing

### Face Alignment
- Uses 68-point facial landmark detection
- Corrects roll angle based on eye positions
- Estimates yaw and pitch for pose analysis
- Ensures consistent face orientation

### Face Recognition (FaceNet)
- **Model**: InceptionResnetV1 pre-trained on VGGFace2
- **Embedding Size**: 512 dimensions
- **Similarity Metrics**:
  - **Euclidean Distance**: $d = ||v_1 - v_2||_2$
  - **Cosine Similarity**: $sim = \frac{v_1 \cdot v_2}{||v_1|| \times ||v_2||}$

### Recognition Thresholds
- **Euclidean Distance**: ≤ 1.1 for match (typically 0.8-1.1 for same person)
- **Cosine Similarity**: ≥ 0.6 for match (0.6-0.9 for same person)

## Use Cases

- **Educational Institutions**: Automated attendance tracking
- **Corporate Offices**: Employee attendance management
- **Events**: Quick check-in systems
- **Security**: Access control systems
- **Time Tracking**: Workforce management

## Configuration

### Adjust Detection Parameters

In `face_detection_mtcnn.py`:
```python
# Minimum face size
min_face_detection_size = 20

# FPS settings
cap.set(cv2.CAP_PROP_FPS, 60)
```

### Adjust Recognition Thresholds

In `face_recognition_facenet.py`:
```python
# Lower for stricter matching, higher for looser matching
euclidean_threshold = 1.1
cosine_threshold = 0.6
```

## Database Schema

### Students Table
```sql
CREATE TABLE students (
    student_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    enrollment_status VARCHAR(100)
);
```

### Face Encodings Table
```sql
CREATE TABLE students_face_encoding (
    face_encoding_id SERIAL PRIMARY KEY,
    student_id SERIAL NOT NULL,
    face_encoding DOUBLE PRECISION[],
    FOREIGN KEY(student_id) REFERENCES students(student_id)
);
```

## Troubleshooting

**Issue**: Low FPS during real-time detection
- **Solution**: Enable GPU acceleration, reduce video resolution, or decrease FPS target

**Issue**: Poor recognition accuracy
- **Solution**: Ensure faces are well-lit and frontal during registration, add more sample images per person

**Issue**: "No face detected" during alignment
- **Solution**: Check if face crops are too small (< 160×160), adjust minimum face size threshold

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size, use CPU mode, or process fewer faces simultaneously

## Future Improvements

- [ ] Add anti-spoofing (liveness detection)
- [ ] Web interface for attendance management
- [ ] Mobile app integration
- [ ] Multi-camera support
- [ ] Attendance report generation
- [ ] Real-time notifications
- [ ] Cloud-based deployment
- [ ] Face mask detection support

## License

This project is available for educational and personal use.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or support, please open an issue in the repository.

---

**Note**: Ensure compliance with local privacy laws and regulations when deploying face recognition systems.
