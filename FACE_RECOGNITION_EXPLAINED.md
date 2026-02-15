# Face Recognition System - Technical Documentation

## Project Overview

This project implements a complete face recognition pipeline with three main phases:
1. **Face Detection** - Detecting faces in images/video
2. **Face Alignment** - Normalizing face orientation
3. **Face Recognition** - Identifying faces through similarity comparison

---

## Phase 1: Face Detection (MTCNN)

**File:** [face_detection_mtcnn.py](src/face_detection_and_recognition/face_detection_mtcnn.py)

### Process
- Uses **MTCNN** (Multi-task Cascaded Convolutional Networks) for face detection
- Captures video from webcam (or processes video files)
- Detects faces with bounding boxes and facial landmarks
- Outputs:
  - Bounding box coordinates (x1, y1, x2, y2)
  - Confidence scores
  - 5 facial landmarks (eyes, nose, mouth corners)
- Crops and saves detected faces to `detected-faces-mtcnn/` folder

### Key Features
- GPU acceleration support (CUDA if available)
- Minimum face size threshold (20x20 pixels)
- Real-time detection with adjustable FPS

---

## Phase 2: Face Alignment

**File:** [align_faces.py](src/face_detection_and_recognition/align_faces.py)

### Purpose
Face alignment is crucial for accurate recognition because faces can be captured at different angles. This phase normalizes face orientation.

### Process

1. **Landmark Detection**
   - Uses **face_alignment** library with 2D landmarks
   - Detects 68 facial landmarks per face

2. **Eye Position Calculation**
   - Left eye: Average of landmarks 36-42
   - Right eye: Average of landmarks 42-48
   - Eye center: Midpoint between both eyes

3. **Roll Angle Correction**
   ```python
   dy = right_eye[1] - left_eye[1]
   dx = right_eye[0] - left_eye[0]
   roll_angle = arctan2(dy, dx)
   ```
   - Rotates face to make eyes horizontal
   - Uses affine transformation with eye center as rotation point

4. **Pose Estimation (Yaw & Pitch)**
   ```python
   yaw_angle = arctan(horizontal_nose_deviation / eye_distance)
   pitch_angle = -arctan(vertical_nose_deviation / eye_distance)
   ```
   - Estimates head rotation angles (for logging/analysis)
   - Not currently used for correction but could be extended

5. **Output**
   - Aligned faces saved to `detected-aligned-faces-mtcnn/` folder
   - Minimum size: 160x160 pixels (resized if smaller)

---

## Phase 3: Face Recognition (FaceNet)

**File:** [face_recognition_facenet.py](src/face_detection_and_recognition/face_recognition_facenet.py)

### Model: InceptionResnetV1 (FaceNet)

**Architecture:**
- Pre-trained on VGGFace2 dataset
- Transforms faces into 512-dimensional embeddings
- Maps faces to points in high-dimensional space where:
  - Similar faces are close together
  - Different faces are far apart

### Embedding Generation Process

```python
def get_embedding(image_path):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),           # Normalize size
        transforms.ToTensor(),                   # Convert to tensor
        transforms.Normalize([0.5]*3, [0.5]*3)   # Normalize pixel values
    ])
    img = Image.fromarray(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)     # Add batch dimension
    
    with torch.no_grad():
        embedding = facenet_model(img_tensor)     # Generate 512-dim vector
    return embedding
```

**Steps:**
1. Resize image to 160x160 pixels (model's expected input)
2. Convert to PyTorch tensor
3. Normalize pixel values from [0, 255] to [-1, 1]
4. Pass through FaceNet model
5. Output: 512-dimensional embedding vector

---

## Similarity Comparison: The Core Recognition Logic

This is the critical part where the system determines if an unknown face matches a known face.

### Data Preparation

**Known Faces:**
- Pre-registered students/individuals
- Face embeddings stored in `known_face_encoding_data.pkl`
- Format: `{name: embedding_vector}`

**Unknown Faces:**
- Detected and aligned faces from Phase 1 & 2
- Embeddings generated in real-time

### Two Similarity Metrics

The system uses **two complementary methods** to compare embeddings:

---

#### 1. Euclidean Distance

**Concept:** Measures straight-line distance between two points in 512-dimensional space.

**Formula:**
$$d = \sqrt{\sum_{i=1}^{512}(known_i - unknown_i)^2}$$

Or in simpler terms:
$$d = ||known - unknown||_2$$

**Implementation:**
```python
def euclidean_distance(known_embeddings, unknown_embedding):
    distances = []
    for emb in known_embeddings:
        dist = np.linalg.norm(emb - unknown_embedding)
        distances.append(dist)
    return np.array(distances)
```

**Process:**
1. Calculate distance from unknown face to **each** known face
2. Find the **minimum distance** (closest match)
3. If `min_distance <= 1.1` → **MATCH FOUND**
4. Otherwise → **UNKNOWN PERSON**

**Threshold:** `1.1`
- Distance < 1.1: Same person
- Distance 1.0-1.4: Different persons (typically)

**Advantages:**
- Intuitive: smaller distance = more similar
- Works well with unnormalized embeddings (FaceNet)

---

#### 2. Cosine Similarity

**Concept:** Measures the angle between two vectors, ignoring magnitude.

**Formula:**
$$\text{cosine\_sim} = \frac{known \cdot unknown}{||known|| \times ||unknown||}$$

Where:
- $known \cdot unknown$ = dot product
- $||known||$ = vector magnitude (L2 norm)

**Implementation:**
```python
def calculate_cosine_similarity(known_embeddings, unknown_embedding):
    # Normalize known embeddings (divide by magnitude)
    known_norm = known_embeddings / np.linalg.norm(known_embeddings, axis=1, keepdims=True)
    
    # Normalize unknown embedding
    unknown_norm = unknown_embedding / np.linalg.norm(unknown_embedding)
    
    # Compute dot product (cosine similarity)
    return np.dot(known_norm, unknown_norm)
```

**Process:**
1. Normalize all vectors to unit length
2. Calculate cosine similarity with **each** known face
3. Find the **maximum similarity** (best match)
4. If `max_similarity >= 0.6` → **MATCH FOUND**
5. Otherwise → **UNKNOWN PERSON**

**Threshold:** `0.6`
- Similarity 0.6-0.9: Same person
- Similarity range: [-1, 1] where 1 = identical, -1 = opposite

**Advantages:**
- Less sensitive to embedding magnitude
- Better for high-dimensional spaces
- Can be converted to percentage: `similarity × 100`

---

### Complete Recognition Flow

```python
for idx, unknown_encoding in enumerate(unknown_image_encoding_np):
    # METHOD 1: Euclidean Distance
    distances = euclidean_distance(known_image_encoding_np, unknown_encoding)
    best_match_index = distances.argmin()  # Find closest match
    min_distance = distances[best_match_index]
    
    if min_distance <= 1.1:
        name = known_face_names[best_match_index]
        print(f"MATCHED: {name} (Euclidean: {min_distance:.3f})")
        detected_faces.append(name)
    else:
        print(f"NO MATCH (Euclidean: {min_distance:.3f})")
    
    # METHOD 2: Cosine Similarity
    cosine_similarity = calculate_cosine_similarity(known_image_encoding_np, unknown_encoding)
    best_cosine_index = cosine_similarity.argmax()  # Find best match
    max_cosine_similarity = cosine_similarity[best_cosine_index]
    
    if max_cosine_similarity >= 0.6:
        name = known_face_names[best_cosine_index]
        print(f"MATCHED: {name} (Cosine: {max_cosine_similarity:.3f})")
        detected_faces.append(name)
    else:
        print(f"NO MATCH (Cosine: {max_cosine_similarity:.3f})")

# Remove duplicates
unique_detected_faces = list(set(detected_faces))
```

---

## Why Two Methods?

Using both Euclidean distance and Cosine similarity provides **redundancy and robustness**:

1. **Different Perspectives:**
   - Euclidean: Absolute distance in space
   - Cosine: Directional similarity (angle)

2. **Complementary Strengths:**
   - Euclidean works well when embedding magnitudes are meaningful
   - Cosine normalizes magnitude, focusing on direction

3. **Validation:**
   - If both methods agree → high confidence
   - If only one matches → moderate confidence
   - If neither matches → likely unknown person

---

## Example Calculation

**Scenario:** Comparing unknown face with 3 known faces

**Known Embeddings:**
- Alice: [0.5, 0.3, 0.8, ..., 0.2] (512 dimensions)
- Bob: [0.1, 0.9, 0.4, ..., 0.7]
- Charlie: [0.6, 0.4, 0.7, ..., 0.3]

**Unknown Embedding:**
- [0.52, 0.31, 0.79, ..., 0.21]

**Euclidean Distances:**
- Alice: 0.85 ✓ (< 1.1, MATCH!)
- Bob: 2.3 ✗
- Charlie: 1.5 ✗

**Cosine Similarities:**
- Alice: 0.92 ✓ (> 0.6, MATCH!)
- Bob: 0.45 ✗
- Charlie: 0.55 ✗

**Result:** **Identified as Alice** (both methods agree)

---

## Database Integration (Optional)

The code includes provisions for PostgreSQL database storage:

- **students** table: Store student names and enrollment status
- **students_face_encoding** table: Store embeddings linked to students

This allows:
- Persistent storage of known faces
- Easy addition of new students
- Tracking enrollment status
- Scalability for large datasets

---

## Summary

### Detection → Alignment → Recognition Pipeline

1. **MTCNN** detects faces and landmarks
2. **Face Alignment** normalizes orientation using eye positions
3. **FaceNet** generates 512-dimensional embeddings
4. **Two similarity metrics** compare embeddings:
   - **Euclidean Distance:** Measures absolute distance (threshold: 1.1)
   - **Cosine Similarity:** Measures directional similarity (threshold: 0.6)
5. **Best match** above threshold → Identity confirmed
6. **No match** → Unknown person

This dual-metric approach ensures robust and accurate face recognition even with variations in lighting, expression, and minor pose differences.
