import cv2
import numpy as np
import pickle
import onnxruntime  # ONNX Runtime for running FaceNet ONNX model
from scipy.spatial.distance import cosine
from PIL import Image
import time
import torch
from yolov8 import YOLOv8  # Menggunakan YOLOv8 dari kode pertama

# Load YOLOv8-face model (dalam format ONNX) dari kodingan pertama
model_path = "D:/Tugas Kuliah/Bengkel Koding/Proyek VA venv/pythonkuenv/yolov8n-face.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.6, iou_thres=0.6)

# Load FaceNet ONNX model for face recognition
facenet_session = onnxruntime.InferenceSession('D:/Tugas Kuliah/Bengkel Koding/Proyek VA venv/pythonkuenv/facenet_model.onnx')

# Load Emotion Detection ONNX model
emotion_model_path = 'D:/Tugas Kuliah/Bengkel Koding/Proyek VA venv/pythonkuenv/emotion.onnx'  # Ganti dengan path model emosi Anda
emotion_session = onnxruntime.InferenceSession(emotion_model_path)

# Load the saved embeddings from the .pkl file
with open('D:/Tugas Kuliah/Bengkel Koding/Proyek VA venv/pythonkuenv/yudyud.pkl', 'rb') as f:
    embeddings_db = pickle.load(f)

# Function to crop face using YOLOv8-face
def crop_face(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    face_crop = image[y1:y2, x1:x2]
    return face_crop

# Function to preprocess face image for FaceNet ONNX
def preprocess_face(face_image):
    img = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    img = img.resize((160, 160))  # FaceNet input size is 160x160
    img = np.array(img).astype(np.float32)
    img = (img - 127.5) / 128.0  # Normalize the image
    img = np.transpose(img, (2, 0, 1))  # Change from HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to extract face embedding using FaceNet ONNX
def extract_embedding(face_image):
    preprocessed_face = preprocess_face(face_image)

    # Run the ONNX model to extract embeddings
    embedding = facenet_session.run(None, {'input': preprocessed_face})[0].flatten()

    return embedding

# Function to match the face embedding with the database and return similarity
def match_face(embedding, threshold=0.6):
    min_distance = float('inf')
    best_match = "Unknown"
    best_similarity = 0  # Initialize similarity score

    for person_name, saved_embedding in embeddings_db.items():
        distance = cosine(embedding, saved_embedding)
        similarity = 1 - distance  # Convert distance to similarity (1 means identical, 0 means completely different)
        
        if distance < min_distance and distance < threshold:
            min_distance = distance
            best_match = person_name
            best_similarity = similarity  # Store the best similarity score

    return best_match, best_similarity

# Function to preprocess image for emotion detection
def preprocess_emotion_image(face_image):
    img = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(img, (48, 48))  # Resize to the input size of emotion model
    img = img.astype(np.float32) / 255.0  # Normalize
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict emotion using the emotion detection ONNX model
def predict_emotion(face_image):
    preprocessed_img = preprocess_emotion_image(face_image)

    input_name = emotion_session.get_inputs()[0].name
    output_name = emotion_session.get_outputs()[0].name

    # Run inference
    result = emotion_session.run([output_name], {input_name: preprocessed_img.astype(np.float32)})
    
    predicted_class = np.argmax(result[0])
    return predicted_class

def class_to_emotion(predicted_emotion):
    res_dict = {0: 'angry',
                1: 'disgusted',
                2: 'fearful',
                3: 'happy',
                4: 'neutral',
                5: 'sad',
                6: 'surprised'}
    return res_dict[predicted_emotion]

# Initialize variables to store detections and time tracking
last_detections = []  # To store bounding boxes, names
entry_times = {}  # Store entry time when someone is first detected
total_time_in_frame = {}  # Store total time for each person
last_seen = {}  # Store the last time the person was seen
max_frame_without_detection = 30  # Tolerate up to 30 frames without detection

# Open webcam using OpenCV
cap = cv2.VideoCapture(0)
framecount = 0

while True:
    framecount += 1
    ret, frame = cap.read()
    if not ret:
        break

    if framecount % 1 == 0:
        # Detect faces in the frame using YOLOv8-face (kodingan pertama)
        boxes, scores, class_ids = yolov8_detector(frame)

        current_frame_names = []  # Track names detected in the current frame

        if len(boxes) > 0:
            last_detections = []  # Clear previous detections
            for box in boxes:
                # Crop face based on bounding box
                cropped_face = crop_face(frame, box)

                # Extract embedding for the cropped face using FaceNet ONNX
                embedding = extract_embedding(cropped_face)

                # Match the embedding with the saved database
                name, similarity = match_face(embedding)
                
                # Predict emotion for the cropped face
                predicted_emotion = predict_emotion(cropped_face)
                emotion_text = class_to_emotion(predicted_emotion)

                # Store the detection (bounding box, name, similarity, emotion)
                last_detections.append((box, name, similarity, emotion_text))
                current_frame_names.append(name)

                # Record entry time if the person is detected for the first time
                if name != "Unknown":
                    if name not in entry_times:
                        entry_times[name] = time.time()  # First time detected
                        total_time_in_frame[name] = 0  # Initialize total time
                    elif name in last_seen:  # Person was seen before
                        # Update total time by adding time since last seen
                        total_time_in_frame[name] += time.time() - last_seen[name]

                    # Update last seen time to current time
                    last_seen[name] = time.time()

    # Check for people who were not detected in the current frame
    for name in list(last_seen.keys()):
        if name not in current_frame_names:
            # Update total time for those who were not seen in this frame
            total_time_in_frame[name] += time.time() - last_seen[name]
            del last_seen[name]  # Remove from last_seen since they are no longer in the frame

    # Display last detected faces with their respective total time in frame and similarity
    if len(last_detections) > 0:
        for detection in last_detections:
            box, name, similarity, emotion = detection
            x1, y1, x2, y2 = map(int, box)
            
            # Display name, similarity, and emotion in the bounding box
            if name != "Unknown" and similarity > 0.7:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                total_time = int(total_time_in_frame.get(name, 0))
                similarity_text = f"{name}: {similarity:.2f}, Emotion: {emotion}"  # Display name, similarity, and emotion
                cv2.putText(frame, similarity_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Time: {total_time}s", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the result in real-time
    cv2.imshow('Face Recognition (YOLOv8 + FaceNet ONNX + Emotion Detection)', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
