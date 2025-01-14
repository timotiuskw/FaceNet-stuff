import cv2
import numpy as np
import pickle
import onnxruntime  # ONNX Runtime for running FaceNet ONNX model
from scipy.spatial.distance import cosine
from PIL import Image
import time
from yolov8 import YOLOv8  # Menggunakan YOLOv8 dari kode pertama

# Load YOLOv8-face model (dalam format ONNX) dari kodingan pertama
model_path = "D:/Tugas Kuliah/Bengkel Koding/Proyek VA venv/pythonkuenv/yolov8n-face.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.6, iou_thres=0.6)

# Load FaceNet ONNX model for face recognition
facenet_session = onnxruntime.InferenceSession('D:/Tugas Kuliah/Bengkel Koding/Proyek VA venv/pythonkuenv/facenet_model.onnx')

# Load Emotion Detection ONNX model
emotion_model_path = 'D:/Tugas Kuliah/Bengkel Koding/Proyek VA venv/pythonkuenv/emotion.onnx'
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
    img = img.resize((160, 160))
    img = np.array(img).astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

# Function to extract face embedding using FaceNet ONNX
def extract_embedding(face_image):
    preprocessed_face = preprocess_face(face_image)
    embedding = facenet_session.run(None, {'input': preprocessed_face})[0].flatten()
    return embedding

# Function to match the face embedding with the database and return similarity
def match_face(embedding, threshold=0.6):
    min_distance = float('inf')
    best_match = "Unknown"
    best_similarity = 0

    for person_name, saved_embedding in embeddings_db.items():
        distance = cosine(embedding, saved_embedding)
        similarity = 1 - distance

        if distance < min_distance and distance < threshold:
            min_distance = distance
            best_match = person_name
            best_similarity = similarity

    return best_match, best_similarity

# Function to preprocess image for emotion detection
def preprocess_emotion_image(face_image):
    img = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

# Function to predict emotion using the emotion detection ONNX model
def predict_emotion(face_image):
    preprocessed_img = preprocess_emotion_image(face_image)
    input_name = emotion_session.get_inputs()[0].name
    output_name = emotion_session.get_outputs()[0].name
    result = emotion_session.run([output_name], {input_name: preprocessed_img.astype(np.float32)})
    predicted_class = np.argmax(result[0])
    return predicted_class

def class_to_emotion(predicted_emotion):
    res_dict = {0: 'angry', 1: 'disgusted', 2: 'fearful', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprised'}
    return res_dict[predicted_emotion]

# Open webcam using OpenCV
cap = cv2.VideoCapture(0)
framecount = 0
fps = 0
prev_time = time.time()

# Variables to calculate average metrics
fps_array = []
inference_times = []
similarities = []

time_window_start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Start timing preprocess
    start_time = time.time()

    # Detect faces in the frame using YOLOv8-face (kodingan pertama)
    boxes, scores, class_ids = yolov8_detector(frame)

    # Start timing inference for FaceNet
    start_inference_time = time.time()

    # Process and extract embedding from cropped face
    for box in boxes:
        cropped_face = crop_face(frame, box)
        embedding = extract_embedding(cropped_face)
        name, similarity = match_face(embedding)
        predicted_emotion = predict_emotion(cropped_face)
        emotion_text = class_to_emotion(predicted_emotion)

        # Save cosine similarity for average calculation
        if similarity is not None:
            similarities.append(similarity)

        x1, y1, x2, y2 = map(int, box)
        if name != "Unknown" and similarity > 0.7:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            similarity_text = f"{name}: {similarity:.2f}, Emotion: {emotion_text}"
            cv2.putText(frame, similarity_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Calculate inference time for FaceNet
    inference_time = time.time() - start_inference_time
    inference_times.append(inference_time * 1000)  # Convert to ms

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    fps_array.append(fps)
    prev_time = current_time

    # Display FPS on the top left of the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the result in real-time
    cv2.imshow('Face Recognition (YOLOv8 + FaceNet ONNX + Emotion Detection)', frame)

    # Press 'q' to exit and print the averages and the arrays
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Calculate averages
        avg_fps = sum(fps_array) / len(fps_array) if fps_array else 0
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0

        # Print average values
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Average Inference Time (FaceNet): {avg_inference_time:.2f} ms")
        print(f"Average Cosine Similarity: {avg_similarity:.2f}")
        
        # Print the arrays
        print("\nFPS Array:", fps_array)
        print("Inference Times Array:", inference_times)
        print("Cosine Similarity Array:", similarities)

        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
