import cv2
import torch
import pickle
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cosine
import numpy as np
from PIL import Image
import time

# Load YOLOv8-face model and move to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('D:\Tugas Kuliah\Bengkel Koding\Proyek VA venv\pythonkuenv\yolov8n-face.pt').to(device)

# Load FaceNet model and move to GPU
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load the saved embeddings from the .pkl file
with open('D:\Tugas Kuliah\Bengkel Koding\Proyek VA venv\pythonkuenv\embeddings.pkl', 'rb') as f:
    embeddings_db = pickle.load(f)

# Function to crop face using YOLOv8-face
def crop_face(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    face_crop = image[y1:y2, x1:x2]
    return face_crop

# Function to extract face embedding using FaceNet and GPU
def extract_embedding(face_image):
    img = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    img = img.resize((160, 160))
    img = np.array(img).astype(np.float32)
    img = (img - 127.5) / 128.0  # Normalize

    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = facenet_model(img_tensor).cpu().numpy().flatten()

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

    if framecount % 30 == 0:
        # Detect faces in the frame
        results = model(frame)
        
        current_frame_names = []  # Track names detected in the current frame

        if len(results[0].boxes) > 0:
            last_detections = []  # Clear previous detections
            for result in results:
                for box in result.boxes.xyxy:
                    # Crop face based on bounding box
                    cropped_face = crop_face(frame, box)

                    # Extract embedding for the cropped face
                    embedding = extract_embedding(cropped_face)

                    # Match the embedding with the saved database
                    name, similarity = match_face(embedding)
                    
                    # Store the detection (bounding box, name, and similarity)
                    last_detections.append((box, name, similarity))
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
            box, name, similarity = detection
            x1, y1, x2, y2 = map(int, box)
            
            
            # Display name and similarity in the bounding box
            if name != "Unknown" and similarity > 0.7:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                total_time = int(total_time_in_frame.get(name, 0))
                similarity_text = f"{name}: {similarity:.2f}"  # Display name and similarity score
                cv2.putText(frame, similarity_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Time: {total_time}s", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


    # Show the result in real-time
    cv2.imshow('Face Recognition', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
