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
model = YOLO('D:\\Tugas Kuliah\\Bengkel Koding\\Proyek VA venv\\pythonkuenv\\yolov8n-face.pt').to(device)

# Load FaceNet model and move to GPU
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load the saved embeddings from the .pkl file
with open('embeddings.pkl', 'rb') as f:
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

# Variables for logging real-time performance
inference_times = []
average_similarities = []  # Log average cosine similarity per frame

# Open webcam using OpenCV
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Start measuring inference time
    start_inference = time.time()

    # Detect faces in the frame
    results = model(frame)
    current_frame_names = []  # Track names detected in the current frame
    frame_similarities = []  # Track similarities for the current frame

    if len(results[0].boxes) > 0:
        for result in results:
            for box in result.boxes.xyxy:
                # Crop face based on bounding box
                cropped_face = crop_face(frame, box)

                # Extract embedding for the cropped face
                embedding = extract_embedding(cropped_face)

                # Match the embedding with the saved database
                name, similarity = match_face(embedding)

                # Store similarity for calculating average similarity
                frame_similarities.append(similarity)

                # Draw bounding box and name on the frame
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name}: {similarity:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Calculate average similarity for the frame
    if frame_similarities:
        avg_similarity = sum(frame_similarities) / len(frame_similarities)
        average_similarities.append(avg_similarity)  # Log average similarity for statistics
        cv2.putText(frame, f"Avg Similarity: {avg_similarity:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Calculate and display FPS
    elapsed_time = time.time() - start_inference
    inference_times.append(elapsed_time)  # Log inference time
    fps = 1 / elapsed_time if elapsed_time > 0 else 0
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display frame
    cv2.imshow('Face Recognition', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate and print statistics after the loop
average_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
average_similarity_overall = sum(average_similarities) / len(average_similarities) if average_similarities else 0

print(f"Average Inference Time: {average_inference_time:.4f} seconds")
print(f"Average FPS: {1 / average_inference_time:.2f}" if average_inference_time > 0 else "Average FPS: 0")
print(f"Average Cosine Similarity: {average_similarity_overall:.2f}")

# Release webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
