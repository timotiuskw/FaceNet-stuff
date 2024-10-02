from ultralytics import YOLO
import cv2
import os

# Load model YOLOv8-face
model = YOLO('yolov8n-face.pt')

def crop_face(image_path, save_path):
    image = cv2.imread(image_path)
    results = model(image)

    # Ambil bounding box pertama (dengan asumsi hanya ada satu wajah)
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            face_crop = image[y1:y2, x1:x2]
            cv2.imwrite(save_path, face_crop)
            return save_path

from facenet_pytorch import InceptionResnetV1
from PIL import Image
import torch
import numpy as np

# Load pretrained FaceNet model
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

def extract_embedding(face_image_path):
    img = Image.open(face_image_path)
    img = img.resize((160, 160))  # FaceNet membutuhkan input 160x160
    img = np.array(img).astype(np.float32)
    img = (img - 127.5) / 128.0  # Normalisasi seperti yang dibutuhkan oleh FaceNet

    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # Ubah menjadi tensor

    with torch.no_grad():
        embedding = facenet_model(img_tensor).numpy().flatten()

    return embedding

import pickle

def save_embeddings(dataset_folder, output_file='embeddings.pkl'):
    embeddings_db = {}
    for person_name in os.listdir(dataset_folder):
        person_folder = os.path.join(dataset_folder, person_name)
        if os.path.isdir(person_folder):
            image_file = os.path.join(person_folder, os.listdir(person_folder)[0])  # Ambil 1 gambar tiap orang
            cropped_face = crop_face(image_file, f"cropped_{person_name}.jpg")
            embedding = extract_embedding(cropped_face)
            embeddings_db[person_name] = embedding

    with open(output_file, 'wb') as f:
        pickle.dump(embeddings_db, f)

# Folder dataset berisi subfolder dengan nama orang
save_embeddings('dataset/SELF')
