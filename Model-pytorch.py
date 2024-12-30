from ultralytics import YOLO
import cv2
import os
from PIL import Image
import pillow_heif
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
import pickle
import time  # Untuk mencatat waktu

# Load YOLOv8-face model
model = YOLO('D:\\Tugas Kuliah\\Bengkel Koding\\Proyek VA venv\\pythonkuenv\\yolov8n-face.pt')

# Fungsi untuk membaca file HEIC dan mengonversinya ke format yang dapat dibaca oleh OpenCV
def read_heic_image(image_path):
    heif_file = pillow_heif.read_heif(image_path)
    image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode
    )
    # Konversi gambar ke format OpenCV (BGR)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

# Fungsi untuk memuat gambar (HEIC atau format lain)
def load_image(image_path):
    if image_path.lower().endswith(".heic"):
        return read_heic_image(image_path)
    else:
        return cv2.imread(image_path)

# Fungsi untuk mendeteksi dan memotong wajah dari gambar
def crop_face(image_path, save_path):
    image = load_image(image_path)
    if image is None:
        print(f"Error: Tidak dapat memuat gambar dari path: {image_path}")
        return None
    results = model(image)

    # Ambil bounding box pertama (asumsi hanya ada satu wajah)
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            face_crop = image[y1:y2, x1:x2]
            cv2.imwrite(save_path, face_crop)
            return save_path
    print(f"No faces detected in image: {image_path}")
    return None

# Load FaceNet model untuk ekstraksi embedding
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

# Fungsi untuk mengekstrak embedding dari gambar wajah
def extract_embedding(face_image_path):
    img = Image.open(face_image_path)
    img = img.resize((160, 160))  # FaceNet membutuhkan input ukuran 160x160
    img = np.array(img).astype(np.float32)
    img = (img - 127.5) / 128.0  # Normalisasi untuk FaceNet

    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # Ubah ke tensor

    with torch.no_grad():
        embedding = facenet_model(img_tensor).numpy().flatten()

    return embedding

# Fungsi untuk menyimpan embedding wajah ke dalam file pickle
def save_embeddings(dataset_folder, output_file='embeddings.pkl'):
    start_time = time.time()  # Catat waktu mulai
    embeddings_db = {}
    for image_file in os.listdir(dataset_folder):
        image_path = os.path.join(dataset_folder, image_file)
        if os.path.isfile(image_path):
            person_name = os.path.splitext(image_file)[0]  # Nama orang diambil dari nama file tanpa ekstensi
            cropped_face = crop_face(image_path, f"cropped_{person_name}.jpg")
            if cropped_face:  # Pastikan wajah berhasil dipotong
                embedding = extract_embedding(cropped_face)
                embeddings_db[person_name] = embedding

    # Simpan embedding ke dalam file pickle
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings_db, f)

    end_time = time.time()  # Catat waktu selesai
    total_time = end_time - start_time
    print(f"Proses selesai. Waktu yang dibutuhkan: {total_time:.2f} detik.")

# Menyimpan embedding dari folder dataset
save_embeddings('D:/Tugas Kuliah/Bengkel Koding/Proyek VA venv/pythonkuenv/Scripts/train')
