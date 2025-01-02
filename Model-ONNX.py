import os
import pickle
import numpy as np
from PIL import Image
import torch
import cv2
from facenet_pytorch import InceptionResnetV1
from yolov8 import YOLOv8
import time
import pillow_heif

# Load YOLOv8-face ONNX model
model_path = "D:/Tugas Kuliah/Bengkel Koding/Proyek VA venv/pythonkuenv/yolov8n-face.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.6, iou_thres=0.6)

# Fungsi untuk memuat gambar (HEIC atau format lain)
def load_image(image_path):
    if image_path.lower().endswith(".heic"):
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
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        return cv2.imread(image_path)

# Fungsi untuk mendeteksi dan memotong wajah dari gambar
def crop_face(image_path):
    image = load_image(image_path)
    if image is None:
        print(f"Error: Tidak dapat memuat gambar dari path: {image_path}")
        return None

    # Deteksi wajah menggunakan YOLOv8
    results = yolov8_detector(image)  # Gunakan objek sebagai callable jika "predict" tidak tersedia

    # Ambil bounding box pertama (asumsi hanya ada satu wajah)
    detections = results[0]  # Ambil hasil deteksi pertama
    if len(detections) > 0:
        x1, y1, x2, y2 = detections[0][:4]  # Ambil bounding box pertama
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        face_crop = image[y1:y2, x1:x2]
        return face_crop

    print(f"No faces detected in image: {image_path}")
    return None

# Load FaceNet model
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

# Fungsi untuk mengekstrak embedding dari gambar wajah
def extract_embedding(face_image):
    img = Image.fromarray(face_image)
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
            cropped_face = crop_face(image_path)
            if cropped_face is not None:  # Pastikan wajah berhasil dipotong
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
