import os
import pickle
import numpy as np
from PIL import Image
import torch
import cv2
from sklearn.metrics import classification_report
from facenet_pytorch import InceptionResnetV1
from yolov8 import YOLOv8

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

# Fungsi untuk menghitung jarak kosinus
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

# Fungsi untuk melakukan testing dataset terhadap embeddings
def test_embeddings(dataset_folder, embeddings_file):
    # Load embeddings dari file
    with open(embeddings_file, 'rb') as f:
        embeddings_db = pickle.load(f)

    y_true = []
    y_pred = []

    # Iterasi melalui dataset
    for person_name in os.listdir(dataset_folder):
        person_folder = os.path.join(dataset_folder, person_name)
        if os.path.isdir(person_folder):
            print(f"Testing untuk: {person_name}")
            for image_file in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_file)
                if os.path.isfile(image_path):
                    print(f"  Menguji gambar: {image_file}")

                    # Potong wajah dari gambar
                    cropped_face = crop_face(image_path)
                    if cropped_face is None:
                        print("    Wajah tidak terdeteksi, melewati gambar ini.")
                        continue

                    # Ekstraksi embedding gambar
                    test_embedding = extract_embedding(cropped_face)

                    # Perhitungan jarak kosinus dengan semua embeddings
                    similarities = {
                        name: cosine_similarity(test_embedding, emb)
                        for name, emb in embeddings_db.items()
                    }

                    # Temukan nama dengan kemiripan tertinggi
                    best_match = max(similarities, key=similarities.get)
                    confidence = similarities[best_match]

                    print(f"    Prediksi: {best_match}, Confidence: {confidence:.4f}")

                    # Append hasil ke y_true dan y_pred
                    y_true.append(person_name)
                    y_pred.append(best_match)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

# Jalankan testing pada dataset './DatasetBanyak' dengan embeddings 'embeddings.pkl'
test_embeddings('C:/Users/ACER/Deta/DatasetBanyak', 'embeddings.pkl')
