import os
from os import listdir
from PIL import Image as Img
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from keras.models import load_model
import numpy as np
import tensorflow as tf

import pickle
import cv2

from keras_facenet import FaceNet

HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))

MyFaceNet = FaceNet()

import os
import cv2
from PIL import Image as Img
import numpy as np

folder = r'D:\Tugas Kuliah\Bengkel Koding\Proyek Virtual Assistance\Dataset'
database = {}

# Loop melalui setiap label (subfolder) dalam folder utama
for label in os.listdir(folder):
    label_path = os.path.join(folder, label)  # Path ke subfolder
    if os.path.isdir(label_path):  # Pastikan ini adalah direktori
        # Loop melalui setiap file dalam subfolder
        for filename in os.listdir(label_path):
            # Cek apakah file adalah gambar
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(label_path, filename)  # Menggunakan os.path.join untuk menghindari masalah path
                gbr1 = cv2.imread(path)

                # Deteksi wajah
                wajah = HaarCascade.detectMultiScale(gbr1, scaleFactor=1.1, minNeighbors=4)

                # Mengambil koordinat wajah
                if len(wajah) > 0:
                    x1, y1, width, height = wajah[0]
                else:
                    x1, y1, width, height = 1, 1, 10, 10

                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height

                # Konversi gambar dari OpenCV ke PIL
                gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
                gbr = Img.fromarray(gbr)
                gbr_array = np.asarray(gbr)

                # Ekstrak wajah
                face = gbr_array[y1:y2, x1:x2]

                # Pastikan face tidak kosong
                if face.size == 0:
                    continue  # Skip if no face detected

                # Resize wajah dan konversi kembali ke array
                face = Img.fromarray(face)
                face = face.resize((160, 160))
                face = np.asarray(face)

                # Tambahkan dimensi batch
                face = np.expand_dims(face, axis=0)

                # Menghitung embedding
                signature = MyFaceNet.embeddings(face)

                # Mengelola penamaan label
                original_label = label
                count = 1
                new_label = original_label

                # Cek jika label sudah ada, tambahkan angka di depannya
                while new_label in database:
                    new_label = f"{original_label}_{count}"
                    count += 1

                # Simpan signature ke database dengan label sebagai kunci
                database[new_label] = [signature]  # Create a new list for each label

myfile = open("data.pkl", "wb")
pickle.dump(database, myfile)
myfile.close()