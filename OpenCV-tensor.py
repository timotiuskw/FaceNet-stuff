from PIL import Image
import numpy as np
from numpy import asarray
from numpy import expand_dims

import pickle
import cv2

from keras_facenet import FaceNet

HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
MyFaceNet = FaceNet()

myfile = open("data.pkl", "rb")
database = pickle.load(myfile)
myfile.close()

cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    # Read frame from camera
    ret, gbr1 = cap.read()
    if not ret:
        break

    wajah = HaarCascade.detectMultiScale(gbr1, 1.1, 4)
    
    frame_count += 1

    # Store identities and distances
    identities = []
    min_confidence = 0.6  # Set a threshold for confidence

    for (x1, y1, width, height) in wajah:
        if frame_count % 1 == 0:
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height

            # Extract face
            face = gbr1[y1:y2, x1:x2]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = Image.fromarray(face)
            face = face.resize((160, 160))
            face = asarray(face)
            face = expand_dims(face, axis=0)

            # Get the embedding for the face
            signature = MyFaceNet.embeddings(face)

            # Find the closest match in the database
            min_dist = float('inf')
            identity = 'Unknown'
            for key, value in database.items():
                dist = np.linalg.norm(value - signature)
                if dist < min_dist:
                    min_dist = dist
                    identity = key
            
        # Check if the distance is less than the minimum confidence
        if min_dist < min_confidence:
            identities.append((identity, (x1, y1, x2, y2)))  # Store identity and coordinates

    # Draw rectangles and labels for all detected faces
    for identity, (x1, y1, x2, y2) in identities:
        cv2.rectangle(gbr1, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(gbr1, identity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # Display the result
    cv2.imshow('res', gbr1)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:  # Press 'Esc' to exit
        break

cv2.destroyAllWindows()
cap.release()