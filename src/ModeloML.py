import os
import cv2
import pandas as pd
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import joblib

df = pd.read_csv("dataset_index.csv")

features = []
labels = []

for index, row in df.iterrows():
    img_path = row['filepath']
    label = row['label']
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error al cargar la imagen: {img_path}")
        continue
    
    img_resized = cv2.resize(img, (64, 64))
    
    hog_features = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys')
    
    features.append(hog_features)
    labels.append(label)

X = features
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(svm_model, "gesture_recognition_svm.pkl")
print("Modelo guardado como 'gesture_recognition_svm.pkl'")