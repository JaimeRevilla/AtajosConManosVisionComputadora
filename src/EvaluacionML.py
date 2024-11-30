import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Cargar el modelo, PCA y escalador
model_path = "Entrenos/gesture_recognition_svm_pca.pkl"
pca_path = "Entrenos/pca_transform.pkl"
scaler_path = "Entrenos/scaler.pkl"

try:
    model = joblib.load(model_path)
    print(f"Modelo SVM cargado correctamente desde: {model_path}")
    pca = joblib.load(pca_path)
    print(f"PCA cargado correctamente desde: {pca_path}")
    scaler = joblib.load(scaler_path)
    print(f"Escalador cargado correctamente desde: {scaler_path}")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Tamaño esperado de las imágenes
img_size = 256  # Ajusta este tamaño al usado en el entrenamiento

# Ruta del archivo CSV con los datos de prueba
dataset_csv = "dataset_resized/dataset_imagenes_relativas_2.csv"

# Verificar que el archivo CSV existe
if not os.path.exists(dataset_csv):
    print(f"Error: No se encontró el archivo de datos en {dataset_csv}")
    exit()

# Cargar el CSV con las rutas y etiquetas
df = pd.read_csv(dataset_csv)

# Verificar que las imágenes existen
missing_files = []
for index, row in df.iterrows():
    if not os.path.exists(row['filepath']):
        missing_files.append(row['filepath'])

if missing_files:
    print(f"Error: Las siguientes imágenes no existen: {missing_files}")
    exit()

# Preparar datos de prueba
X_test = []
y_test = []

print("Procesando imágenes para evaluación...")
for index, row in df.iterrows():
    img_path = row['filepath']
    label = row['label']
    
    # Cargar la imagen en escala de grises
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error al cargar la imagen: {img_path}")
        continue
    
    # Redimensionar la imagen al tamaño esperado
    img_resized = cv2.resize(img, (img_size, img_size))
    
    # Calcular HOG
    hog_features = hog(img_resized, orientations=9, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
    
    X_test.append(hog_features)
    y_test.append(label)

# Convertir a arrays de NumPy
X_test = np.array(X_test)
y_test = np.array(y_test)

# Normalizar las características con el escalador
X_test_scaled = scaler.transform(X_test)

# Aplicar PCA a las características
X_test_pca = pca.transform(X_test_scaled)

# Realizar predicciones
print("Realizando predicciones en los datos de prueba...")
y_pred = model.predict(X_test_pca)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo en los datos de prueba: {accuracy * 100:.2f}%")

# Reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# Prueba con imágenes individuales
while True:
    img_path = input("\nIngresa la ruta de una imagen para probar (o escribe 'salir' para terminar): ")
    if img_path.lower() == 'salir':
        break
    
    if not os.path.exists(img_path):
        print(f"Error: La imagen {img_path} no existe.")
        continue
    
    # Cargar y preprocesar la imagen
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error al cargar la imagen: {img_path}")
        continue
    
    img_resized = cv2.resize(img, (img_size, img_size))
    
    # Calcular HOG
    hog_features = hog(img_resized, orientations=9, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
    
    # Normalizar y aplicar PCA
    hog_features_scaled = scaler.transform([hog_features])
    hog_features_pca = pca.transform(hog_features_scaled)
    
    # Predicción
    prediction = model.predict(hog_features_pca)
    print(f"El modelo predice: {prediction[0]}")
