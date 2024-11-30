import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Directorio del dataset redimensionado
dataset_dir = "dataset_resized"  # Cambia esta ruta al directorio redimensionado
labels = ["1_dedos", "2_dedos", "3_dedos", "4_dedos", "5_dedos", "ok", "perfect"]

# Función para cargar imágenes y calcular HOG
def load_images_and_labels(dataset_dir, labels):
    features = []
    labels_list = []

    for label in labels:
        label_dir = os.path.join(dataset_dir, label)
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Error al cargar la imagen: {img_path}")
                continue

            # Redimensionar la imagen a 256x256 (ajusta este tamaño si es necesario)
            img_resized = cv2.resize(img, (256, 256))

            # Calcular características HOG
            hog_features = hog(img_resized, orientations=9, pixels_per_cell=(16, 16),
                               cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
            features.append(hog_features)
            labels_list.append(label)
    
    return np.array(features), np.array(labels_list)

# Cargar datos
print("Cargando imágenes y calculando características HOG...")
X, y = load_images_and_labels(dataset_dir, labels)

# Normalizar las características
print("Normalizando las características...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reducir dimensiones con PCA
print("Aplicando PCA para reducción de dimensiones...")
pca = PCA(n_components=150)  # Cambia el número de componentes según lo que necesites
X_pca = pca.fit_transform(X)

# Dividir en conjunto de entrenamiento y prueba
print("Dividiendo datos en conjunto de entrenamiento y prueba...")
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Configuración de hiperparámetros para SVM
param_grid = {
    'C': [0.1, 1, 10, 100],       # Parámetro de regularización
    'kernel': ['linear', 'rbf'],  # Kernel lineal y RBF
    'gamma': ['scale', 'auto']    # Coeficiente para kernel RBF
}

# Ajuste del modelo SVM con GridSearchCV
print("Entrenando el modelo SVM con búsqueda de hiperparámetros (GridSearchCV)...")
svm = SVC(probability=True)
grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Mejor modelo encontrado por GridSearchCV
best_model = grid_search.best_estimator_
print(f"Mejores hiperparámetros encontrados: {grid_search.best_params_}")

# Evaluar el modelo
print("Evaluando el modelo en el conjunto de prueba...")
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión en el conjunto de prueba: {accuracy * 100:.2f}%")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Guardar el modelo, PCA y escalador
print("Guardando el modelo SVM, PCA y escalador...")
joblib.dump(best_model, "gesture_recognition_svm_pca.pkl")
joblib.dump(pca, "pca_transform.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Modelo guardado como 'gesture_recognition_svm_pca.pkl'")
print("PCA guardado como 'pca_transform.pkl'")
print("Escalador guardado como 'scaler.pkl'")
