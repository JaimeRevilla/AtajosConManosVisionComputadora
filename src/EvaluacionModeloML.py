import os
import cv2
import pandas as pd
from skimage.feature import hog
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Ruta al modelo guardado
MODEL_PATH = "gesture_recognition_svm.pkl"
DATASET_PATH = "dataset_imagenes.csv"

# Comprobar si el modelo existe
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo en la ruta: {MODEL_PATH}")

# Cargar el modelo
svm_model = joblib.load(MODEL_PATH)
print("Modelo cargado con éxito.")

# Cargar el dataset
df = pd.read_csv(DATASET_PATH)

# Comprobar que todas las rutas de las imágenes sean válidas
missing_files = []
for index, row in df.iterrows():
    if not os.path.exists(row['filepath']):
        missing_files.append(row['filepath'])

if missing_files:
    print(f"Imágenes faltantes o rutas inválidas: {missing_files}")
    raise ValueError("Hay imágenes faltantes o rutas inválidas en el dataset.")
else:
    print("Todas las rutas son válidas.")

# Preparar características y etiquetas
features = []
true_labels = []

for index, row in df.iterrows():
    img_path = row['filepath']
    label = row['label']
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error al cargar la imagen: {img_path}")
        continue
    
    # Redimensionar imagen
    img_resized = cv2.resize(img, (64, 64))
    
    # Extraer características HOG
    hog_features = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys')
    
    features.append(hog_features)
    true_labels.append(label)

# Convertir a matrices de numpy
X_test = np.array(features)
y_test = np.array(true_labels)

# Realizar predicciones
y_pred = svm_model.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy en datos de evaluación: {accuracy}")
print("Informe de clasificación:\n", classification_report(y_test, y_pred))

# Guardar las predicciones en un archivo CSV
predictions_df = pd.DataFrame({
    'filepath': df['filepath'],
    'true_label': y_test,
    'predicted_label': y_pred
})
predictions_df.to_csv("predicciones.csv", index=False)
print("Predicciones guardadas en 'predicciones.csv'.")

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title("Matriz de Confusión")
plt.xlabel("Etiqueta Predicha")
plt.ylabel("Etiqueta Verdadera")
plt.show()

# Curva ROC y AUC (para clasificación binaria o multiclase)
if len(np.unique(y_test)) == 2:  # Clasificación binaria
    y_test_binary = (y_test == np.unique(y_test)[1]).astype(int)
    y_pred_probs = svm_model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("Curva ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()

# Mostrar ejemplos de predicciones correctas e incorrectas
correct_idx = np.where(y_pred == y_test)[0]
incorrect_idx = np.where(y_pred != y_test)[0]

print("Ejemplos de predicciones correctas:")
for i in correct_idx[:5]:  # Mostrar los primeros 5 ejemplos correctos
    print(f"Imagen: {df.iloc[i]['filepath']}, Verdadero: {y_test[i]}, Predicho: {y_pred[i]}")

print("\nEjemplos de predicciones incorrectas:")
for i in incorrect_idx[:5]:  # Mostrar los primeros 5 ejemplos incorrectos
    print(f"Imagen: {df.iloc[i]['filepath']}, Verdadero: {y_test[i]}, Predicho: {y_pred[i]}")
