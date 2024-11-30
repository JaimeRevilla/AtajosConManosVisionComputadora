import os
import cv2
import numpy as np
from albumentations import (
    Compose, Rotate, HorizontalFlip, RandomBrightnessContrast, GaussianBlur, CLAHE, GridDistortion, ElasticTransform
)
from albumentations.core.composition import OneOf

# Directorio del dataset original y el dataset aumentado
input_dir = "dataset_img"  # Directorio con las imágenes originales
output_dir = "dataset_img"  # Directorio donde se guardarán las imágenes aumentadas

# Carpetas específicas a procesar
folders_to_augment = ["ok", "perfect"]

# Crear el directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Configuración de aumentación
augment = Compose([
    Rotate(limit=20, p=0.5),                     # Rotación aleatoria de hasta 20 grados
    HorizontalFlip(p=0.5),                       # Volteo horizontal
    RandomBrightnessContrast(p=0.2),             # Cambios de brillo y contraste
    GaussianBlur(blur_limit=3, p=0.2),           # Aplicar desenfoque
    CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),  # Ecualización adaptativa del histograma
    OneOf([
        GridDistortion(p=0.3),                   # Distorsión en cuadrícula
        ElasticTransform(alpha=1, p=0.3)        # Transformación elástica
    ], p=0.3)
])

# Procesar solo las carpetas especificadas
for label in folders_to_augment:
    input_label_dir = os.path.join(input_dir, label)
    output_label_dir = os.path.join(output_dir, label)

    # Verificar si la carpeta existe
    if not os.path.exists(input_label_dir):
        print(f"Carpeta no encontrada: {input_label_dir}")
        continue

    # Crear subcarpeta en el directorio de salida si no existe
    os.makedirs(output_label_dir, exist_ok=True)

    # Procesar cada imagen en la carpeta
    for img_name in os.listdir(input_label_dir):
        img_path = os.path.join(input_label_dir, img_name)

        # Cargar la imagen
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error al cargar la imagen: {img_path}")
            continue

        # Generar imágenes aumentadas (2 aumentaciones por imagen)
        for i in range(2):  # Cambia este número para generar más aumentaciones por imagen
            augmented = augment(image=img)
            aug_img = augmented["image"]

            # Guardar la imagen aumentada
            base_name, ext = os.path.splitext(img_name)
            output_path = os.path.join(output_label_dir, f"{base_name}_aug_{i}{ext}")
            cv2.imwrite(output_path, aug_img)

            print(f"Imagen aumentada guardada: {output_path}")

print("Aumentación completada.")
