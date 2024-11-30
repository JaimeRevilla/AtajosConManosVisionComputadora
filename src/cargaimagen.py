import os
import cv2
import pandas as pd

# Configuración
input_csv = "dataset_img\dataset_imagenes_relativas.csv"  # Ruta al archivo CSV con las rutas de las imágenes y etiquetas
output_dir = "dataset_resized"  # Ruta donde se guardarán las imágenes redimensionadas
target_size = (512, 512)  # Tamaño al que queremos redimensionar las imágenes

# Crear el directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Leer el archivo CSV
df = pd.read_csv(input_csv)

# Recorrer cada fila del CSV
for index, row in df.iterrows():
    input_image_path = row['filepath']  # Columna con la ruta de la imagen
    label = row['label']  # Columna con la etiqueta/clase

    # Crear un subdirectorio por clase en el directorio de salida
    output_label_dir = os.path.join(output_dir, str(label))
    os.makedirs(output_label_dir, exist_ok=True)

    # Procesar la imagen
    try:
        img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error al cargar la imagen: {input_image_path}")
            continue

        # Redimensionar la imagen a 42x42
        img_resized = cv2.resize(img, target_size)

        # Guardar la imagen redimensionada
        output_image_path = os.path.join(output_label_dir, os.path.basename(input_image_path))
        cv2.imwrite(output_image_path, img_resized)

        print(f"Procesada: {output_image_path}")

    except Exception as e:
        print(f"Error al procesar la imagen {input_image_path}: {e}")

print("Todas las imágenes han sido redimensionadas.")
