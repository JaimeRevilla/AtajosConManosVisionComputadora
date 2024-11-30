import os
import pandas as pd

<<<<<<< Updated upstream

project_dir = os.path.abspath(".")
dataset_dir = "dataset_img"
=======
# Configuración
dataset_dir = "dataset_resized"  # Cambia esta ruta si es necesario
output_csv = "dataset_imagenes_relativas_2.csv"  # Nombre del archivo CSV que se generará

# Inicializar la lista donde guardaremos las rutas y etiquetas
>>>>>>> Stashed changes
data = []

# Verificar si el directorio del dataset existe
if not os.path.exists(dataset_dir):
    print(f"Error: El directorio '{dataset_dir}' no existe.")
else:
    # Recorrer las subcarpetas (categorías/etiquetas)
    for category in os.listdir(dataset_dir):
        category_path = os.path.join(dataset_dir, category)
<<<<<<< Updated upstream
        if os.path.isdir(category_path):
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                if os.path.isfile(img_path):
                    relative_path = os.path.relpath(img_path, project_dir)
                    data.append({"filepath": relative_path, "label": category})
=======
>>>>>>> Stashed changes

        # Asegurarnos de que sea una carpeta
        if os.path.isdir(category_path):
            # Recorrer todas las imágenes dentro de la categoría
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)

                # Verificar si es un archivo válido (imagen)
                if os.path.isfile(img_path):
                    # Guardar la ruta relativa y la etiqueta
                    relative_path = os.path.relpath(img_path, os.getcwd())  # Ruta relativa al directorio actual
                    data.append({"filepath": relative_path, "label": category})

    # Crear un DataFrame con las rutas y etiquetas
    df = pd.DataFrame(data)
<<<<<<< Updated upstream
    df.to_csv("dataset_imagenes_relativas.csv", index=False)
    print("Archivo CSV generado con rutas relativas: dataset_imagenes_relativas.csv")
=======

    # Guardar el DataFrame como CSV
    try:
        df.to_csv(output_csv, index=False)
        print(f"Archivo CSV generado correctamente: {output_csv}")
    except Exception as e:
        print(f"Error al guardar el archivo CSV: {e}")
>>>>>>> Stashed changes
