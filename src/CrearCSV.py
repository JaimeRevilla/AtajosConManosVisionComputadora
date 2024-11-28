import os
import pandas as pd


project_dir = os.path.abspath(".")
dataset_dir = "dataset_img"
data = []

if not os.path.exists(dataset_dir):
    print(f"Error: El directorio '{dataset_dir}' no existe.")
else:
    for category in os.listdir(dataset_dir):
        category_path = os.path.join(dataset_dir, category)
        if os.path.isdir(category_path):
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                if os.path.isfile(img_path):
                    relative_path = os.path.relpath(img_path, project_dir)
                    data.append({"filepath": relative_path, "label": category})

    df = pd.DataFrame(data)
    df.to_csv("dataset_imagenes_relativas.csv", index=False)
    print("Archivo CSV generado con rutas relativas: dataset_imagenes_relativas.csv")
