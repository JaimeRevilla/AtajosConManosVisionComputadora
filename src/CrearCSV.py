import os
import pandas as pd

dataset_dir = "dataset_img"
data = []

if not os.path.exists(dataset_dir):
    print(f"Error: El directorio '{dataset_dir}' no existe.")
else:
    for category in os.listdir(dataset_dir):
        category_path = os.path.join(dataset_dir, category)
        if os.path.isdir(category_path):
            for img_name in os.listdir(category_path):
                img_path = os.path.abspath(os.path.join(category_path, img_name))
                if os.path.isfile(img_path):
                    data.append({"filepath": img_path, "label": category})

    df = pd.DataFrame(data)
    df.to_csv("dataset_imagenes_absolutas.csv", index=False)
    print("Archivo CSV generado con rutas absolutas: dataset.csv")
