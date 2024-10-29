import os
import pandas as pd

dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src/dataset_img")
data = []

if not os.path.exists(dataset_dir):
    print(f"Error: El directorio '{dataset_dir}' no existe.")
else:
    for category in os.listdir(dataset_dir):
        category_path = os.path.join(dataset_dir, category)
        if os.path.isdir(category_path):
            for img_name in os.listdir(category_path):
                img_path = os.path.join("src/dataset_img", category, img_name)
                data.append({"filepath": img_path, "label": category})

    df = pd.DataFrame(data)

    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset_index.csv")
    df.to_csv(csv_path, index=False)
    print(f"Archivo CSV creado exitosamente en {csv_path}")
