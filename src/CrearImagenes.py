import cv2
import os
import time

output_dir = os.path.join(os.path.dirname(__file__), "dataset_img")
category_name = "categoria"  # Cambiar el nombre de la carpeta para cada categoría de gesto
capture_interval = 2

save_path = os.path.join(output_dir, category_name)
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)

img_counter = 0

print(f"Capturando una imagen cada {capture_interval} segundos. Presiona 'q' para salir.")

last_capture_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar la imagen.")
        break

    cv2.imshow("Camera", frame)

    current_time = time.time()
    if current_time - last_capture_time >= capture_interval:
        img_name = f"{category_name}_{img_counter}.png"
        cv2.imwrite(os.path.join(save_path, img_name), frame)
        print(f"Imagen {img_name} guardada en {save_path}")
        img_counter += 1
        last_capture_time = current_time

    if cv2.waitKey(1) == ord("q"):
        print("Saliendo del programa.")
        break

cap.release()
cv2.destroyAllWindows()
