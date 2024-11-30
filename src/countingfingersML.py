import cv2
import numpy as np
import imutils
from skimage.feature import hog
import joblib  # Para cargar el modelo

# Cargar los modelos y transformaciones
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

# Inicializar la captura de video
cap = cv2.VideoCapture(1)
bg = None

# COLORES PARA VISUALIZACIÓN
color_start = (204, 204, 0)
color_end = (204, 0, 204)
color_far = (255, 0, 0)
color_fingers = (0, 255, 255)

# Ajuste de parámetros
umbral_diferencia = 40
distancia_minima_dedos = 20
altura_minima = 20
angulo_maximo = np.pi / 2
kernel = np.ones((5, 5), np.uint8)

# Variable para controlar la detección
realizar_deteccion = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer el frame de la cámara.")
        break

    # Redimensionar y voltear la imagen
    frame = imutils.resize(frame, width=640)
    frame = cv2.flip(frame, 1)
    frameAux = frame.copy()

    if bg is not None:
        # Región de interés (ROI)
        ROI = frame[50:300, 380:600]
        cv2.rectangle(frame, (380-2, 50-2), (600+2, 300+2), color_fingers, 1)
        grayROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)

        # Fondo en la misma región
        bgROI = bg[50:300, 380:600]

        # Calcular diferencia absoluta y aplicar umbral
        dif = cv2.absdiff(grayROI, bgROI)
        _, mask = cv2.threshold(dif, umbral_diferencia, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.medianBlur(mask, 7)

        # Mostrar la máscara en blanco y negro
        cv2.imshow("Mascara", mask)

        if realizar_deteccion:
            # Detectar dedos o gestos
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cnt = max(cnts, key=cv2.contourArea)

                # Dibujar el contorno más grande
                cv2.drawContours(ROI, [cnt], 0, (255, 255, 0), 2)

                # --- Detección sin modelo ---
                hull = cv2.convexHull(cnt, returnPoints=False)
                if hull is not None and len(hull) > 3:
                    defects = cv2.convexityDefects(cnt, hull)

                    dedos_levantados = 0
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(cnt[s][0])
                            end = tuple(cnt[e][0])
                            far = tuple(cnt[f][0])

                            # Calcular distancias y ángulos
                            a = np.linalg.norm(np.array(start) - np.array(far))
                            b = np.linalg.norm(np.array(end) - np.array(far))
                            c = np.linalg.norm(np.array(start) - np.array(end))

                            angulo = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
                            if angulo <= np.pi / 2:  # Ángulo máximo permitido
                                dedos_levantados += 1

                    print(f"Dedos levantados detectados sin modelo: {dedos_levantados + 1}")

                # --- Detección con modelo (basada en la máscara) ---
                resized_mask = cv2.resize(mask, (256, 256))  # Ajustar al tamaño esperado
                hog_features = hog(resized_mask, orientations=9, pixels_per_cell=(16, 16),
                                   cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
                try:
                    # Normalizar las características con el escalador
                    hog_features_scaled = scaler.transform([hog_features])
                    
                    # Aplicar PCA
                    hog_features_pca = pca.transform(hog_features_scaled)
                    
                    # Predicción con el modelo
                    prediction = model.predict(hog_features_pca)
                    print(f"Gesto detectado con modelo: {prediction[0]}")
                except Exception as e:
                    print(f"Error al predecir el gesto: {e}")

            # Desactivar la detección para evitar lecturas continuas
            realizar_deteccion = False

    # Mostrar el frame procesado
    cv2.imshow("Video", frame)

    # Controles de teclado
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Salir del programa
        break
    elif key == ord('b'):  # Configurar fondo
        bg = cv2.cvtColor(frameAux, cv2.COLOR_BGR2GRAY)
        print("Fondo configurado.")
    elif key == ord('d'):  # Activar detección
        realizar_deteccion = True
        print("Detección activada.")

cap.release()
cv2.destroyAllWindows()
