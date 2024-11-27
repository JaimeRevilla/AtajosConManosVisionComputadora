import cv2
import numpy as np
import imutils
from skimage.feature import hog
import joblib

# Cargar el modelo SVM entrenado
modelo_svm = joblib.load('gesture_recognition_svm.pkl')

# Clases de gestos (ajusta según el modelo)
clases = ['puño', 'mano_abierta', 'dedo_indice', 'OK', 'otros']

cap = cv2.VideoCapture(0)
bg = None

# COLORES PARA VISUALIZACIÓN
color_start = (204, 204, 0)
color_end = (204, 0, 204)
color_far = (255, 0, 0)
color_fingers = (0, 255, 255)

# Parámetros de segmentación
umbral_diferencia = 40
kernel = np.ones((5, 5), np.uint8)

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

        # Encontrar contornos (opcional para depuración)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            cv2.drawContours(ROI, [cnt], 0, (255, 255, 0), 2)

        # Preprocesar la máscara para el modelo
        mask_resized = cv2.resize(mask, (64, 64))  # Redimensionar a 64x64
        hog_features = hog(mask_resized, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), block_norm='L2-Hys')

        # Hacer predicción
        prediccion = modelo_svm.predict([hog_features])
        clase_gesto = clases[int(prediccion[0])]

        # Mostrar resultado en la pantalla
        cv2.putText(frame, f'Gesto: {clase_gesto}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Mostrar la máscara binaria
        cv2.imshow('th', mask)

    else:
        print("Fondo no detectado. Presiona 'i' para inicializar el fondo.")

    # Mostrar la imagen original con el gesto detectado
    cv2.imshow('Frame', frame)

    k = cv2.waitKey(20)
    if k == ord('i'):
        bg = cv2.cvtColor(frameAux, cv2.COLOR_BGR2GRAY)
        print("Fondo inicializado")
    if k == 27:  # Presionar 'ESC' para salir
        break

cap.release()
cv2.destroyAllWindows()
