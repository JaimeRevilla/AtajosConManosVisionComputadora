import cv2
import numpy as np
import imutils
import joblib
from skimage.feature import hog

# Cargar el modelo entrenado
svm_model = joblib.load("gesture_recognition_svm.pkl")

# Configuración para HOG (debe coincidir con el entrenamiento)
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys"
}

# Captura de video
cap = cv2.VideoCapture(0)
bg = None  # Fondo inicial para sustracción

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer el frame de la cámara.")
        break

    frame = imutils.resize(frame, width=640)
    frame = cv2.flip(frame, 1)
    frameAux = frame.copy()

    if bg is not None:
        ROI = frame[50:300, 380:600]
        cv2.rectangle(frame, (380-2, 50-2), (600+2, 300+2), (0, 255, 255), 1)
        grayROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        bgROI = bg[50:300, 380:600]

        # Diferencia con el fondo
        dif = cv2.absdiff(grayROI, bgROI)
        _, mask = cv2.threshold(dif, 40, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.medianBlur(mask, 7)

        # Encontrar contornos
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            cv2.drawContours(ROI, [cnt], 0, (255, 255, 0), 2)

            # Extraer ROI del contorno más grande
            x, y, w, h = cv2.boundingRect(cnt)
            handROI = grayROI[y:y+h, x:x+w]

            # Preprocesar ROI para el modelo
            handROI_resized = cv2.resize(handROI, (64, 64))
            hog_features = hog(handROI_resized, **HOG_PARAMS).reshape(1, -1)

            # Clasificar gesto
            prediction = svm_model.predict(hog_features)[0]
            cv2.putText(frame, f"Gesto: {prediction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    # Captura del fondo (presionando 'b')
    if cv2.waitKey(1) & 0xFF == ord('b'):
        bg = cv2.cvtColor(frameAux, cv2.COLOR_BGR2GRAY)
        print("Fondo capturado")

    # Salir (presionando 'q')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
