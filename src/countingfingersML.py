import cv2
import numpy as np
import imutils
import joblib  # Para cargar el modelo

# Cargar el modelo entrenado
model = joblib.load("gesture_recognition_svm.pkl")

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
        cv2.rectangle(frame, (378, 48), (602, 302), color_fingers, 1)
        grayROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)

        # Fondo en la misma región
        bgROI = bg[50:300, 380:600]

        # Calcular diferencia absoluta y aplicar umbral
        dif = cv2.absdiff(grayROI, bgROI)
        _, mask = cv2.threshold(dif, umbral_diferencia, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.medianBlur(mask, 7)

        # Encontrar contornos
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            cv2.drawContours(ROI, [cnt], 0, (255, 255, 0), 2)

            # Calcular el hull convexo
            hull = cv2.convexHull(cnt, returnPoints=False)
            if hull is not None and len(hull) > 3:
                defects = cv2.convexityDefects(cnt, hull)

                # Calcular centro de la palma
                M = cv2.moments(cnt)
                if M["m00"] == 0: M["m00"] = 1
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(ROI, (cx, cy), 5, (0, 255, 0), -1)

                dedos_levantados = 0
                features = []  # Aquí guardaremos las características
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        far = tuple(cnt[f][0])

                        # Distancias desde el defecto al centro
                        dist_centro = np.linalg.norm(np.array([far[0], far[1]]) - np.array([cx, cy]))

                        # Validar si es un dedo
                        if d > distancia_minima_dedos:
                            dedos_levantados += 1

                    # Añadir características básicas para el modelo
                    features = [len(defects), dedos_levantados]

                # Realizar predicción si tenemos características
                if features:
                    gesture = model.predict([features])[0]
                    cv2.putText(frame, f"Gesto: {gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar la imagen procesada
    cv2.imshow("Frame", frame)

    # Tecla para capturar fondo o salir
    key = cv2.waitKey(1) & 0xFF
    if key == ord("b"):  # Capturar el fondo
        bg = cv2.cvtColor(frameAux, cv2.COLOR_BGR2GRAY)
        print("Fondo capturado.")
    elif key == ord("q"):  # Salir del programa
        break

cap.release()
cv2.destroyAllWindows()
