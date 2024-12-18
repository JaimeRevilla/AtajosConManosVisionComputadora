import cv2
import numpy as np
import imutils
import pyautogui  # Para controlar el ordenador

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

# Parámetros para estabilización
frame_interval = 60
finger_history = []
stable_fingers = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer el frame de la cámara.")
        break

    frame = imutils.resize(frame, width=640)
    frame = cv2.flip(frame, 1)
    frameAux = frame.copy()

    if bg is not None:
        ROI = frame[50:300, 100:540]
        cv2.rectangle(frame, (100-2, 50-2), (540+2, 300+2), color_fingers, 1)
        grayROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        bgROI = bg[50:300, 100:540]

        # Procesamiento del ROI
        dif = cv2.absdiff(grayROI, bgROI)
        _, mask = cv2.threshold(dif, umbral_diferencia, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.medianBlur(mask, 7)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dedos_totales = 0
        thumb_direction = None

        for cnt in cnts:
            if cv2.contourArea(cnt) < 1000:
                continue

            cv2.drawContours(ROI, [cnt], 0, (255, 255, 0), 2)

            hull = cv2.convexHull(cnt, returnPoints=False)
            if hull is not None and len(hull) > 3:
                defects = cv2.convexityDefects(cnt, hull)

                M = cv2.moments(cnt)
                if M["m00"] == 0: M["m00"] = 1
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(ROI, (cx, cy), 5, (0, 255, 0), -1)

                dedos_levantados = 0
                thumb_point = None

                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        far = tuple(cnt[f][0])

                        dist_centro = np.linalg.norm(np.array([far[0], far[1]]) - np.array([cx, cy]))
                        if dist_centro > altura_minima:
                            a = np.linalg.norm(np.array(start) - np.array(far))
                            b = np.linalg.norm(np.array(end) - np.array(far))
                            c = np.linalg.norm(np.array(start) - np.array(end))
                            angulo = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))

                            if angulo < angulo_maximo and d > distancia_minima_dedos:
                                dedos_levantados += 1
                                cv2.circle(ROI, start, 5, color_start, -1)
                                cv2.circle(ROI, end, 5, color_end, -1)
                                cv2.circle(ROI, far, 5, color_far, -1)

                        # Determinar el punto más alejado del centro como el pulgar
                        dist_start_to_center = np.linalg.norm(np.array(start) - np.array([cx, cy]))
                        dist_end_to_center = np.linalg.norm(np.array(end) - np.array([cx, cy]))
                        
                        if dist_start_to_center > dist_end_to_center:
                            thumb_point = start
                        else:
                            thumb_point = end

                if dedos_levantados == 0 and thumb_point is not None:
                    # Determinar si el pulgar está arriba o abajo
                    if thumb_point[1] < cy:
                        thumb_direction = "arriba"
                    else:
                        thumb_direction = "abajo"

        # Mostrar dirección del pulgar si es detectado
        if thumb_direction is not None:
            cv2.putText(ROI, f'Thumb: {thumb_direction}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Mostrar máscara binaria
        cv2.imshow('th', mask)

    else:
        print("Fondo no detectado. Presiona 'i' para inicializar el fondo.")

    cv2.imshow('Frame', frame)

    # Manejo de teclas
    k = cv2.waitKey(20)
    if k == ord('i'):
        bg = cv2.cvtColor(frameAux, cv2.COLOR_BGR2GRAY)
        print("Fondo inicializado")
    if k == 27:  # Presionar 'ESC' para salir
        break

cap.release()
cv2.destroyAllWindows()
