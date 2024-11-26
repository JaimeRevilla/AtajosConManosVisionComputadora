import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture(0)
bg = None

# COLORES PARA VISUALIZACIÓN
color_start = (204, 204, 0)
color_end = (204, 0, 204)
color_far = (255, 0, 0)

color_fingers = (0, 255, 255)

# Ajuste de parámetros
umbral_diferencia = 40  # Ajusta este valor según la iluminación
distancia_minima_dedos = 20  # Distancia mínima entre puntos de puntas de dedos (reducido)
altura_minima = 20  # Altura mínima desde el centro de la palma para considerar un dedo (reducido)
angulo_maximo = np.pi / 0.6  # Ángulo máximo permitido (aproximadamente 100°)
kernel = np.ones((5, 5), np.uint8)  # Kernel para operaciones morfológicas

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

        # Encontrar contornos
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

        for cnt in cnts:
            # Dibujar el contorno más grande
            cv2.drawContours(ROI, [cnt], 0, (255, 255, 0), 2)

            # Calcular el hull convexo del contorno
            hull = cv2.convexHull(cnt, returnPoints=True)

            if hull is not None and len(hull) > 3:
                # Calcular centro de la palma
                M = cv2.moments(cnt)
                if M["m00"] == 0: M["m00"] = 1
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(ROI, (cx, cy), 5, (0, 255, 0), -1)

                # Filtrar puntos convexos que sean puntas de dedos
                dedos_levantados = 0
                puntos_filtrados = []
                for i, punto in enumerate(hull):
                    x, y = punto[0]

                    # Validar que el punto esté por encima de la palma y sea suficientemente alto
                    if y < cy and (cy - y) > altura_minima:
                        # Validar ángulo con respecto a puntos vecinos
                        if i > 0 and i < len(hull) - 1:  # Evitar índices fuera de rango
                            anterior = hull[i - 1][0]
                            siguiente = hull[i + 1][0]
                            vector1 = np.array([x - anterior[0], y - anterior[1]])
                            vector2 = np.array([x - siguiente[0], y - siguiente[1]])
                            angulo = np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))

                            # Mostrar ángulo para depuración
                            print(f"Punto ({x},{y}) Ángulo: {np.degrees(angulo):.2f}°")

                            # Ángulo debe ser menor al máximo permitido
                            if angulo < angulo_maximo:
                                # Filtrar puntos cercanos
                                if all(np.linalg.norm(np.array([x, y]) - np.array(p)) > distancia_minima_dedos for p in puntos_filtrados):
                                    puntos_filtrados.append((x, y))
                                    dedos_levantados += 1
                                    cv2.circle(ROI, (x, y), 10, color_far, -1)

                # Mostrar el número de dedos levantados
                cv2.putText(frame, f'Dedos: {dedos_levantados}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color_fingers, 2)

        # Mostrar la máscara binaria
        cv2.imshow('th', mask)
    else:
        print("Fondo no detectado. Presiona 'i' para inicializar el fondo.")

    # Mostrar la imagen original con el contorno
    cv2.imshow('Frame', frame)

    k = cv2.waitKey(20)
    if k == ord('i'):
        bg = cv2.cvtColor(frameAux, cv2.COLOR_BGR2GRAY)
        print("Fondo inicializado")
    if k == 27:  # Presionar 'ESC' para salir
        break

cap.release()
cv2.destroyAllWindows()
