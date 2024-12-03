import cv2
import numpy as np
import imutils

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

# Estado para la funcionalidad de "Hola Mundo"
esperando_confirmacion = False
esperando_cancelacion = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer el frame de la cámara.")
        break

    frame = imutils.resize(frame, width=640)
    frame = cv2.flip(frame, 1)
    frameAux = frame.copy()

    if bg is not None:
        # Región de interés ampliada para ambas manos
        ROI = frame[50:300, 100:540]
        cv2.rectangle(frame, (100-2, 50-2), (540+2, 300+2), color_fingers, 1)
        grayROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        bgROI = bg[50:300, 100:540]

        # Procesamiento del ROI
        dif = cv2.absdiff(grayROI, bgROI)
        _, mask = cv2.threshold(dif, umbral_diferencia, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.medianBlur(mask, 7)

        # Encontrar contornos
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dedos_totales = 0

        for cnt in cnts:
            if cv2.contourArea(cnt) < 1000:  # Filtrar ruido
                continue

            # Dibujar el contorno
            cv2.drawContours(ROI, [cnt], 0, (255, 255, 0), 2)

            # Calcular el hull convexo y defectos
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
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        far = tuple(cnt[f][0])

                        # Validar que el punto es un dedo
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

                dedos_totales += dedos_levantados + 1

        # Actualizar historial de detecciones
        finger_history.append(dedos_totales)
        if len(finger_history) > frame_interval:
            finger_history.pop(0)

        # Comprobar estabilidad
        if finger_history.count(finger_history[0]) == len(finger_history):
            stable_fingers = finger_history[0]

        # Funcionalidad: Detectar 8 dedos, confirmar o cancelar
        if stable_fingers == 8 and not esperando_confirmacion:
            print("Se han detectado 8 dedos, procedemos?")
            esperando_confirmacion = True

        elif esperando_confirmacion:
            if stable_fingers == 1:
                print("Hola Mundo")
                esperando_confirmacion = False
                esperando_cancelacion = False
            elif stable_fingers == 2:
                print("Operación cancelada. Esperando nueva instrucción.")
                esperando_confirmacion = False
                esperando_cancelacion = True

        # Mostrar máscara binaria
        cv2.imshow('th', mask)

    else:
        print("Fondo no detectado. Presiona 'i' para inicializar el fondo.")

    # Mostrar el número estable de dedos detectados
    if stable_fingers is not None:
        cv2.putText(frame, f'Dedos: {stable_fingers}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color_fingers, 2)

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
