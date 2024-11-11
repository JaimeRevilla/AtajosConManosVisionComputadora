import cv2
import mediapipe as mp
import numpy as np
import Acciones
import time

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)

# Captura de video
cap = cv2.VideoCapture(0)

# Variables de tiempo
tiempo_espera = 3  # Tiempo para esperar y detectar gesto (en segundos)
tiempo_descanso = 10  # Tiempo de descanso después de ejecutar un comando (en segundos)
ultimo_tiempo = 0
en_descanso = False

def detectar_gesto(contorno):
    """
    Detecta el gesto basándose en el contorno de la mano.
    """
    area_contorno = cv2.contourArea(contorno)
    hull = cv2.convexHull(contorno)
    area_hull = cv2.contourArea(hull)

    solidity = area_contorno / area_hull

    if solidity > 0.9:
        return "puño"
    elif solidity > 0.5:
        return "okey"
    else:
        return "mano_abierta"

def detectar_y_ejecutar():
    """
    Función para detectar el gesto y ejecutar el comando.
    """
    global ultimo_tiempo, en_descanso

    if time.time() - ultimo_tiempo >= tiempo_espera:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                points = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    points.append([x, y])
                points = np.array(points, dtype=np.int32)

                # Crear contorno y detectar gesto
                hull = cv2.convexHull(points)
                gesto_detectado = detectar_gesto(hull)

                if gesto_detectado:
                    print(f"Gesto detectado: {gesto_detectado}")
                    Acciones.ejecutar_comando(gesto_detectado)
                    ultimo_tiempo = time.time()
                    en_descanso = True
                    return

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Si estamos en descanso, esperamos hasta que pase el tiempo de descanso
    if en_descanso:
        if time.time() - ultimo_tiempo >= tiempo_descanso:
            en_descanso = False
        continue

    # Detectar gesto y ejecutar acción
    detectar_y_ejecutar()

    # Mostrar la imagen en pantalla
    cv2.imshow("Detección de Gesto", frame)

    # Presionar 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
