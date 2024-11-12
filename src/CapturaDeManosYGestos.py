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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mostrar la imagen en pantalla
    cv2.imshow("Detección de Gesto", frame)

    # Presionar 's' para detectar gesto
    if cv2.waitKey(1) & 0xFF == ord('s'):
        print("Detectando gesto...")
        detectar_y_ejecutar()

    # Presionar 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
