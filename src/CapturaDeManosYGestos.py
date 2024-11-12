import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)

# Captura de video
cap = cv2.VideoCapture(0)

# Obtener tamaño de pantalla
screen_width, screen_height = pyautogui.size()

# Inicializar variables de estado
modo_mover_ratón = False
ultimo_tiempo_accion = time.time()
cooldown = 0.5  # Tiempo de espera entre acciones en segundos

def mover_ratón(frame):
    """
    Mueve el ratón usando la posición del dedo índice.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Usar el landmark 8 (punta del dedo índice)
            x = int(hand_landmarks.landmark[8].x * frame.shape[1])
            y = int(hand_landmarks.landmark[8].y * frame.shape[0])

            # Escalar las coordenadas a la pantalla
            screen_x = int(x * screen_width / frame.shape[1])
            screen_y = int(y * screen_height / frame.shape[0])

            # Mover el ratón
            pyautogui.moveTo(screen_x, screen_y, duration=0.05)
            print(f"Moviendo el ratón a: ({screen_x}, {screen_y})")

def hacer_click():
    """
    Realiza un clic izquierdo en la posición actual del ratón.
    """
    pyautogui.click()
    print("Clic realizado.")

def abrir_bloc_de_notas():
    """
    Abre el Bloc de notas.
    """
    pyautogui.hotkey('win', 'r')
    pyautogui.write('notepad')
    pyautogui.press('enter')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el video.")
        break

    # Mostrar la imagen en pantalla
    cv2.imshow("Detección de Mano", frame)

    # Capturar la tecla presionada
    key = cv2.waitKey(1) & 0xFF

    # Control del cooldown para evitar acciones demasiado rápidas
    tiempo_actual = time.time()

    # Presionar 's' para abrir el Bloc de notas
    if key == ord('s') and tiempo_actual - ultimo_tiempo_accion > cooldown:
        print("Tecla 's' presionada. Abriendo Bloc de notas...")
        abrir_bloc_de_notas()
        ultimo_tiempo_accion = tiempo_actual

    # Presionar 'm' para activar/desactivar el modo de mover el ratón
    if key == ord('m') and tiempo_actual - ultimo_tiempo_accion > cooldown:
        modo_mover_ratón = not modo_mover_ratón
        estado = "activado" if modo_mover_ratón else "desactivado"
        print(f"Modo mover ratón {estado}")
        ultimo_tiempo_accion = tiempo_actual

    # Mover el ratón si el modo está activado
    if modo_mover_ratón:
        mover_ratón(frame)

    # Presionar 'd' para hacer clic
    if key == ord('d') and tiempo_actual - ultimo_tiempo_accion > cooldown:
        print("Tecla 'd' presionada. Realizando clic...")
        hacer_click()
        ultimo_tiempo_accion = tiempo_actual

    # Presionar 'q' para salir
    if key == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
