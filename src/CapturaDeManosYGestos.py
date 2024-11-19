import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

screen_width, screen_height = pyautogui.size()

modo_mover_ratón = False
ultimo_tiempo_accion = time.time()
cooldown = 0.5  

def mover_ratón(frame):
    """
    Mueve el ratón usando la posición del dedo índice.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x = int(hand_landmarks.landmark[8].x * frame.shape[1])
            y = int(hand_landmarks.landmark[8].y * frame.shape[0])

            screen_x = int(x * screen_width / frame.shape[1])
            screen_y = int(y * screen_height / frame.shape[0])

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

    cv2.imshow("Detección de Mano", frame)

    key = cv2.waitKey(1) & 0xFF

    tiempo_actual = time.time()

    if key == ord('s') and tiempo_actual - ultimo_tiempo_accion > cooldown:
        print("Tecla 's' presionada. Abriendo Bloc de notas...")
        abrir_bloc_de_notas()
        ultimo_tiempo_accion = tiempo_actual

    if key == ord('m') and tiempo_actual - ultimo_tiempo_accion > cooldown:
        modo_mover_ratón = not modo_mover_ratón
        estado = "activado" if modo_mover_ratón else "desactivado"
        print(f"Modo mover ratón {estado}")
        ultimo_tiempo_accion = tiempo_actual

    if modo_mover_ratón:
        mover_ratón(frame)

    if key == ord('d') and tiempo_actual - ultimo_tiempo_accion > cooldown:
        print("Tecla 'd' presionada. Realizando clic...")
        hacer_click()
        ultimo_tiempo_accion = tiempo_actual

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
