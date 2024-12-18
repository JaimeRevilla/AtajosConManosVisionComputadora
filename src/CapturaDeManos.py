import cv2
import mediapipe as mp

def main():
    # Inicializamos MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Abrimos la cámara
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error al abrir la cámara")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("No se pudo capturar el frame")
            break

        # Convertimos la imagen al formato RGB requerido por MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesamos la imagen para detectar manos
        results = hands.process(rgb_frame)

        # Dibujamos las manos detectadas en el frame original
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Mostramos el frame con las manos detectadas
        cv2.imshow('Detección de Manos', frame)

        # Salir al presionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberamos los recursos
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()