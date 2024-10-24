import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe para la detección de manos
mp_hands = mp.solutions.hands

# Capturar la cámara
cap = cv2.VideoCapture(0)
bg = None

# COLORES PARA VISUALIZACIÓN
color_contorno = (0,255,0)

# Configuraciones de MediaPipe
with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error al leer el frame de la cámara.")
            break

        # Convertir la imagen a RGB (MediaPipe trabaja en RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar la imagen para detectar manos
        result = hands.process(rgb_frame)
        
        # Si se detecta una mano
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Extraer los puntos clave para crear el contorno
                h, w, c = frame.shape
                hand_points = np.array([[int(landmark.x * w), int(landmark.y * h)] for landmark in hand_landmarks.landmark])
                
                # Crear una máscara binaria de la mano
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [hand_points], 255)
                
                # Encontrar contornos de la mano en la máscara binaria
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Dibujar el contorno de la mano en la imagen original
                if cnts:
                    cv2.drawContours(frame, cnts, -1, color_contorno, 2)
                
                # Mostrar la imagen binaria de la mano
                cv2.imshow('Mascara de Mano', mask)
        
        # Mostrar la imagen original con el contorno dibujado
        cv2.imshow('Detección de Contorno con MediaPipe', frame)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
