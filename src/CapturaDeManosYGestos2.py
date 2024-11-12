import cv2
import numpy as np
import mediapipe as mp
from skimage.feature import hog
from joblib import load
import Acciones
import time

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Cargar el modelo SVM entrenado
svm_model = load("gesture_recognition_svm.pkl")

# Iniciar la captura de video
cap = cv2.VideoCapture(0)
print("Presiona 'q' para salir")

# Función para detectar el gesto basado en el contorno de la mano
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

# Función para procesar y reconocer gestos en tiempo real
def detectar_y_ejecutar(frame):
    """
    Función para detectar el gesto y ejecutar el comando basado en SVM.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Procesamiento con MediaPipe para detectar contornos de mano
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            points = []
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                points.append([x, y])
            points = np.array(points, dtype=np.int32)

            hull = cv2.convexHull(points)
            gesto_detectado = detectar_gesto(hull)

            if gesto_detectado:
                print(f"Gesto detectado (MediaPipe): {gesto_detectado}")
                Acciones.ejecutar_comando(gesto_detectado)  # Ejecutar la acción correspondiente en Acciones.py

    # Procesamiento con SVM para reconocimiento de gestos basado en características HOG
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (64, 64))
    hog_features = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys')
    hog_features = np.array(hog_features).reshape(1, -1)  # Cambiar la forma para SVM

    # Hacer la predicción con el modelo SVM
    prediction = svm_model.predict(hog_features)
    confidence = svm_model.predict_proba(hog_features).max()
    label = prediction[0]
    
    # Mostrar el resultado en la ventana de video
    cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Ejecutar acción si se cumple una condición específica
    if label == "puño" and confidence > 0.8:
        Acciones.captura_pantalla()  # Ejemplo de acción para "puño" detectado por SVM
    elif label == "saludo" and confidence > 0.8:
        Acciones.abrir_aplicacion()  # Ejemplo de acción para "saludo" detectado por SVM

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Llamar a la función de detección y ejecución que combina MediaPipe y SVM
    detectar_y_ejecutar(frame)

    # Mostrar la ventana de video con los resultados
    cv2.imshow("Detección de Gestos con SVM y MediaPipe", frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
