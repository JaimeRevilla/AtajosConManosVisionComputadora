import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

landmarks_relevantes = [0, 4, 8, 12, 16, 20]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            puntos_externos = []
            
            for idx in landmarks_relevantes:
                x = int(hand_landmarks.landmark[idx].x * frame.shape[1])
                y = int(hand_landmarks.landmark[idx].y * frame.shape[0])
                puntos_externos.append([x, y])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            puntos_externos = np.array(puntos_externos, dtype=np.int32)
            cv2.fillPoly(mask, [puntos_externos], 255)

    blurred_mask = cv2.GaussianBlur(mask, (15, 15), 0)
    kernel = np.ones((10, 10), np.uint8)
    refined_mask = cv2.dilate(blurred_mask, kernel, iterations=1)
    refined_mask = cv2.erode(refined_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

    mask_display = cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2BGR)
    combined_display = np.hstack((frame, mask_display))
    cv2.imshow("Contorno Definido de la Mano", combined_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
