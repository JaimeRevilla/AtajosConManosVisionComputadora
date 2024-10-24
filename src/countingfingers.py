import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture(0)
bg = None

# COLORES PARA VISUALIZACIÓN
color_start = (204,204,0)
color_end = (204,0,204)
color_far = (255,0,0)

color_fingers = (0,255,255)

while True:
    ret, frame = cap.read()
    if ret == False:
        print("Error al leer el frame de la cámara.")
        break

    # Redimensionar la imagen para que tenga un ancho de 640
    frame = imutils.resize(frame, width=640)
    frame = cv2.flip(frame, 1)
    frameAux = frame.copy()
    
    if bg is not None:
        print("Fondo detectado, aplicando diferencias.")
        # Determinar la región de interés
        ROI = frame[50:300, 380:600]
        cv2.rectangle(frame, (380-2, 50-2), (600+2, 300+2), color_fingers, 1)
        grayROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)

        # Región de interés del fondo de la imagen
        bgROI = bg[50:300, 380:600]

        # Determinar la imagen binaria (background vs foreground)
        dif = cv2.absdiff(grayROI, bgROI)
        _, th = cv2.threshold(dif, 30, 255, cv2.THRESH_BINARY)
        th = cv2.medianBlur(th, 7)
        
        # Encontrando los contornos de la imagen binaria
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

        for cnt in cnts:
            # Encontrar el centro del contorno
            M = cv2.moments(cnt)
            if M["m00"] == 0: M["m00"] = 1
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            cv2.circle(ROI, tuple([x, y]), 5, (0, 255, 0), -1)

            # Contorno encontrado a través de cv2.convexHull
            hull1 = cv2.convexHull(cnt)
            cv2.drawContours(ROI, [hull1], 0, (0, 255, 0), 2)
        
        # Mostrar la imagen binaria
        cv2.imshow('th', th)  # Asegúrate de que esta ventana se esté abriendo

    else:
        print("Fondo no detectado. Presiona 'i' para inicializar el fondo.")
    
    # Mostrar la imagen original con el contorno
    cv2.imshow('Frame', frame)

    k = cv2.waitKey(20)
    if k == ord('i'):
        bg = cv2.cvtColor(frameAux, cv2.COLOR_BGR2GRAY)
        print("Fondo inicializado")
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
