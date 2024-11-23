import cv2
import numpy as np

def detect_hand():
    # Iniciar captura de video
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("No se pudo acceder a la cámara.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar la imagen.")
            break

        # Dimensiones de la imagen
        h, w, _ = frame.shape
        
        # Definir área de interés (ROI)
        roi = frame[h // 4:3 * h // 4, w // 4:3 * w // 4]
        cv2.rectangle(frame, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 255, 0), 2)

        # Convertir a escala de grises
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Aplicar desenfoque para reducir ruido
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Aplicar umbralización binaria
        _, binary = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)  # Inverso para resaltar la mano

        # Detectar contornos
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Seleccionar el contorno más grande
            max_contour = max(contours, key=cv2.contourArea)

            # Dibujar el contorno en la ROI
            cv2.drawContours(roi, [max_contour], -1, (0, 255, 0), 2)

            # Calcular el hull convexo
            hull = cv2.convexHull(max_contour)
            cv2.drawContours(roi, [hull], -1, (255, 0, 0), 2)

            # Detectar defectos de convexidad
            hull_indices = cv2.convexHull(max_contour, returnPoints=False)
            if len(hull_indices) > 3:  # Evitar errores si hay pocos puntos
                defects = cv2.convexityDefects(max_contour, hull_indices)

                if defects is not None:
                    count_defects = 0

                    for i in range(defects.shape[0]):
                        s, e, f, depth = defects[i, 0]
                        start = tuple(max_contour[s][0])
                        end = tuple(max_contour[e][0])
                        far = tuple(max_contour[f][0])

                        # Dibujar líneas y puntos de defectos convexos
                        cv2.line(roi, start, end, (0, 255, 0), 2)
                        cv2.circle(roi, far, 5, (0, 0, 255), -1)

                        # Contar defectos profundos (dedos levantados)
                        if depth > 1000:  # Ajustar este valor según el tamaño de la mano
                            count_defects += 1

                    # Mostrar número de dedos detectados
                    cv2.putText(roi, f"Dedos: {count_defects + 1}", (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Mostrar las imágenes
        cv2.imshow('Frame', frame)
        cv2.imshow('ROI', roi)
        cv2.imshow('Binary', binary)

        # Salir al presionar ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Ejecutar la función principal
if __name__ == "__main__":
    detect_hand()
