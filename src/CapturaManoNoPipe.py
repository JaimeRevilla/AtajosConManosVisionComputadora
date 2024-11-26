import cv2
import numpy as np

def detectar_dedos(frame):
    # Convertir a escala de grises
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar un umbral binario inverso (fondo blanco -> 255, mano -> 0)
    _, binaria = cv2.threshold(gris, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Encontrar contornos
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return frame, 0  # No se detectó contorno
    
    # Seleccionar el contorno más grande (supuestamente la mano)
    contorno_max = max(contornos, key=cv2.contourArea)
    
    # Calcular el hull convexo del contorno
    hull = cv2.convexHull(contorno_max, returnPoints=False)
    if hull is None:
        return frame, 0  # Sin hull
    
    # Detectar defectos de convexidad
    defectos = cv2.convexityDefects(contorno_max, hull)
    if defectos is None:
        return frame, 0  # Sin defectos
    
    # Filtrar defectos para contar dedos levantados
    dedos_levantados = 0
    for i in range(defectos.shape[0]):
        inicio, fin, fondo, profundidad = defectos[i, 0]
        punto_inicio = tuple(contorno_max[inicio][0])
        punto_fin = tuple(contorno_max[fin][0])
        punto_fondo = tuple(contorno_max[fondo][0])
        
        # Calcular distancias y ángulos para filtrar puntos irrelevantes
        a = np.linalg.norm(np.array(punto_inicio) - np.array(punto_fin))
        b = np.linalg.norm(np.array(punto_inicio) - np.array(punto_fondo))
        c = np.linalg.norm(np.array(punto_fin) - np.array(punto_fondo))
        angulo = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
        
        # Dedos levantados: defectos profundos y ángulo menor a 90 grados
        if angulo < np.pi / 2 and profundidad > 20:  # Ajustar profundidad según escala
            dedos_levantados += 1
            cv2.circle(frame, punto_fondo, 5, (0, 255, 0), -1)

        # Dibujar líneas de los dedos detectados
        cv2.line(frame, punto_inicio, punto_fin, (255, 0, 0), 2)
        cv2.line(frame, punto_inicio, punto_fondo, (0, 255, 0), 1)
        cv2.line(frame, punto_fin, punto_fondo, (0, 255, 0), 1)

    # Visualizar el número de dedos detectados
    cv2.putText(frame, f'Dedos levantados: {dedos_levantados}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return frame, dedos_levantados


# Main
cap = cv2.VideoCapture(0)  # Cambiar a un archivo si usas video pregrabado

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame, dedos = detectar_dedos(frame)
    
    # Mostrar la imagen con detecciones
    cv2.imshow('Detección de dedos', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Presionar 'q' para salir
        break

cap.release()
cv2.destroyAllWindows()
