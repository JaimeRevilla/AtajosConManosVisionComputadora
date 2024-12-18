import cv2
import numpy as np
import imutils
import pyautogui  # Para controlar el ordenador

cap = cv2.VideoCapture(1)
bg = None

# COLORES PARA VISUALIZACIÓN
color_primario = (204, 204, 0)
color_final = (204, 0, 204)
color_lejos = (255, 0, 0)
color_dedos = (0, 255, 255)

# Ajuste de parámetros
umbral_diferencia = 40
distancia_minima_dedos = 20
altura_minima = 20
angulo_maximo = np.pi / 2
kernel = np.ones((5, 5), np.uint8)

# Parámetros para estabilización
frame_interval = 60
historial_dedos = []
dedos_estables = None

# Estado para funcionalidades
accion_pendiente = None  # Puede ser "abrir_navegador" o "doble_click"
esperando_confirmacion = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer el frame de la cámara.")
        break

    frame = imutils.resize(frame, width=640)
    frame = cv2.flip(frame, 1)
    frameAux = frame.copy()

    if bg is not None:
        # Región de interés ampliada para las 2 manos
        ROI = frame[50:300, 100:540]
        cv2.rectangle(frame, (100-2, 50-2), (540+2, 300+2), color_dedos, 1)
        grayROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        bgROI = bg[50:300, 100:540]

        # Procesamiento del ROI
        dif = cv2.absdiff(grayROI, bgROI)
        _, mask = cv2.threshold(dif, umbral_diferencia, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.medianBlur(mask, 7)

        # Encontrar contornos (RETR_EXTERNAL-> Contornos externos, CHAIN_APPROX_SIMPLE -> Reducir el num de puntos aproximandolos para la memoria)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dedos_totales = 0

        for cnt in cnts:
            if cv2.contourArea(cnt) < 1000:  # Filtrar ruido (descarta contornos pequeños < 1000 pixeles)
                continue

            # Dibujar el contorno valido dentro del ROI con fondo azul y borde 2 pixeles
            cv2.drawContours(ROI, [cnt], 0, (255, 255, 0), 2)

            # Calcular el hull convexo y defectos
            hull = cv2.convexHull(cnt, returnPoints=False) #Calculamos el casco convexo del contorno 
            if hull is not None and len(hull) > 3: #Verificamos que el casco en correcto y tiene por lo menos 3 puntos 
                defectos = cv2.convexityDefects(cnt, hull) #detectamos huecos en el contorno

                # Calcular centro de la palma A traves de detectar momentos -> Distribucion espacial 
                M = cv2.moments(cnt)
                if M["m00"] == 0: M["m00"] = 1 #m00 -> calcula el area, si es 0 se pone a 1 para evitar divisiones raras
                cx = int(M["m10"] / M["m00"]) # Calculamos dimensiones del contorno m10 y m01 son momentos espaciales
                cy = int(M["m01"] / M["m00"])
                cv2.circle(ROI, (cx, cy), 5, (0, 255, 0), -1) #Dibujamos circulo verde para detectar el centroide de la palma

                dedos_levantados = 0
                if defectos is not None:
                    for i in range(defectos.shape[0]):
                        s, e, f, d = defectos[i, 0]
                        principio = tuple(cnt[s][0])
                        final = tuple(cnt[e][0])
                        lejano = tuple(cnt[f][0])

                        # Validar que el punto es un dedo
                        dist_centro = np.linalg.norm(np.array([lejano[0], lejano[1]]) - np.array([cx, cy]))
                        if dist_centro > altura_minima:
                            a = np.linalg.norm(np.array(principio) - np.array(lejano))
                            b = np.linalg.norm(np.array(final) - np.array(lejano))
                            c = np.linalg.norm(np.array(principio) - np.array(final))
                            angulo = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))

                            if angulo < angulo_maximo and d > distancia_minima_dedos:
                                dedos_levantados += 1
                                cv2.circle(ROI, principio, 5, color_primario, -1)
                                cv2.circle(ROI, final, 5, color_final, -1)
                                cv2.circle(ROI, lejano, 5, color_lejos, -1)

                dedos_totales += dedos_levantados + 1

        # Actualizar historial de detecciones Arreglamos el problema de que nos detecta algo nada mas entra en el ROI
        # Creamos un array en el que se guardan los dedos levantados durante 60 frames.
        # Eliminamos los valores iniciales del array cuando llega a la capacidad de 60 datos para poder recoger mas datos
        historial_dedos.append(dedos_totales)
        if len(historial_dedos) > frame_interval:
            historial_dedos.pop(0)
    
        # Comprobamos la estabilidad cuando en el array los 60 valores son iguales
        if historial_dedos.count(historial_dedos[0]) == len(historial_dedos):
            dedos_estables = historial_dedos[0]

        # Lógica de gestos
        if not esperando_confirmacion:
            if dedos_estables == 3:
                print("Se han detectado 3 dedos. Presionando Enter.")
                accion_pendiente = "enter"
                esperando_confirmacion = True

            elif dedos_estables == 4:
                print("Se han detectado 4 dedos. Cerrando ventana (Alt + F4).")
                accion_pendiente = "cerrar_ventana"
                esperando_confirmacion = True

            elif dedos_estables == 5:
                print("Se han detectado 5 dedos. Confirmar para realizar doble clic.")
                accion_pendiente = "doble_click"
                esperando_confirmacion = True

            elif dedos_estables == 6:
                print("Se han detectado 6 dedos. Presionando tecla 'm'.")
                accion_pendiente = "tecla_m"
                esperando_confirmacion = True

            elif dedos_estables == 7:
                print("Se han detectado 7 dedos. Abriendo Microsoft Word.")
                accion_pendiente = "word"
                esperando_confirmacion = True

            elif dedos_estables == 8:
                print("Se han detectado 8 dedos. Confirmar para abrir el navegador.")
                accion_pendiente = "abrir_navegador"
                esperando_confirmacion = True

            elif dedos_estables == 9:
                print("Se han detectado 9 dedos. Abriendo Bloc de Notas.")
                accion_pendiente = "notas"
                esperando_confirmacion = True

            elif dedos_estables == 10:
                print("Se han detectado 10 dedos. Cerrando programa.")
                accion_pendiente = "cerrar_programa"
                esperando_confirmacion = True

        elif esperando_confirmacion:
            if dedos_estables == 1:
                print("Se ha detectado 1 dedo...")
                if accion_pendiente == "enter":
                    print("Pulsando enter...")
                    pyautogui.press('enter')

                elif accion_pendiente == "cerrar_ventana":
                    print("Cerrando ventana...")
                    pyautogui.hotkey('alt', 'f4') 

                elif accion_pendiente == "doble_click":
                    print("Realizando doble clic...")
                    pyautogui.doubleClick()

                elif accion_pendiente == "tecla_m":
                    print("Clicando tecla m...")
                    pyautogui.press('m')

                elif accion_pendiente == "word":
                    print("Abriendo word...")
                    pyautogui.hotkey('win', 'r')
                    pyautogui.write('winword')
                    pyautogui.press('enter') 

                elif accion_pendiente == "abrir_navegador":
                    print("Abriendo navegador...")
                    pyautogui.hotkey('ctrl', 'alt', 'j')

                elif accion_pendiente == "notas":
                    print("Abriendo bloc...")
                    pyautogui.hotkey('win', 'r')
                    pyautogui.write('notepad')
                    pyautogui.press('enter')  

                elif accion_pendiente == "cerrar_programa":
                    print("Cerrando programa...")
                    break  
                esperando_confirmacion = False
                accion_pendiente = None
                
            elif dedos_estables == 2:
                print("Se ha detectado 2 dedos...")
                print("Operación cancelada.")
                esperando_confirmacion = False
                accion_pendiente = None

        # Mostrar máscara binaria
        cv2.imshow('th', mask)

    else:
        print("Fondo no detectado. Presiona 'i' para inicializar el fondo.")

    # Mostrar el número estable de dedos detectados
    if dedos_estables is not None:
        cv2.putText(frame, f'Dedos: {dedos_estables}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color_dedos, 2)

    cv2.imshow('Frame', frame)

    # Manejo de teclas
    k = cv2.waitKey(20)
    if k == ord('i'):
        bg = cv2.cvtColor(frameAux, cv2.COLOR_BGR2GRAY)
        print("Fondo inicializado")
    if k == 27:  # Presionamos 'ESC' para salir
        break

cap.release()
cv2.destroyAllWindows()
