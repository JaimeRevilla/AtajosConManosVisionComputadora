### Descripcion 

Nuestro proyecto consiste en el desarrollo de un sistema de reconocimiento de gestos de la mano en tiempo real, utilizando la biblioteca MediaPipe únicamente para la localización y seguimiento de la mano. Una vez localizada la mano, se emplearán técnicas clásicas de procesamiento de imágenes, como la detección de contornos y el uso de convex hull (envoltura convexa), para analizar la posición de los dedos y reconocer gestos específicos (como la formación de una "W"). En función de los gestos detectados, el sistema ejecutará acciones específicas en el escritorio de Windows, como abrir aplicaciones o realizar comandos de control.



## Procesos de instalacion 

- pip install opencv-python mediapipe
- pip install pyautogui
