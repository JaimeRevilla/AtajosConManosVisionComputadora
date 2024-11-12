import os
import subprocess
import pyautogui

# Diccionario para mapear gestos a comandos del sistema
GESTOS_COMANDOS = {
    "okey": "%windir%\\system32\\notepad.exe",  # Ruta de bloc de notas
    "puño": "notepad.exe",  # Abre el notepad
    "dedo_arriba": "explorer.exe",  # Abre el Explorador de archivos
    "victoria": "calc.exe",  # Abre la Calculadora
    "mano_abierta": "mspaint.exe"  # Abre Paint
}

def captura_pantalla():
    """Captura la pantalla y la guarda en el directorio actual."""
    screenshot = pyautogui.screenshot()
    screenshot.save("captura.png")
    print("Captura de pantalla guardada como 'captura.png'.")

def ejecutar_comando(gesto):
    """
    Ejecuta un comando del sistema o una acción específica dependiendo del gesto detectado.

    Parámetros:
    gesto (str): El nombre del gesto detectado.
    """
    # Si el gesto está en el diccionario, ejecuta el comando correspondiente
    comando = GESTOS_COMANDOS.get(gesto)
    if comando:
        try:
            # Usamos startfile para abrir aplicaciones que requieren interfaz gráfica
            if os.path.isfile(comando):
                os.startfile(comando)
            else:
                subprocess.run(comando, shell=True)
            print(f"Ejecutando comando para el gesto: {gesto}")
        except Exception as e:
            print(f"Error al ejecutar el comando para el gesto '{gesto}': {e}")
    else:
        # Si el gesto no tiene un comando específico, realiza una acción personalizada
        if gesto == "puño":
            captura_pantalla()
        else:
            print(f"Gesto '{gesto}' no reconocido o sin acción asignada.")

if __name__ == "__main__":
    # Prueba rápida
    ejecutar_comando("okey")
