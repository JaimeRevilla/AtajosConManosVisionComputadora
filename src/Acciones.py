import os
import subprocess

# Diccionario para mapear gestos a comandos del sistema
GESTOS_COMANDOS = {
    "okey": "%windir%\system32\notepad.exe",  # Ruta de blocdenotas
    "puño": "notepad.exe",  # Abre el notepad
    "dedo_arriba": "explorer.exe",  # Abre el Explorador de archivos
    "victoria": "calc.exe",  # Abre la Calculadora
    "mano_abierta": "mspaint.exe"  # Abre Paint
}

def ejecutar_comando(gesto):
    """
    Ejecuta un comando del sistema dependiendo del gesto detectado.

    Parámetros:
    gesto (str): El nombre del gesto detectado.
    """
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
        print(f"Gesto '{gesto}' no reconocido.")

if __name__ == "__main__":
    ejecutar_comando("okey")
