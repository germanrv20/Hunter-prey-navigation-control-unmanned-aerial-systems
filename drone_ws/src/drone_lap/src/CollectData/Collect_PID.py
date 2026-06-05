#!/usr/bin/env python3
import rospy
import numpy as np
import os
import signal
import sys
import argparse
from geometry_msgs.msg import PointStamped

# --- PARÁMETROS DE CONFIGURACIÓN ---
# ¡NUEVOS UMBRALES! Como el C++ normaliza (-1 a 1), 0.1 equivale a un 10% del ancho de pantalla
UMBRAL_X_NORM = 0.10   # Aprox 32 píxeles de desvío
UMBRAL_Y_NORM = 0.10   # Aprox 24 píxeles de desvío
UMBRAL_Z_M    = 0.5    # 0.5 metros de desvío en profundidad

PORCENTAJE_META = 0.10 # Banda del ±10% del pico del error
TIEMPO_MANTENIMIENTO = 0.5 # Segundos de estabilidad requerida

# --- VARIABLES GLOBALES ---
tiempos_x = []
tiempos_y = []
tiempos_z = []
modo_activo = "ALL"

class PIDTracker:
    """Clase que rastrea el tiempo de establecimiento real evitando falsos positivos por oscilación"""
    def __init__(self, nombre, umbral):
        self.nombre = nombre
        self.umbral = umbral
        
        self.estado = 'ESPERANDO' # Estados: ESPERANDO, OSCILANDO, EN_BANDA
        self.t_inicio = 0.0
        self.t_entrada_banda = 0.0
        self.error_pico = 0.0

    def procesar_error(self, e_actual, t_actual, lista_resultados):
        e_abs = abs(e_actual)

        # ESTADO 1: Vuelo normal, esperando el "tirón"
        if self.estado == 'ESPERANDO':
            if e_abs > self.umbral:
                self.estado = 'OSCILANDO'
                self.error_pico = e_abs
                self.t_inicio = t_actual
                rospy.loginfo(f"[{self.nombre}] Perturbación detectada (Error: {e_abs:.2f}). Midiendo...")

        # ESTADO 2: El dron corrige el error
        elif self.estado == 'OSCILANDO':
            # Actualizamos el pico si el error crece para calcular bien el 10%
            if e_abs > self.error_pico:
                self.error_pico = e_abs
                
            limite_banda = self.error_pico * PORCENTAJE_META
            
            # Entra en la banda del ±10%
            if e_abs <= limite_banda:
                self.estado = 'EN_BANDA'
                self.t_entrada_banda = t_actual

        # ESTADO 3: Comprobamos si se mantiene estable en la banda
        elif self.estado == 'EN_BANDA':
            limite_banda = self.error_pico * PORCENTAJE_META
            
            if e_abs > limite_banda:
                # Falsa alarma (se salió de la banda)
                self.estado = 'OSCILANDO'
            else:
                # Sigue dentro de la banda. ¿Pasó el tiempo de seguridad?
                if (t_actual - self.t_entrada_banda) >= TIEMPO_MANTENIMIENTO:
                    # ¡Estable! 
                    ts = self.t_entrada_banda - self.t_inicio
                    lista_resultados.append(ts)
                    rospy.loginfo(f"[{self.nombre}] ESTABILIZADO! ts = {ts:.2f}s (Pico inicial: {self.error_pico:.2f})")
                    
                    self.estado = 'ESPERANDO'
                    self.error_pico = 0.0

class LivePIDAnalyzer:
    def __init__(self):
        rospy.init_node('live_pid_analyzer', anonymous=True)

        self.tracker_x = PIDTracker("YAW", UMBRAL_X_NORM)
        self.tracker_y = PIDTracker("ALTURA", UMBRAL_Y_NORM)
        self.tracker_z = PIDTracker("DISTANCIA", UMBRAL_Z_M)

        rospy.Subscriber("/drone1/vision_error", PointStamped, self.error_callback)
        
        rospy.loginfo("========================================")
        rospy.loginfo(f" NODO ANALIZADOR PID - MODO: {modo_activo}")
        rospy.loginfo(" Arrastra el objetivo en Gazebo para evaluar.")
        rospy.loginfo(" Pulsa Ctrl+C para generar archivo .npy")
        rospy.loginfo("========================================")

    def error_callback(self, msg):
        if msg.point.z == -999.0:
            # YOLO perdido, reseteamos todos los trackers
            self.tracker_x.estado = 'ESPERANDO'
            self.tracker_y.estado = 'ESPERANDO'
            self.tracker_z.estado = 'ESPERANDO'
            return

        t_actual = msg.header.stamp.to_sec()
        
        # Procesamos solo el eje que el usuario haya seleccionado
        if modo_activo in ["YAW", "ALL"]:
            self.tracker_x.procesar_error(msg.point.x, t_actual, tiempos_x)
            
        if modo_activo in ["ALTURA", "ALL"]:
            self.tracker_y.procesar_error(msg.point.y, t_actual, tiempos_y)
            
        if modo_activo in ["DISTANCIA", "ALL"]:
            self.tracker_z.procesar_error(msg.point.z, t_actual, tiempos_z)


def guardar_matriz(sig, frame):
    print("\n\n--- CERRANDO NODO Y GENERANDO ARCHIVO .NPY ---")
    
    def imprimir_resumen(nombre, lista):
        if len(lista) > 0:
            media = np.mean(lista)
            print(f"-> {nombre} ({len(lista)} pruebas): ts Medio = {media:.2f} s")
        else:
            print(f"-> {nombre}: 0 pruebas completadas.")

    if modo_activo in ["YAW", "ALL"]: imprimir_resumen("YAW (X)", tiempos_x)
    if modo_activo in ["ALTURA", "ALL"]: imprimir_resumen("ALTURA (Y)", tiempos_y)
    if modo_activo in ["DISTANCIA", "ALL"]: imprimir_resumen("DISTANCIA (Z)", tiempos_z)

    # Lógica de guardado según el modo
    carpeta_salida = os.path.join(os.path.expanduser('~'), "drone_ws/src/drone_lap/data/")
    os.makedirs(carpeta_salida, exist_ok=True)

    if modo_activo == "ALL":
        # Modo antiguo: guarda los 3 en una matriz con padding de NaNs
        max_len = max(len(tiempos_x), len(tiempos_y), len(tiempos_z), 1) 
        ts_x_pad = tiempos_x + [np.nan] * (max_len - len(tiempos_x))
        ts_y_pad = tiempos_y + [np.nan] * (max_len - len(tiempos_y))
        ts_z_pad = tiempos_z + [np.nan] * (max_len - len(tiempos_z))
        
        matriz_final = np.column_stack((ts_x_pad, ts_y_pad, ts_z_pad))
        ruta_salida = os.path.join(carpeta_salida, "ts_vuelo_vivo_ALL.npy")
        np.save(ruta_salida, matriz_final)
        
    else:
        # Modo específico: Guarda un array 1D limpio solo con los datos de ese eje
        if modo_activo == "YAW":
            data_to_save = np.array(tiempos_x)
            ruta_salida = os.path.join(carpeta_salida, "dataset_ts_YAW.npy")
        elif modo_activo == "ALTURA":
            data_to_save = np.array(tiempos_y)
            ruta_salida = os.path.join(carpeta_salida, "dataset_ts_ALTURA.npy")
        elif modo_activo == "DISTANCIA":
            data_to_save = np.array(tiempos_z)
            ruta_salida = os.path.join(carpeta_salida, "dataset_ts_DISTANCIA.npy")
            
        np.save(ruta_salida, data_to_save)

    print(f"\n¡Dataset guardado con éxito en:\n {ruta_salida}")
    print("=============================================\n")
    sys.exit(0)

if __name__ == '__main__':
    # Configuración de Argparse para permitir elegir el modo desde la consola
    parser = argparse.ArgumentParser(description="Analizador de PID para el Dron (Calcula Tiempo de Establecimiento)")
    parser.add_argument('--modo', type=str, default='ALL', choices=['YAW', 'ALTURA', 'DISTANCIA', 'ALL'], 
                        help="Elige qué eje analizar: YAW, ALTURA, DISTANCIA, o ALL")
    
    # rospy.myargv saca los argumentos propios de ROS para que argparse no se rompa al lanzarlo con rosrun
    args = parser.parse_args(rospy.myargv()[1:])
    modo_activo = args.modo

    signal.signal(signal.SIGINT, guardar_matriz)
    try:
        analyzer = LivePIDAnalyzer()
        rospy.spin()
    except Exception as e:
        pass