#!/usr/bin/env python3
import rospy
import os
import time
import numpy as np
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64

class DistanceDataLogger:
    def __init__(self):
        rospy.init_node('distance_data_logger_npy', anonymous=True)

        # 1. Crear carpeta de destino "data" si no existe
        self.log_dir = os.path.expanduser("~/drone_ws/src/drone_lap/data")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # 2. Generar el nombre del archivo final
        filename = time.strftime("vuelo_%Y%m%d_%H%M%S.npy")
        self.npy_path = os.path.join(self.log_dir, filename)

        # 3. Variables de sincronización
        self.latest_ia_dist = -999.0
        self.latest_error_z = -999.0
        self.recording_active = False

        # 4. Lista RAM donde guardaremos los datos temporalmente
        self.historial_datos = []
        
        rospy.loginfo(f"NODO LOGGER INICIADO. Registrando datos en RAM...")
        rospy.loginfo(f"Al cerrar el nodo (Ctrl+C), se guardará en: {self.npy_path}")

        # 5. Configurar el evento de cierre (IMPORTANTE para guardar el .npy)
        rospy.on_shutdown(self.guardar_archivo_npy)

        # 6. Suscriptores
        rospy.Subscriber('/drone1/estimated_distance', Float64, self.ia_dist_callback)
        rospy.Subscriber('/drone1/vision_error', PointStamped, self.error_callback)

    def ia_dist_callback(self, msg):
        # Actualizamos la última distancia leída por la IA
        self.latest_ia_dist = msg.data

    def error_callback(self, msg):
        self.latest_error_z = msg.point.z
        
        # Ignorar los datos si YOLO está perdido (-999.0)
        if self.latest_error_z != -999.0 and self.latest_ia_dist != -999.0:
            
            # Reconstruir la distancia real (Z_depth)
            distancia_real = self.latest_ia_dist - self.latest_error_z
            tiempo_actual = msg.header.stamp.to_sec()

            # Guardar la fila temporalmente en la lista de RAM
            # Formato: [Tiempo, Distancia_IA, Distancia_Real, Error_Z]
            fila = [tiempo_actual, self.latest_ia_dist, distancia_real, self.latest_error_z]
            self.historial_datos.append(fila)
            
            self.recording_active = True
        else:
            if self.recording_active:
                rospy.logwarn_throttle(2, "YOLO perdido. Pausando grabación de datos...")
                self.recording_active = False

    def guardar_archivo_npy(self):
        """Esta función se llama automáticamente al pulsar Ctrl+C"""
        rospy.loginfo("Deteniendo nodo y procesando matriz de datos...")
        
        if len(self.historial_datos) > 0:
            # Convertimos la lista completa a una matriz NumPy de una sola vez
            matriz_final = np.array(self.historial_datos, dtype=np.float32)
            
            # Guardamos el archivo .npy
            np.save(self.npy_path, matriz_final)
            
            rospy.loginfo(f"¡ÉXITO! Matriz de tamaño {matriz_final.shape} guardada en:")
            rospy.loginfo(self.npy_path)
        else:
            rospy.logwarn("No se registraron datos válidos durante el vuelo. No se creó archivo.")

if __name__ == '__main__':
    try:
        logger = DistanceDataLogger()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
