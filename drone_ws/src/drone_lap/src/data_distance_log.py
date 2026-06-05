#!/usr/bin/env python3
import rospy
import os
import time
import math
import numpy as np
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64

class GroundTruthLogger:
    def __init__(self):
        rospy.init_node('ia_vs_real_logger', anonymous=True)

        # 1. Crear carpeta destino
        self.log_dir = os.path.expanduser("~/drone_ws/src/drone_lap/data")
        os.makedirs(self.log_dir, exist_ok=True)

        # 2. Generar archivo
        filename = time.strftime("distancia_IA_vs_Real_%Y%m%d_%H%M%S.npy")
        self.npy_path = os.path.join(self.log_dir, filename)

        # 3. Variables de sincronización
        self.latest_ia_dist = -999.0
        self.pos_dron1 = None
        self.pos_dron2 = None
        
        self.historial_datos = []
        
        rospy.loginfo("================================================")
        rospy.loginfo(" NODO LOGGER: IA vs GAZEBO GROUND TRUTH INICIADO")
        rospy.loginfo(" Recopilando datos en RAM...")
        rospy.loginfo(f" Al pulsar Ctrl+C se guardará en: {filename}")
        rospy.loginfo("================================================")

        rospy.on_shutdown(self.guardar_archivo_npy)

        # 4. Suscriptores (IA y Ground Truth de Gazebo)
        rospy.Subscriber('/drone1/estimated_distance', Float64, self.ia_callback)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.gazebo_callback)

    def ia_callback(self, msg):
        self.latest_ia_dist = msg.data

    def gazebo_callback(self, msg):
        # 1. Buscar las posiciones de ambos drones en la simulación
        try:
            idx_d1 = msg.name.index("drone1") # El dron que persigue
            idx_d2 = msg.name.index("drone2") # El dron objetivo
            
            self.pos_dron1 = msg.pose[idx_d1].position
            self.pos_dron2 = msg.pose[idx_d2].position
        except ValueError:
            # Si no encuentra a los drones en Gazebo, no hacemos nada
            return

        # 2. Si tenemos la posición real de ambos y YOLO no ha perdido el objetivo
        if self.pos_dron1 and self.pos_dron2 and self.latest_ia_dist != -999.0:
            
            # Calcular la Distancia Real EUCLIDIANA (Pitágoras en 3D) entre los dos drones
            dx = self.pos_dron2.x - self.pos_dron1.x
            dy = self.pos_dron2.y - self.pos_dron1.y
            dz = self.pos_dron2.z - self.pos_dron1.z
            distancia_real_absoluta = math.sqrt(dx**2 + dy**2 + dz**2)

            # Calcular el error crudo de la IA
            error_absoluto = abs(self.latest_ia_dist - distancia_real_absoluta)
            tiempo_actual = rospy.Time.now().to_sec()

            # Guardar la fila temporalmente en la lista de RAM
            # [0: Tiempo, 1: Distancia IA, 2: Distancia Real, 3: Error de la IA]
            fila = [tiempo_actual, self.latest_ia_dist, distancia_real_absoluta, error_absoluto]
            self.historial_datos.append(fila)

    def guardar_archivo_npy(self):
        rospy.loginfo("\nProcesando matriz de datos y guardando...")
        
        if len(self.historial_datos) > 0:
            matriz_final = np.array(self.historial_datos, dtype=np.float32)
            np.save(self.npy_path, matriz_final)
            rospy.loginfo(f"¡ÉXITO! Se guardaron {len(self.historial_datos)} muestras perfectamente validadas.")
        else:
            rospy.logwarn("No se registraron datos válidos durante el vuelo.")

if __name__ == '__main__':
    try:
        logger = GroundTruthLogger()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
