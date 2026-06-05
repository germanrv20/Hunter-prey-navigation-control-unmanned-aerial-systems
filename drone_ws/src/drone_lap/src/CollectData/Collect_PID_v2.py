#!/usr/bin/env python3
import rospy
import numpy as np
import os
import sys
import time
from geometry_msgs.msg import PointStamped
from gazebo_msgs.msg import ModelState, ModelStates

# --- PARÁMETROS DE CONFIGURACIÓN DEL EXPERIMENTO ---
# ¿Cuánto se moverá el drone objetivo para probar cada eje? (En metros)
ESCALON_YAW_Y     = 0.4  # Mueve el objetivo 0.4m de lado
ESCALON_ALTURA_Z  = 0.4  # Mueve el objetivo 0.4m hacia arriba
ESCALON_DISTANCIA_X = 0.8  # Aleja el objetivo 0.8m

PORCENTAJE_META = 0.10 # Banda del ±10%
TIEMPO_MANTENIMIENTO = 1.5 # Segundos dentro de la banda para considerarlo estable

class AutomatedPIDTracker:
    def __init__(self, nombre):
        self.nombre = nombre
        self.estado = 'INACTIVO' # INACTIVO, MIDIENDO, ESTABLE
        self.t_inicio = 0.0
        self.t_entrada_banda = 0.0
        self.error_pico = 0.0
        self.ts_final = np.nan

    def iniciar_medicion(self):
        """Se llama justo cuando teletransportamos el dron en Gazebo"""
        self.estado = 'MIDIENDO'
        self.t_inicio = rospy.Time.now().to_sec()
        self.error_pico = 0.0
        self.ts_final = np.nan
        rospy.loginfo(f"[{self.nombre}] ESCALÓN INYECTADO. Midiendo tiempo de establecimiento...")

    def procesar_error(self, e_actual, t_actual):
        if self.estado == 'INACTIVO' or self.estado == 'ESTABLE':
            return

        # Valor absoluto para la banda ±10%
        e_abs = abs(e_actual)

        if self.estado == 'MIDIENDO':
            # La cámara tarda un par de frames en ver el salto máximo
            if e_abs > self.error_pico:
                self.error_pico = e_abs

            # Evitar cálculos si aún no hay un pico válido
            if self.error_pico < 1.0: 
                return

            limite_banda = self.error_pico * PORCENTAJE_META
            
            if e_abs <= limite_banda:
                self.estado = 'EN_BANDA'
                self.t_entrada_banda = t_actual

        elif self.estado == 'EN_BANDA':
            limite_banda = self.error_pico * PORCENTAJE_META
            
            if e_abs > limite_banda:
                # Falsa alarma (oscilación). Vuelve a buscar.
                self.estado = 'MIDIENDO'
            else:
                # Sigue dentro. ¿Ha aguantado el tiempo requerido?
                if (t_actual - self.t_entrada_banda) >= TIEMPO_MANTENIMIENTO:
                    self.ts_final = self.t_entrada_banda - self.t_inicio
                    self.estado = 'ESTABLE'
                    rospy.loginfo(f"[{self.nombre}] PRUEBA SUPERADA! ts = {self.ts_final:.2f}s (Error Pico: {self.error_pico:.1f})")


class AutomatedBench:
    def __init__(self):
        rospy.init_node('automated_pid_bench', anonymous=True)

        self.tracker_yaw = AutomatedPIDTracker("YAW")
        self.tracker_alt = AutomatedPIDTracker("ALTURA")
        self.tracker_dist = AutomatedPIDTracker("DISTANCIA")

        # Pose actual del drone objetivo
        self.pose_drone2 = None

        # Publicador para mover el drone en Gazebo
        self.set_state_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        
        # Suscriptores
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.gazebo_callback)
        rospy.Subscriber('/drone1/vision_error', PointStamped, self.error_callback)

        rospy.loginfo("==================================================")
        rospy.loginfo(" BANCO DE PRUEBAS AUTOMÁTICO INICIADO")
        rospy.loginfo(" Esperando a que el sistema esté listo...")
        rospy.loginfo("==================================================")

    def gazebo_callback(self, msg):
        try:
            idx = msg.name.index("drone2") # Nombre de tu dron objetivo
            self.pose_drone2 = msg.pose[idx]
        except ValueError:
            pass

    def error_callback(self, msg):
        if msg.point.z == -999.0:
            return # YOLO perdido
            
        t_actual = msg.header.stamp.to_sec()
        self.tracker_yaw.procesar_error(msg.point.x, t_actual)
        self.tracker_alt.procesar_error(msg.point.y, t_actual)
        self.tracker_dist.procesar_error(msg.point.z, t_actual)

    def mover_objetivo(self, offset_x=0.0, offset_y=0.0, offset_z=0.0):
        """Teletransporta el drone2 de forma instantánea sumando un offset a su posición actual"""
        if self.pose_drone2 is None:
            rospy.logwarn("No se encuentra drone2 en Gazebo.")
            return False

        msg = ModelState()
        msg.model_name = "drone2"
        msg.pose = self.pose_drone2
        # Aplicamos el salto exacto
        msg.pose.position.x += offset_x
        msg.pose.position.y += offset_y
        msg.pose.position.z += offset_z
        
        # Anulamos inercias para que sea un escalón perfecto
        msg.twist.linear.x = 0; msg.twist.linear.y = 0; msg.twist.linear.z = 0
        
        self.set_state_pub.publish(msg)
        return True

    def ejecutar_bateria_pruebas(self):
        rospy.sleep(3) # Dejar que el dron se estabilice al arrancar

        # --- PRUEBA 1: YAW (Lateral) ---
        rospy.loginfo("\n>>> INICIANDO PRUEBA 1: YAW (Y += 0.4m)")
        if self.mover_objetivo(offset_y=ESCALON_YAW_Y):
            self.tracker_yaw.iniciar_medicion()
            # Esperar hasta que se estabilice
            while self.tracker_yaw.estado != 'ESTABLE' and not rospy.is_shutdown():
                rospy.sleep(0.1)

        rospy.sleep(2) # Pausa entre pruebas

        # --- PRUEBA 2: ALTURA (Vertical) ---
        rospy.loginfo("\n>>> INICIANDO PRUEBA 2: ALTURA (Z += 0.4m)")
        if self.mover_objetivo(offset_z=ESCALON_ALTURA_Z):
            self.tracker_alt.iniciar_medicion()
            while self.tracker_alt.estado != 'ESTABLE' and not rospy.is_shutdown():
                rospy.sleep(0.1)

        rospy.sleep(2)

        # --- PRUEBA 3: DISTANCIA (Avance) ---
        rospy.loginfo("\n>>> INICIANDO PRUEBA 3: DISTANCIA (X += 0.8m)")
        if self.mover_objetivo(offset_x=ESCALON_DISTANCIA_X):
            self.tracker_dist.iniciar_medicion()
            while self.tracker_dist.estado != 'ESTABLE' and not rospy.is_shutdown():
                rospy.sleep(0.1)

        # --- FIN DE LAS PRUEBAS: GUARDAR DATOS ---
        rospy.loginfo("\n==================================================")
        rospy.loginfo(" TODAS LAS PRUEBAS COMPLETADAS. GUARDANDO DATOS...")
        
        matriz_final = np.array([[
            self.tracker_yaw.ts_final, 
            self.tracker_alt.ts_final, 
            self.tracker_dist.ts_final
        ]])

        ruta_salida = os.path.expanduser("~/drone_ws/src/drone_lap/data/tiempos_PID_auto.npy")
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        np.save(ruta_salida, matriz_final)
        
        rospy.loginfo(f"Matriz guardada en: {ruta_salida}")
        rospy.loginfo(f"Resultados -> YAW: {self.tracker_yaw.ts_final:.2f}s | ALT: {self.tracker_alt.ts_final:.2f}s | DIST: {self.tracker_dist.ts_final:.2f}s")
        rospy.loginfo("==================================================")
        
        # Apagar nodo automáticamente
        rospy.signal_shutdown("Pruebas finalizadas")

if __name__ == '__main__':
    try:
        bench = AutomatedBench()
        bench.ejecutar_bateria_pruebas()
    except rospy.ROSInterruptException:
        pass
