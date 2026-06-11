#!/usr/bin/env python3
import rospy
import numpy as np
import os
import signal
import sys
import copy
from geometry_msgs.msg import PointStamped, TwistStamped
from gazebo_msgs.msg import ModelState, ModelStates

# ==========================================
# --- PARÁMETROS DE CONFIGURACIÓN ---
# ==========================================

MODO_ACTIVO = "ALTURA"  

# !!! IMPORTANTE: Cambia este nombre para cada prueba !!!
NOMBRE_PRUEBA = "PID_V1_Optimo" 

PORCENTAJE_META = 0.10 
TIEMPO_MANTENIMIENTO = 1.5 
TIEMPO_DURACION_EXPERIMENTO = 15.0  # Duración TOTAL de la grabación

# --- CONFIGURACIÓN DEL ESCALÓN ---
MAGNITUD_YAW = 1.0           # Metros a la izquierda/derecha
MAGNITUD_ALTURA = 1.0       # Metros hacia arriba
MAGNITUD_DISTANCIA = -7.0     # Metros hacia adelante

# --- VARIABLES GLOBALES ---
datos_x = []
datos_y = []
datos_z = []

class PIDTracker:
    def __init__(self, nombre):
        self.nombre = nombre
        self.estado = 'ESPERANDO'
        self.t_inicio = 0.0
        self.t_primera_llegada = 0.0
        self.t_entrada_banda = 0.0
        self.error_pico = 0.0
        self.ha_llegado_una_vez = False
        self.curva_actual = [] 

    def procesar_error(self, e_actual, t_actual, lista_resultados):
        e_abs = abs(e_actual)

        # ESTADO 1: Primer instante exacto en el que el PID empieza a actuar
        if self.estado == 'ESPERANDO':
            self.estado = 'OSCILANDO'
            self.error_pico = e_abs # El error que hay AHORA es el máximo del escalón
            self.t_inicio = t_actual
            self.ha_llegado_una_vez = False
            self.curva_actual = [[0.0, e_actual]] 
            rospy.loginfo(f"[{self.nombre}] Midiendo desde el Error Inicial: {e_abs:.2f}")

        # ESTADO 2 y 3: Grabando y analizando
        elif self.estado != 'ESPERANDO':
            self.curva_actual.append([t_actual - self.t_inicio, e_actual])
            
            # (Seguridad) Por si el dron se va en dirección contraria y empeora el error
            if e_abs > self.error_pico:
                self.error_pico = e_abs
                
            limite_banda = self.error_pico * PORCENTAJE_META

            if self.estado == 'OSCILANDO':
                if e_abs <= limite_banda:
                    if not self.ha_llegado_una_vez:
                        self.t_primera_llegada = t_actual
                        self.ha_llegado_una_vez = True
                    self.estado = 'EN_BANDA'
                    self.t_entrada_banda = t_actual

            elif self.estado == 'EN_BANDA':
                if e_abs > limite_banda:
                    self.estado = 'OSCILANDO'
                else:
                    if (t_actual - self.t_entrada_banda) >= TIEMPO_MANTENIMIENTO:
                        t_subida = self.t_primera_llegada - self.t_inicio
                        t_establecimiento = self.t_entrada_banda - self.t_inicio
                        lista_resultados.append([self.error_pico, t_subida, t_establecimiento])
                        
                        rospy.loginfo(f"[{self.nombre}] ESTABILIZADO: ts = {t_establecimiento:.2f}s. (Sigue grabando...)")
                        self.estado = 'ESTABILIZADO' 

class LivePIDAnalyzer:
    def __init__(self):
        rospy.init_node('live_pid_analyzer', anonymous=True)

        self.tracker_x = PIDTracker("YAW")
        self.tracker_y = PIDTracker("ALTURA")
        self.tracker_z = PIDTracker("DISTANCIA")

        self.pose_actual_drone2 = None
        self.pose_inicial_guardada = None
        self.pose_fija_congelador = None 
        
        self.escena_preparada = False
        self.pid_detectado = False

        # Suscriptores
        rospy.Subscriber("/drone1/vision_error", PointStamped, self.error_callback)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.gazebo_cb)
        
        # EL TRUCO: Suscripción a los motores del dron para detectar cuándo arranca el C++
        rospy.Subscriber("/drone1/mavros/setpoint_velocity/cmd_vel", TwistStamped, self.cmd_vel_callback)
        
        self.gazebo_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10)
        
        rospy.loginfo("========================================")
        rospy.loginfo(f" ANALIZADOR PID - MODO: {MODO_ACTIVO} | PRUEBA: {NOMBRE_PRUEBA}")
        rospy.loginfo("========================================")

        # Esperamos 1 segundo para asegurarnos de tener la posición del dron y luego preparamos la escena
        rospy.Timer(rospy.Duration(1.0), self.preparar_escena, oneshot=True)
        
        # Bucle para congelar al drone2
        rospy.Timer(rospy.Duration(0.02), self.congelar_drone)

    def gazebo_cb(self, msg):
        try:
            idx = msg.name.index("drone2")
            self.pose_actual_drone2 = msg.pose[idx]
        except ValueError:
            pass

    def preparar_escena(self, event):
        if self.pose_actual_drone2 is None:
            rospy.logwarn("Buscando drone2 en Gazebo...")
            rospy.Timer(rospy.Duration(1.0), self.preparar_escena, oneshot=True)
            return

        rospy.loginfo(">>> COLOCANDO DRON 2 EN LA POSICIÓN DE PRUEBA... <<<")
        self.pose_inicial_guardada = copy.deepcopy(self.pose_actual_drone2)
        nueva_pose = copy.deepcopy(self.pose_actual_drone2)

        if MODO_ACTIVO == "YAW":
            nueva_pose.position.y += MAGNITUD_YAW
            nueva_pose.position.x += MAGNITUD_DISTANCIA  

        elif MODO_ACTIVO == "ALTURA": 
            nueva_pose.position.z += MAGNITUD_ALTURA
            nueva_pose.position.x += MAGNITUD_DISTANCIA  
        elif MODO_ACTIVO == "DISTANCIA": nueva_pose.position.x += MAGNITUD_DISTANCIA 
        elif MODO_ACTIVO == "ALL":
            nueva_pose.position.y += MAGNITUD_YAW
            nueva_pose.position.z += MAGNITUD_ALTURA
            nueva_pose.position.x += MAGNITUD_DISTANCIA

        self.pose_fija_congelador = nueva_pose
        self.escena_preparada = True

        rospy.loginfo("======================================================")
        rospy.loginfo(" ESCENA LISTA Y CONGELADA.")
        rospy.loginfo(" => AHORA PUEDES LANZAR TU NODO C++ EN OTRA TERMINAL.")
        rospy.loginfo("======================================================")

    def congelar_drone(self, event):
        if self.pose_fija_congelador is not None:
            msg_state = ModelState()
            msg_state.model_name = "drone2"
            msg_state.pose = self.pose_fija_congelador
            self.gazebo_pub.publish(msg_state)

    def cmd_vel_callback(self, msg):
        # Si la escena ya está puesta y aún no habíamos detectado el PID...
        if self.escena_preparada and not self.pid_detectado:
            # Comprobamos si el C++ está publicando alguna velocidad mayor que 0
            vel_x = abs(msg.twist.linear.x)
            vel_z = abs(msg.twist.linear.z)
            vel_yaw = abs(msg.twist.angular.z)
            
            if vel_x > 0.01 or vel_z > 0.01 or vel_yaw > 0.01:
                rospy.loginfo(">>> ¡NODO PID DETECTADO! EMPIEZA LA GRABACIÓN (15s) <<<")
                self.pid_detectado = True
                # Programamos el fin de la grabación a los 15s de este exacto momento
                rospy.Timer(rospy.Duration(TIEMPO_DURACION_EXPERIMENTO), guardar_matriz, oneshot=True)

    def error_callback(self, msg):
        if msg.point.z == -999.0: return
        t_actual = msg.header.stamp.to_sec()
        
        # SOLO procesamos los errores si el PID ya ha sido encendido por el usuario
        if self.pid_detectado:
            if MODO_ACTIVO in ["YAW", "ALL"]:
                self.tracker_x.procesar_error(msg.point.x, t_actual, datos_x)
            if MODO_ACTIVO in ["ALTURA", "ALL"]:
                self.tracker_y.procesar_error(msg.point.y, t_actual, datos_y)
            if MODO_ACTIVO in ["DISTANCIA", "ALL"]:
                self.tracker_z.procesar_error(msg.point.z, t_actual, datos_z)

def guardar_matriz(*args):
    print(f"\n\n--- FIN DEL EXPERIMENTO ({TIEMPO_DURACION_EXPERIMENTO}s COMPLETADOS) ---")
    carpeta_salida = os.path.join(os.path.expanduser('~'), "drone_ws/src/drone_lap/data/")
    os.makedirs(carpeta_salida, exist_ok=True)
    
    def guardar_curva(nombre_eje, prefijo_archivo, tracker):
        if tracker.estado != 'ESPERANDO' and len(tracker.curva_actual) > 0:
            ruta_curvas = os.path.join(carpeta_salida, f"curvas_{prefijo_archivo}_{NOMBRE_PRUEBA}.npy")
            np.save(ruta_curvas, np.array([tracker.curva_actual], dtype=object), allow_pickle=True)
            print(f"-> {nombre_eje}: Curva de {len(tracker.curva_actual)} puntos guardada en:\n   {ruta_curvas}")
        else:
            print(f"-> {nombre_eje}: No se grabó curva.")

    if MODO_ACTIVO in ["YAW", "ALL"]: guardar_curva("YAW (X)", "YAW", analyzer.tracker_x)
    if MODO_ACTIVO in ["ALTURA", "ALL"]: guardar_curva("ALTURA (Y)", "ALTURA", analyzer.tracker_y)
    if MODO_ACTIVO in ["DISTANCIA", "ALL"]: guardar_curva("DISTANCIA (Z)", "DISTANCIA", analyzer.tracker_z)

    # Devolvemos el dron a su sitio por cortesía
    if analyzer.pose_inicial_guardada:
        msg_state = ModelState()
        msg_state.model_name = "drone2"
        msg_state.pose = analyzer.pose_inicial_guardada
        analyzer.gazebo_pub.publish(msg_state)

    print("Cerrando nodo...")
    rospy.signal_shutdown("Completado.")
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, guardar_matriz)
    try:
        global analyzer 
        analyzer = LivePIDAnalyzer()
        rospy.spin()
    except Exception as e:
        pass