#!/usr/bin/env python

import rospy
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseStamped
from collections import deque
import numpy as np

# --- CONFIGURACIÓN ---
DRONE_NS = "/drone2" # Cambia a /drone1 si quieres ver el seguidor
WINDOW_SIZE = 400    # Cuántos puntos guardar (aprox. 20 segundos a 20Hz)
# ---------------------

# Buffers para Setpoint (Lo que ordenas)
ref_x = deque(maxlen=WINDOW_SIZE)
ref_y = deque(maxlen=WINDOW_SIZE)
ref_z = deque(maxlen=WINDOW_SIZE)
ref_t = deque(maxlen=WINDOW_SIZE)

# Buffers para Realidad (Lo que hace el dron)
act_x = deque(maxlen=WINDOW_SIZE)
act_y = deque(maxlen=WINDOW_SIZE)
act_z = deque(maxlen=WINDOW_SIZE)
act_t = deque(maxlen=WINDOW_SIZE)

start_time = None

def setpoint_cb(msg):
    global start_time
    if start_time is None: start_time = rospy.Time.now()
    
    ref_x.append(msg.pose.position.x)
    ref_y.append(msg.pose.position.y)
    ref_z.append(msg.pose.position.z)
    ref_t.append((rospy.Time.now() - start_time).to_sec())

def local_pose_cb(msg):
    global start_time
    if start_time is None: start_time = rospy.Time.now()

    act_x.append(msg.pose.position.x)
    act_y.append(msg.pose.position.y)
    act_z.append(msg.pose.position.z)
    act_t.append((rospy.Time.now() - start_time).to_sec())

def live_plotter():
    rospy.init_node('live_trajectory_plotter', anonymous=True)
    
    # Suscripciones
    rospy.Subscriber(f"{DRONE_NS}/mavros/setpoint_position/local", PoseStamped, setpoint_cb)
    rospy.Subscriber(f"{DRONE_NS}/mavros/local_position/pose", PoseStamped, local_pose_cb)
    
    # Configurar Gráfica
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    rospy.loginfo(f"Iniciando ploteo para {DRONE_NS}...")
    
    rate = rospy.Rate(10) # Refresco visual a 10Hz
    
    while not rospy.is_shutdown():
        if len(ref_x) > 0 and len(act_x) > 0:
            
            # --- GRÁFICO 1: TRAYECTORIA X-Y (VISTA SUPERIOR) ---
            ax1.clear()
            # Dibujar lo que ordenamos (Azul punteado)
            ax1.plot(ref_x, ref_y, 'b--', label='Setpoint (Ideal)', alpha=0.6)
            # Dibujar lo que hace el dron (Rojo sólido)
            ax1.plot(act_x, act_y, 'r-', label='Real (Dron)', linewidth=2)
            
            ax1.set_title("Trayectoria X-Y (Círculo)")
            ax1.set_xlabel("X (m)")
            ax1.set_ylabel("Y (m)")
            ax1.axis('equal') # Para que el círculo se vea redondo
            ax1.grid(True)
            ax1.legend(loc='upper right')

            # --- GRÁFICO 2: ALTURA Z vs TIEMPO ---
            ax2.clear()
            ax2.plot(ref_t, ref_z, 'b--', label='Setpoint Z')
            ax2.plot(act_t, act_z, 'r-', label='Real Z')
            
            ax2.set_title("Altura (Z) vs Tiempo")
            ax2.set_xlabel("Tiempo (s)")
            ax2.set_ylabel("Altura (m)")
            ax2.set_ylim(0, 15) # Ajusta si vuelas más alto
            ax2.grid(True)
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            
        rate.sleep()

if __name__ == '__main__':
    try:
        live_plotter()
    except rospy.ROSInterruptException:
        pass