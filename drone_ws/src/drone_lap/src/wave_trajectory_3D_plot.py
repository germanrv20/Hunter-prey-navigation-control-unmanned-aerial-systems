#!/usr/bin/env python3

import rospy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Necesario para 3D
from geometry_msgs.msg import PoseStamped
from collections import deque
import numpy as np

# --- CONFIGURACIÓN ---
WINDOW_SIZE = 400     # Historial de puntos
# ---------------------

# Buffers Drone 2 (Presa)
ref2_x = deque(maxlen=WINDOW_SIZE)
ref2_y = deque(maxlen=WINDOW_SIZE)
ref2_z = deque(maxlen=WINDOW_SIZE)

act2_x = deque(maxlen=WINDOW_SIZE)
act2_y = deque(maxlen=WINDOW_SIZE)
act2_z = deque(maxlen=WINDOW_SIZE)

# Buffers Drone 1 (Cazador)
act1_x = deque(maxlen=WINDOW_SIZE)
act1_y = deque(maxlen=WINDOW_SIZE)
act1_z = deque(maxlen=WINDOW_SIZE)

# --- CALLBACKS ---
def setpoint_d2_cb(msg):
    ref2_x.append(msg.pose.position.x)
    ref2_y.append(msg.pose.position.y)
    ref2_z.append(msg.pose.position.z)

def local_pose_d2_cb(msg):
    act2_x.append(msg.pose.position.x)
    act2_y.append(msg.pose.position.y)
    act2_z.append(msg.pose.position.z)

def local_pose_d1_cb(msg):
    act1_x.append(msg.pose.position.x)
    act1_y.append(msg.pose.position.y)
    act1_z.append(msg.pose.position.z)

# --- BUCLE PRINCIPAL ---
def live_plotter_3d():
    rospy.init_node('live_3d_plotter_dual', anonymous=True)
    
    # Suscripciones Drone 2 (Presa)
    rospy.Subscriber("/drone2/mavros/setpoint_position/local", PoseStamped, setpoint_d2_cb)
    rospy.Subscriber("/drone2/mavros/local_position/pose", PoseStamped, local_pose_d2_cb)
    
    # Suscripción Drone 1 (Cazador)
    rospy.Subscriber("/drone1/mavros/local_position/pose", PoseStamped, local_pose_d1_cb)
    
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d') 
    
    rospy.loginfo("Iniciando ploteo 3D DUAL: Cazador (Dron 1) vs Presa (Dron 2)...")
    
    rate = rospy.Rate(10) # 10 Hz
    
    while not rospy.is_shutdown():
        # Esperamos a tener datos de ambos drones
        if len(ref2_x) > 0 and len(act2_x) > 0 and len(act1_x) > 0:
            
            ax.clear()
            
            # --- PLOTEAR DATOS ---
            # Setpoint Drone 2 (Ideal) - Azul Punteado
            ax.plot(ref2_x, ref2_y, ref2_z, 'b--', label='Setpoint Dron 2', linewidth=1)
            
            # Realidad Drone 2 (Presa) - Rojo Sólido
            ax.plot(act2_x, act2_y, act2_z, 'r-', label='Dron 2 (Presa)', linewidth=2)
            
            # Realidad Drone 1 (Cazador) - Verde Sólido
            ax.plot(act1_x, act1_y, act1_z, 'g-', label='Dron 1 (Cazador)', linewidth=2)
            
            # --- CONFIGURACIÓN DE EJES ---
            ax.set_title("Trayectoria 3D de Persecución", fontweight='bold')
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (Altura - m)")
            
            # 🚧 LÍMITES AMPLIADOS 🚧
            # He ampliado los límites para asegurarme de que el Dron 1 (que suele empezar en 0,0) también se vea.
            # Ajústalos si ves que se salen del marco.
            ax.set_xlim(-10, 10)
            ax.set_ylim(-15, 5)
            ax.set_zlim(0, 15)
            
            ax.legend(loc='upper right')
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            
        rate.sleep()

if __name__ == '__main__':
    try:
        live_plotter_3d()
    except rospy.ROSInterruptException:
        pass