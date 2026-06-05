#!/usr/bin/env python
import rospy
import matplotlib.pyplot as plt
import numpy as np
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PointStamped

# Listas para guardar los datos de X e Y
actual_x = []
actual_y = []
desired_x = []
desired_y = []

# Parámetros del círculo (Deben coincidir con tu nodo C++: centro en (0,0), radio 3.0)
CENTER_X = 0.0
CENTER_Y = 0.0
RADIUS = 3.0

def actual_pose_callback(msg):
    """Callback para la pose REAL de Gazebo (Línea Roja)"""
    try:
        # Busca la posición actual del drone2
        idx = msg.name.index("drone2")
        pose = msg.pose[idx]
        actual_x.append(pose.position.x)
        actual_y.append(pose.position.y)
    except ValueError:
        pass

def desired_pose_callback(msg):
    """Callback para la pose DESEADA (Línea Azul)"""
    # El nodo C++ publica el punto GLOBAL que quiere alcanzar
    desired_x.append(msg.point.x)
    desired_y.append(msg.point.y)

def plot_trajectory():
    rospy.init_node('trajectory_plotter', anonymous=True)
    
    # Suscripción a la posición REAL del dron
    rospy.Subscriber("/gazebo/model_states", ModelStates, actual_pose_callback)
    # Suscripción al TARGET GLOBAL que tu nodo C++ está publicando
    rospy.Subscriber("/drone2/debug/global_target", PointStamped, desired_pose_callback)
    
    rospy.loginfo("Iniciando ploteo. Colectando datos durante 45 segundos...")
    
    # Esperar 45 segundos para colectar datos suficientes para varias vueltas
    rospy.sleep(45.0) 
    
    rospy.loginfo("Generando gráfico...")
    
    # --- Generar el círculo de REFERENCIA (Línea Verde) ---
    theta = np.linspace(0, 2 * np.pi, 100)
    ref_x = CENTER_X + RADIUS * np.cos(theta)
    ref_y = CENTER_Y + RADIUS * np.sin(theta)
    
    plt.figure(figsize=(10, 10))
    
    # Plot 1: Trayectoria de Referencia (Círculo Perfecto)
    plt.plot(ref_x, ref_y, 'g--', linewidth=1.5, label='Referencia Teórica (R=3m)')
    
    # Plot 2: Trayectoria Deseada (La que calculó tu código C++)
    if desired_x:
        # Usamos puntos pequeños para mostrar la resolución del setpoint
        plt.plot(desired_x, desired_y, 'b.', markersize=2, label='Deseada (Global Target)')
        
    # Plot 3: Trayectoria Real (Donde está el dron en Gazebo)
    if actual_x:
        # Usamos una línea para mostrar la trayectoria del dron con la deriva
        plt.plot(actual_x, actual_y, 'r-', linewidth=1.5, label='Real (Drift del EKF)')

    plt.xlabel('Posición X Global (m)')
    plt.ylabel('Posición Y Global (m)')
    plt.title('Comparativa de Trayectoria Circular: Deseada vs. Real')
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # Esencial para que el círculo se vea redondo
    
    plot_filename = "trajectory_drift_comparison.png"
    plt.savefig(plot_filename)
    rospy.loginfo("Gráfico guardado en %s", plot_filename)
    
    # Muestra el gráfico
    plt.show() 

if __name__ == '__main__':
    try:
        plot_trajectory()
    except rospy.ROSInterruptException:
        pass