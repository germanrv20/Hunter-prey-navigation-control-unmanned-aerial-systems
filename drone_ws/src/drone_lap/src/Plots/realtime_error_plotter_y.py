#!/usr/bin/env python

import rospy
import matplotlib.pyplot as plt
from collections import deque
from geometry_msgs.msg import PointStamped
import numpy as np

# --- CONFIGURACIÓN ---
BUFFER_SIZE = 600     # 30 segundos de historial (a 20Hz)
PLOT_RATE = 5         # Refresco visual
SENTINEL_VALUE = -999.0 # Valor centinela de "Objetivo Perdido"
# ---------------------

# Buffers de datos
error_data = deque(maxlen=BUFFER_SIZE)
time_data = deque(maxlen=BUFFER_SIZE)
start_time = rospy.Time(0)

def error_callback(msg):
    """Callback que recibe el error VERTICAL del drone1."""
    global start_time
    
    # Leemos el campo Y (msg.point.y)
    if msg.point.y >= SENTINEL_VALUE + 0.1:
        error = msg.point.y 
    else:
        error = np.nan 
        
    if start_time.is_zero():
        start_time = rospy.Time.now()
        
    relative_time = (rospy.Time.now() - start_time).to_sec()
    
    error_data.append(error)
    time_data.append(relative_time)


def plotter_loop():
    rospy.init_node('realtime_error_plotter_y', anonymous=True)
    
    # Nos suscribimos al MISMO topic, pero leeremos Y
    rospy.Subscriber("/drone1/vision_error", PointStamped, error_callback)
    
    plt.ion() 
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rospy.loginfo("Iniciando ploteo de Error VERTICAL (Ey)...")
    
    rate = rospy.Rate(PLOT_RATE)
    
    while not rospy.is_shutdown():
        
        if len(error_data) > 0:
            
            ax.clear()
            
            # Ventana de 30 segundos
            current_time_end = time_data[-1]
            ax.set_xlim(current_time_end - 30.0, current_time_end + 1.0) 
            
            # --- TRAYECTORIA DEL ERROR Y (Línea Verde) ---
            ax.plot(time_data, error_data, label='Error Vertical (Ey)', color='g', linewidth=2)
            
            # Objetivo
            ax.axhline(0, color='r', linestyle='--', linewidth=2, label='Objetivo (Error = 0)')
            
            # Configuración
            ax.set_title('Control PID: Error Vertical / Altura (Ey)')
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Error en Píxeles (px)')
            
            # Rango Y ajustado (-100 a 100 px suele ser suficiente para ver la oscilación)
            ax.set_ylim(-1.0, 1.0) 
            
            ax.legend()
            ax.grid(True)
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            
        rate.sleep()

if __name__ == '__main__':
    try:
        plotter_loop()
    except rospy.ROSInterruptException:
        pass