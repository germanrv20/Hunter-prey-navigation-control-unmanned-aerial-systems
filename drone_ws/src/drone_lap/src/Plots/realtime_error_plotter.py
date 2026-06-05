#!/usr/bin/env python

import rospy
import matplotlib.pyplot as plt
from collections import deque
from geometry_msgs.msg import PointStamped
import numpy as np

# --- CONFIGURACIÓN ---
BUFFER_SIZE = 1200
PLOT_RATE = 5
SENTINEL_VALUE = -1.0
# ---------------------

# Buffers de datos
error_data = deque(maxlen=BUFFER_SIZE)
time_data = deque(maxlen=BUFFER_SIZE)
start_time = rospy.Time(0)

def error_callback(msg):
    """Callback que recibe el error horizontal del drone1."""
    global start_time
    
    if msg.point.x >= SENTINEL_VALUE + 0.1:
        error = msg.point.x 
    else:
        error = np.nan 
        
    if start_time.is_zero():
        start_time = rospy.Time.now()
        
    relative_time = (rospy.Time.now() - start_time).to_sec()
    
    error_data.append(error)
    time_data.append(relative_time)


def plotter_loop():
    """Bucle principal para inicializar y actualizar el gráfico."""
    rospy.init_node('realtime_error_plotter', anonymous=True)
    rospy.Subscriber("/drone1/vision_error", PointStamped, error_callback)
    
    plt.ion() 
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rospy.loginfo("Iniciando ploteo en tiempo real. Ventana de 30 segundos activa...")
    
    rate = rospy.Rate(PLOT_RATE)
    
    while not rospy.is_shutdown():
        
        if len(error_data) > 0:
            
            ax.clear()
            
            # --- LÓGICA CLAVE: FIJAR LA VENTANA DE TIEMPO A 30s ---
            current_time_end = time_data[-1]
            ax.set_xlim(current_time_end - 30.0, current_time_end + 1.0) 
            
            # --- TRAYECTORIA DEL ERROR (Línea Azul) ---
            ax.plot(time_data, error_data, label='Error Angular (Ex)', color='b', linewidth=2)
            
            # --- OBJETIVO (Línea Roja Fija en Y=0) ---
            ax.axhline(0, color='r', linestyle='--', linewidth=2, label='Objetivo (Error = 0)')
            
            # Configuración
            ax.set_title('Control PID: Error Angular Horizontal (Ex)')
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Error en Píxeles (px)')
            
            # === CORRECCIÓN CRÍTICA: Rango Y (-100 a 100) ===
            ax.set_ylim(-100, 100) 
            # ==================================================
            
            ax.legend()
            ax.grid(True)
            
            # Dibujar y pausar brevemente para actualizar
            fig.canvas.draw()
            fig.canvas.flush_events()
            
        rate.sleep()

if __name__ == '__main__':
    try:
        plotter_loop()
    except rospy.ROSInterruptException:
        pass