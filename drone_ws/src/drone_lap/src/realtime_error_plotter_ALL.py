#!/usr/bin/env python3
import rospy
import matplotlib.pyplot as plt
from collections import deque
from geometry_msgs.msg import PointStamped
import numpy as np

# --- CONFIGURACIÓN ---
BUFFER_SIZE = 1200      # Historial de datos
PLOT_RATE = 5           # Hz de refresco visual
SENTINEL_VALUE = -990.0 # Consideramos todo lo menor a esto como "Objetivo Perdido" (-999.0 en C++)
# ---------------------

# Buffers de datos
error_x_data = deque(maxlen=BUFFER_SIZE) # Error Yaw
error_y_data = deque(maxlen=BUFFER_SIZE) # Error Altura
time_data = deque(maxlen=BUFFER_SIZE)
start_time = rospy.Time(0)

def error_callback(msg):
    """Callback que recibe ambos errores a la vez (X e Y) de la cámara."""
    global start_time
    
    if start_time.is_zero():
        start_time = rospy.Time.now()
        
    relative_time = (rospy.Time.now() - start_time).to_sec()
    
    # --- LEER ERROR HORIZONTAL (YAW) ---
    if msg.point.x > SENTINEL_VALUE:
        error_x_data.append(msg.point.x)
    else:
        error_x_data.append(np.nan) # np.nan hace que la gráfica no dibuje nada si se pierde
        
    # --- LEER ERROR VERTICAL (ALTURA) ---
    if msg.point.y > SENTINEL_VALUE:
        error_y_data.append(msg.point.y)
    else:
        error_y_data.append(np.nan)
        
    time_data.append(relative_time)

def plotter_loop():
    """Bucle principal para inicializar y actualizar el gráfico doble."""
    rospy.init_node('realtime_error_combined_plotter', anonymous=True)
    
    # Nos suscribimos al topic que tiene ambas coordenadas
    rospy.Subscriber("/drone1/vision_error", PointStamped, error_callback)
    
    plt.ion() 
    # Creamos 2 subgráficas (ax1 y ax2) que comparten el eje X (sharex=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.canvas.manager.set_window_title('Análisis de Control PID DUAL')
    
    rospy.loginfo("Iniciando ploteo COMBINADO (Yaw y Altura). Ventana de 30s...")
    
    rate = rospy.Rate(PLOT_RATE)
    
    while not rospy.is_shutdown():
        if len(time_data) > 0:
            
            ax1.clear()
            ax2.clear()
            
            # Fijar la ventana de 30s
            current_time_end = time_data[-1]
            ax1.set_xlim(current_time_end - 30.0, current_time_end + 1.0)
            ax2.set_xlim(current_time_end - 30.0, current_time_end + 1.0)
            
            # ==========================================
            # GRÁFICA SUPERIOR: ERROR YAW (Eje X)
            # ==========================================
            ax1.plot(time_data, error_x_data, label='Error Yaw (Ex)', color='blue', linewidth=2)
            ax1.axhline(0, color='red', linestyle='--', linewidth=2, label='Objetivo (0 px)')
            
            ax1.set_title('Control PID Horizontal (Giro / Yaw)', fontweight='bold')
            ax1.set_ylabel('Error en Píxeles (px)')
            ax1.set_ylim(-100, 100) # Rango fijo para ver bien las oscilaciones
            ax1.legend(loc='upper right')
            ax1.grid(True)
            
            # ==========================================
            # GRÁFICA INFERIOR: ERROR ALTURA (Eje Y)
            # ==========================================
            ax2.plot(time_data, error_y_data, label='Error Altura (Ey)', color='green', linewidth=2)
            ax2.axhline(0, color='red', linestyle='--', linewidth=2, label='Objetivo (0 px)')
            
            ax2.set_title('Control PID Vertical (Subida-Bajada / Z)', fontweight='bold')
            ax2.set_xlabel('Tiempo (s)')
            ax2.set_ylabel('Error en Píxeles (px)')
            ax2.set_ylim(-100, 100) # Rango fijo
            ax2.legend(loc='upper right')
            ax2.grid(True)
            
            # Ajustar los márgenes para que no se superpongan los textos
            plt.tight_layout()
            
            # Dibujar y refrescar
            fig.canvas.draw()
            fig.canvas.flush_events()
            
        rate.sleep()

if __name__ == '__main__':
    try:
        plotter_loop()
    except rospy.ROSInterruptException:
        pass