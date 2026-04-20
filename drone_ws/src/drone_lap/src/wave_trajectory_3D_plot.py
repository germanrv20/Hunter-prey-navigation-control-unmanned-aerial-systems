#!/usr/bin/env python3

import rospy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gazebo_msgs.msg import ModelStates
from collections import deque
import numpy as np

# --- CONFIGURACIÓN ---
# 5000 puntos es perfecto para ver todo el recorrido sin borrar rápido
WINDOW_SIZE = 5000 
# ---------------------

class VisualizadorRobusto:
    def __init__(self):
        rospy.init_node('plotter_trayectoria_larga', anonymous=True)

        self.d1_x, self.d1_y, self.d1_z = deque(maxlen=WINDOW_SIZE), deque(maxlen=WINDOW_SIZE), deque(maxlen=WINDOW_SIZE)
        self.d2_x, self.d2_y, self.d2_z = deque(maxlen=WINDOW_SIZE), deque(maxlen=WINDOW_SIZE), deque(maxlen=WINDOW_SIZE)

        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.linea_c1, = self.ax.plot([], [], [], 'g-', linewidth=1.5, label='Cazador')
        self.linea_p2, = self.ax.plot([], [], [], 'r-', linewidth=1.5, label='Presa')
        self.dot_c1, = self.ax.plot([], [], [], 'go', markersize=7)
        self.dot_p2, = self.ax.plot([], [], [], 'ro', markersize=7)

        self.ax.set_xlim(-15, 15)
        self.ax.set_ylim(-15, 15)
        self.ax.set_zlim(0, 12)
        self.ax.legend()

        rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback)
        rospy.loginfo("Visualizador listo. Capturando trayectoria larga...")

    def callback(self, msg):
        try:
            idx1, idx2 = msg.name.index("drone1"), msg.name.index("drone2")
            # Drone 1
            p1 = msg.pose[idx1].position
            self.d1_x.append(p1.x); self.d1_y.append(p1.y); self.d1_z.append(p1.z)
            # Drone 2
            p2 = msg.pose[idx2].position
            self.d2_x.append(p2.x); self.d2_y.append(p2.y); self.d2_z.append(p2.z)
        except: pass

    def actualizar_plot(self):
        # --- SOLUCIÓN AL ERROR DE BROADCAST ---
        # Convertimos a lista y tomamos el tamaño mínimo actual para asegurar consistencia
        x1, y1, z1 = list(self.d1_x), list(self.d1_y), list(self.d1_z)
        x2, y2, z2 = list(self.d2_x), list(self.d2_y), list(self.d2_z)
        
        n1 = min(len(x1), len(y1), len(z1))
        n2 = min(len(x2), len(y2), len(z2))

        if n1 > 1:
            # Dibujamos solo hasta n1 para que X, Y y Z midan lo mismo
            self.linea_c1.set_data(x1[:n1], y1[:n1])
            self.linea_c1.set_3d_properties(z1[:n1])
            self.dot_c1.set_data([x1[n1-1]], [y1[n1-1]])
            self.dot_c1.set_3d_properties([z1[n1-1]])

        if n2 > 1:
            self.linea_p2.set_data(x2[:n2], y2[:n2])
            self.linea_p2.set_3d_properties(z2[:n2])
            self.dot_p2.set_data([x2[n2-1]], [y2[n2-1]])
            self.dot_p2.set_3d_properties([z2[n2-1]])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def run(self):
        # Bajamos un poco la frecuencia de dibujo para no saturar la CPU con 5000 puntos
        rate = rospy.Rate(8) 
        while not rospy.is_shutdown():
            self.actualizar_plot()
            rate.sleep()

if __name__ == '__main__':
    try:
        VisualizadorRobusto().run()
    except rospy.ROSInterruptException:
        pass