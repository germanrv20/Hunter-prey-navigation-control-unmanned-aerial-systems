#!/usr/bin/env python3

import rospy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gazebo_msgs.msg import ModelStates

class VisualizadorRobusto:
    def __init__(self):
        rospy.init_node('plotter_trayectoria_larga', anonymous=True)

        # 1. Cambiamos 'deque' por listas vacías '[]' para que NO se borre ningún punto
        self.d1_x, self.d1_y, self.d1_z = [], [], []
        self.d2_x, self.d2_y, self.d2_z = [], [], []

        self.sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback)
        rospy.loginfo("==================================================")
        rospy.loginfo(" Visualizador listo. Grabando trayectoria...")
        rospy.loginfo(" Mueve los drones. La gráfica aparecerá en 40 segundos.")
        rospy.loginfo("==================================================")

    def callback(self, msg):
        try:
            idx1, idx2 = msg.name.index("drone1"), msg.name.index("drone2")
            # Drone 1 (Cazador)
            p1 = msg.pose[idx1].position
            self.d1_x.append(p1.x)
            self.d1_y.append(p1.y)
            self.d1_z.append(p1.z)
            
            # Drone 2 (Presa)
            p2 = msg.pose[idx2].position
            self.d2_x.append(p2.x)
            self.d2_y.append(p2.y)
            self.d2_z.append(p2.z)
        except: 
            pass

    def run(self):
        # 2. En lugar de actualizar en tiempo real, esperamos 40 segundos exactos
        rospy.sleep(500.0)
        
        # 3. Cortamos la comunicación para congelar los datos
        self.sub.unregister()
        rospy.loginfo("¡40 segundos completados! Generando gráfica 3D...")
        
        # 4. Llamamos a la función que dibuja todo de golpe
        self.mostrar_grafica()

    def mostrar_grafica(self):
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # A) Dibujamos las trayectorias
        self.ax.plot(self.d1_x, self.d1_y, self.d1_z, 'g-', linewidth=1.5, label='Trayectoria Cazador')
        self.ax.plot(self.d2_x, self.d2_y, self.d2_z, 'r-', linewidth=1.5, label='Trayectoria Presa')
        
        # B) Marcamos la POSICIÓN FINAL
        if len(self.d1_x) > 0 and len(self.d2_x) > 0:
            self.ax.plot([self.d1_x[-1]], [self.d1_y[-1]], [self.d1_z[-1]], 'go', markersize=10, label='Posición Final Cazador')
            self.ax.plot([self.d2_x[-1]], [self.d2_y[-1]], [self.d2_z[-1]], 'ro', markersize=10, label='Posición Final Presa')

        # C) AUTO-AJUSTE DINÁMICO
        # Unimos todas las coordenadas en una lista para encontrar el máximo y el mínimo global
        todas_x = self.d1_x + self.d2_x
        todas_y = self.d1_y + self.d2_y
        todas_z = self.d1_z + self.d2_z

        # Definimos un margen para que los drones no toquen los bordes de la ventana
        margen = 2.0 
        
        self.ax.set_xlim(min(todas_x) - margen, max(todas_x) + margen)
        self.ax.set_ylim(min(todas_y) - margen, max(todas_y) + margen)
        self.ax.set_zlim(min(todas_z) - margen, max(todas_z) + margen)
        
        self.ax.set_title("Trayectoria Real de los Drones", fontweight='bold')
        self.ax.set_xlabel('Eje X (m)')
        self.ax.set_ylabel('Eje Y (m)')
        self.ax.set_zlabel('Eje Z (m)')
        self.ax.legend()

        plt.show()

if __name__ == '__main__':
    try:
        VisualizadorRobusto().run()
    except rospy.ROSInterruptException:
        pass