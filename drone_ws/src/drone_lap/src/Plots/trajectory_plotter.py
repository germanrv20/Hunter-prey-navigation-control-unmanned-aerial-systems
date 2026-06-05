#!/usr/bin/env python3
import rospy
from gazebo_msgs.msg import ModelStates
import matplotlib.pyplot as plt

class TrajectoryPlotter3D:
    def __init__(self):
        rospy.init_node('trajectory_plotter_3d_node')
        
        # Suscriptor para leer las posiciones reales de Gazebo
        self.sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback)

        # Listas para guardar las coordenadas históricas
        self.d1_x, self.d1_y, self.d1_z = [], [], []  # Cazador (Drone 1)
        self.d2_x, self.d2_y, self.d2_z = [], [], []  # Objetivo (Drone 2)

        # --- CONFIGURACIÓN DE LA GRÁFICA 3D ---
        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Línea azul para el cazador, Línea roja a trazos para el objetivo
        self.line1, = self.ax.plot([], [], [], 'b-', linewidth=2, label='Cazador (Drone 1)')
        self.line2, = self.ax.plot([], [], [], 'r--', linewidth=2, label='Objetivo (Drone 2)')
        
        self.ax.set_xlabel('Posición X (metros)', fontsize=10, labelpad=10)
        self.ax.set_ylabel('Posición Y (metros)', fontsize=10, labelpad=10)
        self.ax.set_zlabel('Altura Z (metros)', fontsize=10, labelpad=10)
        self.ax.set_title('Seguimiento de Trayectoria 3D en Tiempo Real', fontsize=14, fontweight='bold')
        self.ax.legend(loc='upper right')

    def callback(self, msg):
        try:
            if 'drone1' in msg.name and 'drone2' in msg.name:
                idx1 = msg.name.index('drone1')
                idx2 = msg.name.index('drone2')

                self.d1_x.append(msg.pose[idx1].position.x)
                self.d1_y.append(msg.pose[idx1].position.y)
                self.d1_z.append(msg.pose[idx1].position.z)
                
                self.d2_x.append(msg.pose[idx2].position.x)
                self.d2_y.append(msg.pose[idx2].position.y)
                self.d2_z.append(msg.pose[idx2].position.z)
        except ValueError:
            pass

    def update_plot(self):
        rate = rospy.Rate(5)
        
        print("\n" + "="*50)
        print("🚀 PLOTTER 3D EN TIEMPO REAL INICIADO")
        print("Gira la gráfica con el ratón para verla desde varios ángulos.")
        print("Cierra con Ctrl+C cuando termines.")
        print("="*50 + "\n")

        while not rospy.is_shutdown():
            # --- ESCUDO ANTI-CRASHEOS ---
            # Obtenemos la longitud mínima de las listas para evitar que se desincronicen
            len1 = min(len(self.d1_x), len(self.d1_y), len(self.d1_z))
            len2 = min(len(self.d2_x), len(self.d2_y), len(self.d2_z))

            if len1 > 0 and len2 > 0:
                # Cortamos los datos al mismo tamaño exacto
                c_d1_x = self.d1_x[:len1]
                c_d1_y = self.d1_y[:len1]
                c_d1_z = self.d1_z[:len1]

                c_d2_x = self.d2_x[:len2]
                c_d2_y = self.d2_y[:len2]
                c_d2_z = self.d2_z[:len2]

                # --- ACTUALIZAR LÍNEAS EN 3D ---
                self.line1.set_data(c_d1_x, c_d1_y)
                self.line1.set_3d_properties(c_d1_z)
                
                self.line2.set_data(c_d2_x, c_d2_y)
                self.line2.set_3d_properties(c_d2_z)

                # Ajustar los límites dinámicamente
                all_x = c_d1_x + c_d2_x
                all_y = c_d1_y + c_d2_y
                all_z = c_d1_z + c_d2_z
                margen = 1.0 
                
                self.ax.set_xlim(min(all_x) - margen, max(all_x) + margen)
                self.ax.set_ylim(min(all_y) - margen, max(all_y) + margen)
                self.ax.set_zlim(max(0, min(all_z) - margen), max(all_z) + margen)

                # Dibujar
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                
            rate.sleep()

if __name__ == '__main__':
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        plotter = TrajectoryPlotter3D()
        plotter.update_plot()
    except rospy.ROSInterruptException:
        pass