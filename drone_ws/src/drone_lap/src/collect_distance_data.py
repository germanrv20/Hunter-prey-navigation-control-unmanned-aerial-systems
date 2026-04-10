#!/usr/bin/env python3
import rospy
import os
import numpy as np
import tf.transformations as tf_trans
from geometry_msgs.msg import Pose, Quaternion
from gazebo_msgs.msg import ModelStates

class DistanceDataCollector:
    def __init__(self):
        rospy.init_node('collect_distance_data', anonymous=True)

        # Variables para almacenar las poses
        self.pose_d1 = None
        self.pose_d2 = None
        
        # Variables YOLO
        self.yolo_detected = False
        self.yolo_bbox_area = 0.0
        
        # Lista temporal para acumular los datos en la memoria RAM
        self.data_list = []
        
        # Archivo NumPy (.npy)
        self.npy_file_path = os.path.expanduser("~/drone_ws/src/drone_lap/data/distance_data_v2.npy")
        os.makedirs(os.path.dirname(self.npy_file_path), exist_ok=True)

        rospy.loginfo(f"Data collector iniciado. Los datos se guardarán en: {self.npy_file_path} al pulsar Ctrl+C")

        # Matriz de rotación fija FLU a CV (OpenCV)
        self.R_flu2cv = np.array([
            [ 0, -1,  0],
            [ 0,  0, -1],
            [ 1,  0,  0]
        ])

        # Suscriptores
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_states_cb)
        rospy.Subscriber("/drone1/yolo_pixel_coords", Quaternion, self.yolo_cb)

        # Usamos el hook de ROS para asegurar que se guarda al cerrar el nodo (Ctrl+C)
        rospy.on_shutdown(self.save_to_numpy)

        # Temporizador para capturar datos a 5Hz
        self.timer = rospy.Timer(rospy.Duration(0.2), self.process_data_cb)

    def pose_to_matrix(self, pose):
        """ Equivalente a poseGazeboToTransform """
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        T = tf_trans.quaternion_matrix(q)
        T[0, 3] = pose.position.x
        T[1, 3] = pose.position.y
        T[2, 3] = pose.position.z
        return T

    def get_camera_transform(self):
        """ Equivalente a getCameraTransform """
        T_c_d1 = np.eye(4)
        T_c_d1[2, 3] = 0.13 # Offset en Z
        return T_c_d1

    def model_states_cb(self, msg):
        try:
            idx_d1 = msg.name.index("drone1")
            idx_d2 = msg.name.index("drone2")
            self.pose_d1 = msg.pose[idx_d1]
            self.pose_d2 = msg.pose[idx_d2]
        except ValueError:
            pass 

    def yolo_cb(self, msg):
        if msg.z > msg.x:
            yolo_bbox_width = msg.z - msg.x
            yolo_bbox_height = msg.w - msg.y
            self.yolo_bbox_area = yolo_bbox_width * yolo_bbox_height
            self.yolo_detected = True
        else:
            self.yolo_detected = False

    def process_data_cb(self, event):
        if self.pose_d1 and self.pose_d2 and self.yolo_detected:
            # 1. Obtener matrices T de Gazebo
            Tm_d1 = self.pose_to_matrix(self.pose_d1)
            Tm_d2 = self.pose_to_matrix(self.pose_d2)

            # 2. Inversas
            Td1_m = np.linalg.inv(Tm_d1)
            Tc_d1 = self.get_camera_transform()
            Tc_d1_inv = np.linalg.inv(Tc_d1)

            # 3. Transformación
            Tc_d2_flu = Tc_d1_inv @ Td1_m @ Tm_d2
            P_math_4d = np.array([0.0, 0.0, 0.0, 1.0])
            P_math_flu = Tc_d2_flu @ P_math_4d
            P_math_cv = self.R_flu2cv @ P_math_flu[:3]

            # 4. Extraer Z_depth (profundidad en el eje óptico)
            Z_depth = P_math_cv[2]

            # 5. Guardar el par [Distancia, Área] en la lista de memoria
            self.data_list.append([Z_depth, self.yolo_bbox_area])
            
            rospy.loginfo(f"Capturado -> Z_depth: {Z_depth:.2f}m | Area: {self.yolo_bbox_area:.1f} px2 (Muestras: {len(self.data_list)})")

            # Autoguardado de seguridad cada 50 muestras
            if len(self.data_list) % 50 == 0:
                self.save_to_numpy()

    def save_to_numpy(self):
        """ Guarda toda la lista en un array de NumPy al cerrar el script """
        if len(self.data_list) > 0:
            data_array = np.array(self.data_list)
            np.save(self.npy_file_path, data_array)
            rospy.loginfo(f"\n--- ¡Datos guardados con éxito en formato NumPy! --- \nArchivo: {self.npy_file_path} \nTotal: {len(data_array)} muestras registradas.\n")

if __name__ == '__main__':
    try:
        collector = DistanceDataCollector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass