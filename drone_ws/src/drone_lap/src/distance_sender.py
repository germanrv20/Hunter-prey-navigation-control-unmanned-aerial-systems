#!/usr/bin/env python3
import rospy
import os
import torch
import torch.nn as nn
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Float64

# 1. DEFINICIÓN DE LA RED NEURONAL
# Tiene que ser EXACTAMENTE igual a la que usaste para entrenar
class MlpDistance(nn.Module):
    def __init__(self):
        super(MlpDistance, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 32),   
            nn.ELU(),
            nn.Linear(32, 16), 
            nn.ELU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

class DistanceSenderNode:
    def __init__(self):
        rospy.init_node('distance_sender_node', anonymous=False)

        # -- Cargar el Modelo y el Scaler --
        model_path = os.path.expanduser("~/drone_ws/src/drone_lap/models/mlp_distance_model.pth")
        self.model = MlpDistance()
        self.model.eval() # Modo evaluación (apaga cálculos innecesarios para inferencia)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.scaler_mean = checkpoint['scaler_mean']
            self.scaler_scale = checkpoint['scaler_scale']
            rospy.loginfo("--- NODO IA INICIADO ---")
            rospy.loginfo(f"Modelo cargado. Scaler -> Media: {self.scaler_mean:.2f} | Desviacion: {self.scaler_scale:.2f}")
        else:
            rospy.logerr(f"No se encontro el archivo del modelo en {model_path}")
            rospy.signal_shutdown("Falta el modelo de IA")
            return

        # -- Publicadores y Suscriptores --
        # Publicamos la distancia en Float64
        self.dist_pub = rospy.Publisher('/drone1/estimated_distance', Float64, queue_size=10)
        
        # Nos suscribimos a YOLO (x=x1, y=y1, z=x2, w=y2)
        self.yolo_sub = rospy.Subscriber('/drone1/yolo_pixel_coords', Quaternion, self.yolo_callback)

    def yolo_callback(self, msg):
        x1, y1 = msg.x, msg.y
        x2, y2 = msg.z, msg.w

        if x2 > x1 and y2 > y1:
            area = (x2 - x1) * (y2 - y1)

            if area > 0:
                import math
                # EL TRUCO MATEMÁTICO EN ROS
                linear_size = 1.0 / math.sqrt(area)
                
                # ESCALAMOS
                scaled_size = (linear_size - self.scaler_mean) / self.scaler_scale

                with torch.no_grad():
                    input_tensor = torch.tensor([[scaled_size]], dtype=torch.float32)
                    estimated_distance = self.model(input_tensor).item()

                dist_msg = Float64()
                dist_msg.data = estimated_distance
                self.dist_pub.publish(dist_msg)
        else:
            # Si se pierde YOLO, puedes decidir qué publicar. 
            # Mandamos un valor negativo para que C++ sepa que no hay dato válido.
            dist_msg = Float64()
            dist_msg.data = -999.0
            self.dist_pub.publish(dist_msg)

if __name__ == '__main__':
    try:
        node = DistanceSenderNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass