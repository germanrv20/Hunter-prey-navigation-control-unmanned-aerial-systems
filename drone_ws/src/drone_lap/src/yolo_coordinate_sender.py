#!/usr/bin/env python3
import rospy
import cv2
import os
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Quaternion 
from cv_bridge import CvBridge
from ultralytics import YOLO
from std_msgs.msg import Int32

class KalmanBBoxTracker:
    def __init__(self):
        # 8 variables de estado: [x1, y1, x2, y2, vel_x1, vel_y1, vel_x2, vel_y2]
        # 4 variables de medida: [x1, y1, x2, y2]
        self.kf = cv2.KalmanFilter(8, 4)
        
        # Matriz de Transición (A) - Modelo de Velocidad Constante
        dt = 0.05 # Asumimos salto de tiempo unitario por frame
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, dt, 0,  0,  0],
            [0, 1, 0, 0, 0,  dt, 0,  0],
            [0, 0, 1, 0, 0,  0,  dt, 0],
            [0, 0, 0, 1, 0,  0,  0,  dt],
            [0, 0, 0, 0, 1,  0,  0,  0],
            [0, 0, 0, 0, 0,  1,  0,  0],
            [0, 0, 0, 0, 0,  0,  1,  0],
            [0, 0, 0, 0, 0,  0,  0,  1]
        ], np.float32)

        # Matriz de Medida (H) - Solo leemos las 4 posiciones espaciales
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], np.float32)

        # --- TUNEADO DEL FILTRO  ---
        # Q: Ruido del proceso. Si lo bajas, el tracking es más suave pero tarda más en girar.
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.1
        
        # R: Ruido de la medida. Si lo subes, confías MENOS en YOLO y MÁS en las matemáticas.
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.5
        
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
        self.initialized = False

    def update(self, box):
        """Actualiza el filtro con una nueva lectura de YOLO"""
        meas = np.array(box, dtype=np.float32).reshape(-1, 1)
        
        if not self.initialized:
            # Inicializamos el estado en la primera posición detectada
            self.kf.statePost = np.array([box[0], box[1], box[2], box[3], 0, 0, 0, 0], np.float32).reshape(-1, 1)
            self.initialized = True
            return box
            
        # 1. Predecir 
        self.kf.predict()
        # 2. Corregir con la lectura real
        self.kf.correct(meas)
        
        # Devolvemos las 4 coordenadas filtradas
        return self.kf.statePost[:4].flatten()

    def predict_only(self):
        """Si YOLO falla, el filtro adivina dónde está el dron"""
        if self.initialized:
            pred = self.kf.predict()
            return pred[:4].flatten()
        return None

class YoloSender:
    def __init__(self):
        rospy.init_node('yolo_sender_node')
        
        path = os.path.join(os.path.expanduser('~'), 'drone_ws/src/drone_lap/models/best.pt')
        try:
            self.model = YOLO(path)
            print(f"Modelo cargado correctamente desde: {path}")
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            exit()
        
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("webcam/image_raw", Image, self.callback) #/capture_node/camera/image
        self.coord_pub = rospy.Publisher("/drone1/yolo_pixel_coords", Quaternion, queue_size=1)
        self.state_pub = rospy.Publisher("/drone1/yolo_state", Int32, queue_size=1)

        # Instanciamos el Filtro de Kalman
        self.tracker = KalmanBBoxTracker()
        
        # Contador de frames que YOLO no ve al dron
        self.lost_frames = 0
        self.max_lost_frames = 15 # Si lo pierde de vista 15 frames seguidos, se rinde.

    def callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            results = self.model.predict(img, verbose=False, conf=0.50)
            
            x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0
            state_id = 0 # 0: BUSCANDO, 1: OK, 2: PREDICIENDO
            
            # --- LÓGICA DE ESTADOS ---
            if len(results[0].boxes) > 0:
                # CASO 1: YOLO DETECTA
                self.lost_frames = 0
                state_id = 1
                raw_box = results[0].boxes[0].xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = self.tracker.update(raw_box)
            
            elif self.tracker.initialized and self.lost_frames < self.max_lost_frames:
                # CASO 2: YOLO NO VE, PERO KALMAN PREDICE
                self.lost_frames += 1
                state_id = 2
                pred = self.tracker.predict_only()
                if pred is not None:
                    x1, y1, x2, y2 = pred
            
            else:
                # CASO 3: PERDIDO
                self.tracker.initialized = False
                state_id = 0

            # --- PUBLICAR DATOS ---
            # Coordenadas
            coords_msg = Quaternion(float(x1), float(y1), float(x2), float(y2))
            self.coord_pub.publish(coords_msg)
            
            # Estado
            self.state_pub.publish(Int32(state_id))

        except Exception as e:
            rospy.logerr(f"Error en YoloSender: {e}")

if __name__ == '__main__':
    try:
        YoloSender()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()