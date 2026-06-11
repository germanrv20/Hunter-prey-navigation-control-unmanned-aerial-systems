#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Int32

class MetricsEvaluator:
    def __init__(self):
        rospy.init_node('metrics_evaluator_node', anonymous=True)

        # --- Acumuladores de Error (EPE y RMSE) ---
        self.sum_epe_yolo = 0.0
        self.sum_epe_kalman = 0.0
        self.sum_sq_err_z = 0.0
        
        # --- Contadores de Frames por Estado ---
        self.frames_yolo = 0    # Estado 1: YOLO detecta
        self.frames_kalman = 0  # Estado 2: Kalman predice
        self.frames_lost = 0    # Estado 0: Perdido total

        # Variable para sincronizar el estado actual
        self.current_state = 0 

        # --- Suscriptores ---
        rospy.Subscriber("/drone1/vision_error", PointStamped, self.error_callback)
        rospy.Subscriber("/drone1/yolo_state", Int32, self.state_callback)

        rospy.on_shutdown(self.print_final_metrics)
        rospy.loginfo("--- EVALUADOR EPE/RMSE/TLR INICIADO ---")
        rospy.loginfo("Graba el vuelo y pulsa Ctrl+C para generar el informe.")

    def state_callback(self, msg):
        self.current_state = msg.data
        
        # Contamos los frames totales y por estado
        if self.current_state == 0:
            self.frames_lost += 1
        elif self.current_state == 1:
            self.frames_yolo += 1
        elif self.current_state == 2:
            self.frames_kalman += 1

    def error_callback(self, msg):
        err_x = msg.point.x
        err_y = msg.point.y
        err_z = msg.point.z

        # Ignoramos los saltos de emergencia a -999.0
        if err_x > -990.0:
            # Cálculo de la magnitud del error en el plano de la imagen (End-Point Error)
            magnitude_2d = math.sqrt(err_x**2 + err_y**2)

            # Clasificamos el error según si lo detecta YOLO o lo predice Kalman
            if self.current_state == 1:
                self.sum_epe_yolo += magnitude_2d
            elif self.current_state == 2:
                self.sum_epe_kalman += magnitude_2d

            # Acumulamos para el RMSE de la distancia Z
            if err_z > -990.0:
                self.sum_sq_err_z += err_z**2

    def print_final_metrics(self):
        total_frames = self.frames_yolo + self.frames_kalman + self.frames_lost
        valid_frames = self.frames_yolo + self.frames_kalman

        rospy.loginfo("\n==================================================")
        rospy.loginfo("       INFORME DE RENDIMIENTO DEL TFG (IBVS)      ")
        rospy.loginfo("==================================================")

        if total_frames > 0:
            # 1. CÁLCULO DEL TLR
            tlr_percent = (self.frames_lost / total_frames) * 100.0
            yolo_percent = (self.frames_yolo / total_frames) * 100.0
            kalman_percent = (self.frames_kalman / total_frames) * 100.0

            rospy.loginfo("[1] MÉTRICAS DE VISIÓN Y PÉRDIDA (TLR)")
            rospy.loginfo(f"    - Frames Analizados: {total_frames}")
            rospy.loginfo(f"    - Visión Activa (YOLO):     {yolo_percent:.2f} %")
            rospy.loginfo(f"    - Tracking Ciego (Kalman):  {kalman_percent:.2f} %")
            rospy.loginfo(f"    -> Target Loss Rate (TLR):  {tlr_percent:.2f} %")
            
            # 2. CÁLCULO DEL EPE (Separado por tu sugerencia)
            rospy.loginfo("\n[2] MÉTRICAS DE CONTROL: END-POINT ERROR (EPE 2D)")
            if self.frames_yolo > 0:
                epe_yolo = self.sum_epe_yolo / self.frames_yolo
                rospy.loginfo(f"    - EPE durante YOLO:   {epe_yolo:.4f} (Magnitud Normalizada)")
            else:
                rospy.loginfo("    - EPE durante YOLO:   N/A (No hubo detecciones)")

            if self.frames_kalman > 0:
                epe_kalman = self.sum_epe_kalman / self.frames_kalman
                rospy.loginfo(f"    - EPE durante Kalman: {epe_kalman:.4f} (Magnitud Normalizada)")
            else:
                rospy.loginfo("    - EPE durante Kalman: N/A (No se usó Kalman)")

            if valid_frames > 0:
                epe_total = (self.sum_epe_yolo + self.sum_epe_kalman) / valid_frames
                rospy.loginfo(f"    -> EPE Medio Total:   {epe_total:.4f} (Magnitud Normalizada)")

            # 3. CÁLCULO DEL RMSE EN DISTANCIA
            if valid_frames > 0:
                rmse_z = math.sqrt(self.sum_sq_err_z / valid_frames)
                rospy.loginfo("\n[3] MÉTRICAS DE CONTROL: DISTANCIA (EUCLÍDEA)")
                rospy.loginfo(f"    -> RMSE Distancia (Z): {rmse_z:.4f} metros")

        else:
            rospy.logwarn("No se registraron datos durante el vuelo.")
            
        rospy.loginfo("==================================================\n")

if __name__ == '__main__':
    try:
        evaluator = MetricsEvaluator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
