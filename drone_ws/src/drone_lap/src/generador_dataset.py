#!/usr/bin/env python3

import rospy
import cv2
import os
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class DatasetGenerator:
    def __init__(self):
        rospy.init_node('generador_dataset_node')
        
        # --- CONFIGURACIÓN ---
        self.meta_fotos = 500        # Número de fotos a sacar
        self.intervalo = 0.5         # Segundos entre cada foto (para que no sean idénticas)
        self.nombre_carpeta = "dataset_tfg_v2" # Nombre de la carpeta donde se guardarán
        
        # Ruta completa: /home/tu_usuario/dataset_tfg_v2
        self.ruta_guardado = os.path.join(os.path.expanduser('~'), self.nombre_carpeta)
        
        # Crear carpeta si no existe
        if not os.path.exists(self.ruta_guardado):
            os.makedirs(self.ruta_guardado)
            print(f"📁 Carpeta creada: {self.ruta_guardado}")
        else:
            print(f"📂 Usando carpeta existente: {self.ruta_guardado}")
            
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber("/webcam/image_raw", Image, self.callback)
        
        self.contador = 0
        self.ultimo_tiempo = time.time()
        
        print("\n" + "="*40)
        print(f" INICIANDO CAPTURA DE {self.meta_fotos} IMÁGENES")
        print(" Mueve el dron objetivo para variar el ángulo y la distancia.")
        print("="*40 + "\n")

    def callback(self, msg):
        # Si ya llegamos a 400, cerramos
        if self.contador >= self.meta_fotos:
            print("\n✅ ¡Meta alcanzada! Dataset completado.")
            rospy.signal_shutdown("Proceso terminado")
            return

        # Control de tiempo (para no sacar 30 fotos iguales en 1 segundo)
        ahora = time.time()
        if ahora - self.ultimo_tiempo > self.intervalo:
            try:
                # Convertir imagen
                frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                
                # Nombre del archivo: img_0001.jpg, img_0002.jpg...
                nombre_archivo = f"img_{self.contador:04d}.jpg"
                ruta_completa = os.path.join(self.ruta_guardado, nombre_archivo)
                
                # Guardar en disco
                cv2.imwrite(ruta_completa, frame)
                
                # Actualizar contadores
                self.contador += 1
                self.ultimo_tiempo = ahora
                
                # Feedback en terminal
                print(f"📸 Guardada {self.contador}/{self.meta_fotos}: {nombre_archivo}")
                
                # Mostrar en pantalla con contador
                cv2.putText(frame, f"REC: {self.contador}/{self.meta_fotos}", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Generando Dataset...", frame)
                cv2.waitKey(1)
                
            except Exception as e:
                print(e)

if __name__ == '__main__':
    try:
        DatasetGenerator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
