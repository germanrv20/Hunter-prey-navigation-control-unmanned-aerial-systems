
# 🚁 Hunter-Prey Navigation & Control for Unmanned Aerial Systems (UAS)

![ROS](https://img.shields.io/badge/ROS-Noetic-green?logo=ros)
![C++](https://img.shields.io/badge/C++-17-blue?logo=c%2B%2B)
![Python](https://img.shields.io/badge/Python-3.8+-yellow?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Deep_Learning-red)
![TensorRT](https://img.shields.io/badge/TensorRT-FP16-76B900)
![Gazebo](https://img.shields.io/badge/Gazebo-Simulation-orange)

## 📌 Descripción del Proyecto

Este repositorio contiene el código fuente para el desarrollo de un sistema autónomo de navegación y persecución entre dos vehículos aéreos no tripulados (drones). Este proyecto ha sido desarrollado como Trabajo de Fin de Grado (TFG) en la Universidad de Granada (UGR)[cite: 7].

El sistema implementa una arquitectura de control de tipo **"Cazador-Presa"** (Hunter-Prey) validada en un entorno de simulación realista (Gazebo + ArduPilot SITL) y preparada para su despliegue en hardware real (NVIDIA Jetson Orin Nano)[cite: 7]. El dron "cazador" es capaz de localizar, seguir y mantener una distancia de seguridad con el dron "presa" utilizando exclusivamente **visión monocular** (una sola cámara), prescindiendo del GPS (haciéndolo inmune al *spoofing*) y de sensores pesados como los LiDAR[cite: 7].

### ✨ Características Principales
* **Visión Artificial:** Detección en tiempo real del dron objetivo mediante **YOLOv8n** optimizado con **TensorRT (FP16)** para maximizar los fotogramas por segundo (FPS) en hardware embebido[cite: 7].
* **Dataset Propio:** Entrenamiento del modelo con un dataset mixto compuesto por imágenes sintéticas generadas en Gazebo e imágenes reales capturadas en pruebas de vuelo de campo en la Base Aérea Militar de Armilla y en el club de vuelo junto al pantano de Cubillas.
  * 🔗 **Enlace al Dataset (Roboflow):** [Dataset Synthetic Drone](https://app.roboflow.com/germanrv-uyz9j/synthetic-drone-d2ius/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)
* **Estimación de Distancia Geométrica:** Tras evaluar modelos de regresión de aprendizaje automático, la estimación de distancia longitudinal en la versión final se realiza mediante un modelo geométrico Pinhole fundamentado en la anchura del *bounding box* detectado ($Z_{c} = \frac{f_{x} \cdot X_{c}}{x}$)[cite: 7]. Esto garantiza la obtención de métricas con un coste computacional de $O(1)$.
* **Filtrado Dual:** Uso de un Filtro de Kalman y un Filtro de Media Móvil Exponencial (EMA) para estabilizar las cuatro coordenadas de la detección visual y proporcionar continuidad temporal y predictiva ante oclusiones o ruido del sensor[cite: 7].
* **Control IBVS (Image-Based Visual Servoing):** Implementación de lazos de control PID independientes para guiñada (Yaw), altura y avance longitudinal, compensando el *Roll* y el *Pitch* de la cámara mediante rotaciones matriciales de actitud[cite: 7].

---

## 🏗 Arquitectura del Sistema (Separation of Concerns)


<img width="1808" height="635" alt="Captura de pantalla de 2026-07-04 14-44-26" src="https://github.com/user-attachments/assets/3fc9576d-cc4e-458b-82f6-0bf33dd194b1" />

Para asegurar el aislamiento de los procesos y evitar cuellos de botella asíncronos, el sistema divide las responsabilidades en múltiples nodos de ROS:

1. **`yolo_sender_node` (Python):** Procesa la imagen de la cámara (`/webcam/image_raw`), ejecuta la inferencia acelerada con la red neuronal y publica las coordenadas del *bounding box* en el tópico `/drone1/yolo_pixel_coords`[cite: 7].
2. **`drone_projection_opencv_node` (C++):** Nodo central de procesamiento geométrico y visión[cite: 7]. 
   * Aplica los filtros Kalman y EMA a la caja delimitadora[cite: 7].
   * Calcula la distancia estimada mediante trigonometría[cite: 7].
   * Realiza la compensación de la actitud de la cámara usando el cuaternión de orientación inercial (aislando las matrices de rotación) para desplazar el centro virtual[cite: 7].
   * Publica los errores normalizados en `/drone1/vision_error`[cite: 7].
3. **`drone1_follow_xy_pid` (C++):** Nodo de actuación que aloja los controladores PID reactivos[cite: 7]. Transforma los errores visuales y de profundidad en consignas cinemáticas de velocidad (`cmd_vel`) y las inyecta en el bucle de control del autopiloto a través de MAVROS[cite: 7].

---

## ⚙️ Requisitos y Dependencias

Este proyecto está diseñado para funcionar sobre **Ubuntu 20.04** y **ROS Noetic**.

* **ROS Noetic** (desktop-full)
* **Gazebo 11**
* **ArduPilot SITL** y MAVROS (para simulación de vuelo y enlace telemétrico)[cite: 7]
* **Hardware Recomendado:** NVIDIA Jetson Orin Nano (para ejecución en mundo real)[cite: 7]
* **Python 3.8+:** `ultralytics`, `numpy`, `opencv-python`, librerías TensorRT
* **C++ 17:** OpenCV (`cv_bridge`), Eigen3, tf2[cite: 7]

---

## 🚀 Instalación y Uso

### 1. Clonar y Compilar
Clona este repositorio dentro de la carpeta `src` de tu *workspace* de ROS (ej. `~/drone_ws/src/`):

```bash
cd ~/drone_ws/src
git clone [https://github.com/germanrv20/Hunter-prey-navigation-control-unmanned-aerial-systems.git](https://github.com/germanrv20/Hunter-prey-navigation-control-unmanned-aerial-systems.git) drone_lap
cd ~/drone_ws
catkin_make
source devel/setup.bash<img width="1808" height="635" alt="Captura de pantalla de 2026-07-04 14-44-26" src="https://github.com/user-attachments/assets/3b34cb73-1d04-4b0a-8630-8125af80a509" />
<img width="1808" height="635" alt="Captura de pantalla de 2026-07-04 14-44-26" src="https://github.com/user-attachments/assets/b67f9aec-4268-4b8b-b322-8ff79084b067" />
<img width="1808" height="635" alt="Captura de pantalla de 2026-07-04 14-44-26" src="https://github.com/user-attachments/assets/a55789ca-b2e9-400e-8857-bb06a1b98a02" />
