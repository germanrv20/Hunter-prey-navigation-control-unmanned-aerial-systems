# 🚁 Hunter-Prey Navigation & Control for Unmanned Aerial Systems (UAS)

![ROS](https://img.shields.io/badge/ROS-Noetic-green?logo=ros)
![C++](https://img.shields.io/badge/C++-17-blue?logo=c%2B%2B)
![Python](https://img.shields.io/badge/Python-3.8+-yellow?logo=python)
![TensorRT](https://img.shields.io/badge/NVIDIA-TensorRT-76B900?logo=nvidia)
![Gazebo](https://img.shields.io/badge/Gazebo-Simulation-orange)

> 🎓 **Trabajo de Fin de Grado (Ingeniería Informática - Universidad de Granada)**
> Calificación: **10 (Propuesta a Matrícula de Honor)**

## 📖 Descripción del Proyecto

Este repositorio contiene el código fuente de un sistema autónomo de navegación y persecución entre dos vehículos aéreos no tripulados (drones). 

El sistema implementa una arquitectura de control **"Cazador-Presa"** (Hunter-Prey) desplegada tanto en simulación realista (Gazebo + ArduPilot SITL) como en hardware físico (NVIDIA Jetson Orin Nano). El dron "cazador" es capaz de localizar, seguir y mantener una distancia de seguridad con el dron "presa" utilizando **exclusivamente visión monocular a bordo**, logrando un sistema **inmune al GPS-Spoofing** al prescindir de balizas externas o sensores pesados como LiDAR.

### ✨ Características Principales
* **Percepción Visual en Tiempo Real:** Detección del UAV objetivo mediante **YOLOv8n**, optimizado con **TensorRT (FP16)** para su ejecución en *Edge Computing* alcanzando hasta 85 FPS.
* **Filtrado Estocástico y Continuidad:** Implementación de un doble filtrado (**Kalman + EMA**) directamente sobre las coordenadas del *bounding box* para lidiar con oclusiones temporales y ruido del sensor.
* **Geometría Proyectiva y Compensación de Actitud:** Uso de OpenCV y Eigen3 para rotar el rayo de proyección de la cámara basándose en la IMU (Roll y Pitch). Esto evita que el dron persiga falsos errores visuales producidos por su propia inclinación al avanzar.
* **Estimación de Distancia $O(1)$:** Cálculo geométrico de la profundidad utilizando el modelo de cámara estenopeica (Pinhole) a partir del ancho aparente de la detección, logrando coste computacional casi nulo.
* **Control IBVS (Image-Based Visual Servoing):** Triple controlador PID desacoplado (Yaw, Altura, Distancia) que transforma el error normalizado en píxeles y metros en consignas reactivas de velocidad.

---

## 🛩️ Custom Dataset: Vuelo Real

Para garantizar la robustez del modelo YOLOv8n en el mundo real, los pesos no dependen únicamente de simulaciones. **El dataset de entrenamiento fue creado desde cero mediante campañas de vuelo real** realizadas en dos localizaciones clave:
* 🪖 Base Aérea Militar de Armilla (Granada).
* 🌊 Club de vuelo situado junto al Pantano de Cubillas.

Esto permitió entrenar la red neuronal con una inmensa variabilidad de fondos, condiciones de iluminación y perspectivas dinámicas reales.

---

## 🏗️ Arquitectura del Sistema (Separation of Concerns)

Para maximizar el rendimiento y evitar cuellos de botella, la arquitectura distribuye las responsabilidades en múltiples nodos de ROS altamente desacoplados:

1. **`yolo_sender_node` (Python):** Captura el flujo de `/webcam/image_raw`, ejecuta el motor de TensorRT con el modelo de YOLO y publica un array (`Quaternion`) con las coordenadas del *bounding box*.
2. **`drone_projection_opencv_node` (C++):** El núcleo matemático. Se suscribe a YOLO, aplica el filtro de Kalman + EMA, proyecta el plano virtual mediante Eigen3 para compensar actitud, estima la distancia real y publica un `PointStamped` con los errores limpios ($e_x, e_y, e_z$).
3. **`drone1_follow_xy_pid` (C++ / Python):** Nodo de control puro. Lee los errores geométricos y ejecuta la lógica de los PIDs enviando las correcciones de velocidad de vuelta al autopiloto a través de `MAVROS` (`cmd_vel`).

---

## 🛠️ Requisitos y Dependencias

Este proyecto está diseñado y validado sobre **Ubuntu 20.04** nativo y **ROS Noetic**.

* **Core:** ROS Noetic (desktop-full), MAVROS, ArduPilot SITL.
* **Simulación:** Gazebo 11, *iq_sim* (Intelligent Quads).
* **Deep Learning (Python 3.x):** `ultralytics`, `torch`, `tensorrt`, `numpy`, `cv_bridge`.
* **Procesamiento de Vuelo (C++17):** 
  * `OpenCV 4.x` (Geometría, distorsión y transformaciones).
  * `Eigen3` (Álgebra lineal para cuaterniones y matrices 4x4).
  * `tf2_geometry_msgs`

---

## 🚀 Instalación y Uso

### 1. Clonar y Compilar
Asegúrate de tener un *workspace* de ROS configurado (ej. `~/drone_ws/src/`).

```bash
cd ~/drone_ws/src
git clone [https://github.com/germanrv20/Hunter-prey-navigation-control-unmanned-aerial-systems.git](https://github.com/germanrv20/Hunter-prey-navigation-control-unmanned-aerial-systems.git) drone_lap
cd ~/drone_ws
catkin build  # O catkin_make, dependiendo de tu configuración
source devel/setup.bash
