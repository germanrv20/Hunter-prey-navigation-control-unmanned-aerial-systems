# 🦅 Hunter-Prey Navigation & Control for Unmanned Aerial Systems (UAS)

![ROS](https://img.shields.io/badge/ROS-Noetic-green?logo=ros)
![C++](https://img.shields.io/badge/C++-17-blue?logo=c%2B%2B)
![Python](https://img.shields.io/badge/Python-3.8+-yellow?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red?logo=pytorch)
![Gazebo](https://img.shields.io/badge/Gazebo-Simulation-orange)

## 📖 Descripción del Proyecto

Este repositorio contiene el código fuente para un Trabajo de Fin de Grado (TFG) centrado en el desarrollo de un sistema autónomo de navegación y persecución entre dos vehículos aéreos no tripulados (drones). 

El sistema implementa una arquitectura de control de tipo **"Cazador-Presa"** (Hunter-Prey) en un entorno de simulación realista (Gazebo + ArduPilot SITL). El dron "cazador" es capaz de localizar, seguir y mantener una distancia de seguridad con el dron "presa" utilizando exclusivamente **visión monocular** (una sola cámara), prescindiendo de sensores pesados y costosos como los LiDAR.

### ✨ Características Principales
* **Visión Artificial:** Detección en tiempo real del dron objetivo mediante **YOLO**.
* **Estimación de Distancia con Deep Learning:** Una Red Neuronal Perceptrón Multicapa (MLP) entrenada en PyTorch infiere la distancia física en metros basándose en el área del *bounding box* de YOLO.
* **Física y Feature Engineering:** El modelo de IA implementa conceptos de física óptica (inversa de la raíz cuadrada del área) para linealizar la predicción, logrando precisiones centimétricas a largas distancias usando una sola neurona.
* **Arquitectura Híbrida de Alto Rendimiento:** * **Python (Nodos AI):** Encargados de la inferencia pesada (YOLO y PyTorch).
  * **C++ (Nodo de Control):** Encargado de calcular los errores (PID), transformaciones geométricas (OpenCV/tf2) y enviar comandos de vuelo a alta velocidad.

---

## 🏗️ Arquitectura del Sistema (Separation of Concerns)

Para evitar cuellos de botella y problemas de compatibilidad con C++17, el sistema divide las responsabilidades en múltiples nodos de ROS que se comunican a través de *topics*:

1. **`yolo_coordinate_sender.py`**: Procesa la imagen de la cámara, ejecuta YOLO y publica las coordenadas del *bounding box* (Píxeles).
2. **`distance_sender.py`**: Carga el modelo `.pth` de PyTorch, procesa el área de la caja detectada por YOLO y publica la distancia real estimada (Metros).
3. **`drone_projectionV2.cpp`**: Nodo principal en C++. Escucha a la IA, proyecta los vectores 3D de compensación del movimiento de la cámara (Roll/Pitch), calcula el error para el controlador PID y pinta la interfaz visual para depuración.

---

## ⚙️ Requisitos y Dependencias

Este proyecto está diseñado para funcionar sobre **Ubuntu 20.04** y **ROS Noetic**.

* **ROS Noetic** (desktop-full)
* **Gazebo 11**
* **ArduPilot SITL** y MAVROS (para simulación de vuelo)
* **Python 3.x:** `torch`, `numpy`, `scikit-learn`, `matplotlib`, `opencv-python`
* **C++:** OpenCV (`cv_bridge`), Eigen3, tf2

---

## 🚀 Instalación y Uso

### 1. Clonar y Compilar
Clona este repositorio dentro de la carpeta `src` de tu *workspace* de ROS (ej. `~/drone_ws/src/`):

```bash
cd ~/drone_ws/src
git clone [https://github.com/germanrv20/Hunter-prey-navigation-control-unmanned-aerial-systems.git](https://github.com/germanrv20/Hunter-prey-navigation-control-unmanned-aerial-systems.git) drone_lap
cd ~/drone_ws
catkin_make
source devel/setup.bash
