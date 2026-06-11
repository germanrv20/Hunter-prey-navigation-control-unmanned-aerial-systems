#include "ros/ros.h"
#include "geometry_msgs/PoseStamped.h" 
#include "mavros_msgs/State.h"
#include <cmath>

// -----------------------------------------
// Variables Globales
// -----------------------------------------
mavros_msgs::State current_state;
geometry_msgs::PoseStamped current_local_pose;
bool local_pose_received = false;

// --- CONFIGURACIÓN DEL MOVIMIENTO ---
const double FIXED_Z = 7.0;       // Altura fija
const double SPEED_Y = 3;       // Velocidad a la que avanza: 1.0 m/s reales
// ------------------------------------

void state_cb(const mavros_msgs::State::ConstPtr& msg) {
    current_state = *msg;
}

void local_pose_cb(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    current_local_pose = *msg;
    local_pose_received = true;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "drone2_distance_test_node");
    ros::NodeHandle nh;

    // Suscriptores y Publicadores
    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>("/drone2/mavros/state", 10, state_cb);
    ros::Subscriber local_pose_sub = nh.subscribe<geometry_msgs::PoseStamped>("/drone2/mavros/local_position/pose", 10, local_pose_cb);
    ros::Publisher pos_pub = nh.advertise<geometry_msgs::PoseStamped>("/drone2/mavros/setpoint_position/local", 10);
    
    ros::Rate rate(20.0); 

    // 1. Esperar conexión y primera lectura de posición
    ROS_INFO("Esperando conexion con FCU de Drone 2 y pose local...");
    while (ros::ok() && (!current_state.connected || !local_pose_received)) {
        ros::spinOnce();
        rate.sleep();
    }
    ROS_INFO("Drone 2 listo. Calculando ruta dinamica...");

    // =========================================================
    // CÁLCULO DINÁMICO DE RUTA
    // Leemos dónde está el dron ahora mismo en el eje Y
    double start_y = current_local_pose.pose.position.y;
    double end_y = start_y + 25.0; // Le sumamos 25 metros a su posición inicial
    // =========================================================

    ROS_INFO("Posicion inicial Y detectada: %.2f m.", start_y);
    ROS_INFO("La meta sera Y: %.2f m. Iniciando a %.1f m/s.", end_y, SPEED_Y);

    geometry_msgs::PoseStamped pose_msg;
    pose_msg.pose.position.x = current_local_pose.pose.position.x; // Mantenemos su X actual también
    geometry_msgs::Quaternion fixed_quat = current_local_pose.pose.orientation;

    // 2. Despegue vertical manteniendo la posición inicial (start_y)
    ROS_INFO("Despegando a Z=%.1fm...", FIXED_Z);
    
    while (ros::ok() && std::abs(current_local_pose.pose.position.z - FIXED_Z) > 0.5) {
        pose_msg.header.stamp = ros::Time::now();
        pose_msg.pose.position.y = start_y;
        pose_msg.pose.position.z = FIXED_Z;
        pose_msg.pose.orientation = fixed_quat;
        
        pos_pub.publish(pose_msg);
        
        ROS_INFO_THROTTLE(2.0, "Ascendiendo... Z actual: %.1fm", current_local_pose.pose.position.z);
        ros::spinOnce();
        rate.sleep();
    }
    ROS_INFO("Altura alcanzada. Empezando a avanzar a %.2f m/s reales.", SPEED_Y);

    // 3. Bucle de Avance Lineal (Con Tiempo Real ROS)
    ros::Time tiempo_anterior = ros::Time::now();
    double y_target = start_y;
    
    // Como siempre le sumamos 25, la dirección siempre será positiva
    double direction = 1.0; 

    while (ros::ok()) {
        // --- CÁLCULO DE TIEMPO REAL ---
        ros::Time tiempo_actual = ros::Time::now();
        double dt = (tiempo_actual - tiempo_anterior).toSec();
        tiempo_anterior = tiempo_actual;

        // Mover y_target progresivamente según la velocidad definida y el TIEMPO REAL
        if (std::abs(y_target - end_y) > 0.01) {
            y_target += direction * SPEED_Y * dt;
            
            // Tope de seguridad
            if (y_target > end_y) {
                y_target = end_y;
            }
        }

        pose_msg.header.stamp = ros::Time::now();
        pose_msg.pose.position.y = y_target; 
        pose_msg.pose.position.z = FIXED_Z;  
        pose_msg.pose.orientation = fixed_quat;

        pos_pub.publish(pose_msg);

        if (y_target == end_y) {
            ROS_INFO_THROTTLE(2.0, "Meta alcanzada. Drone 2 estatico en Y: %.2fm", y_target);
        } else {
            ROS_INFO_THROTTLE(0.5, "Avanzando a 1 m/s reales -> Target Y: %.2fm", y_target);
        }

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}