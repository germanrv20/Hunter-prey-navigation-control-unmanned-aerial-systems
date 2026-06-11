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
const double CENTER_Z = 7.0;    // Altura media (Despegue)
const double AMPLITUDE = 7.0;   // Oscila +/- 7 metros (Rango: 0m a 14m)
const double PERIOD = 20.0;     // Tarda 20 segundos por ciclo (para que sea suave)
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
    ros::init(argc, argv, "drone2_spin_up_down_node");
    ros::NodeHandle nh;

    // Suscriptores y Publicadores
    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>("/drone2/mavros/state", 10, state_cb);
    ros::Subscriber local_pose_sub = nh.subscribe<geometry_msgs::PoseStamped>("/drone2/mavros/local_position/pose", 10, local_pose_cb);
    ros::Publisher pos_pub = nh.advertise<geometry_msgs::PoseStamped>("/drone2/mavros/setpoint_position/local", 10);
    
    ros::Rate rate(20.0); 

    // 1. Esperar conexión
    ROS_INFO("Esperando conexion con FCU de Drone 2...");
    while (ros::ok() && (!current_state.connected || !local_pose_received)) {
        ros::spinOnce();
        rate.sleep();
    }
    ROS_INFO("Drone 2 listo. Iniciando secuencia de oscilacion vertical.");

    geometry_msgs::PoseStamped pose_msg;
    pose_msg.pose.position.x = 0.0; 
    
    // ==========================================
    // CAMBIO AQUÍ: Posición Y a 3.0 metros
    // ==========================================
    pose_msg.pose.position.y = -7.5; 
    
    // Guardamos la orientación actual para no girar
    geometry_msgs::Quaternion fixed_quat = current_local_pose.pose.orientation;

    // 2. Despegue a la altura central (7.0m) y desplazamiento a Y=3.0m
    ROS_INFO("Despegando a %.1fm y moviendose a Y=3.0m...", CENTER_Z);
    
    // Enviamos setpoints hasta llegar a la altura objetivo
    while (ros::ok() && std::abs(current_local_pose.pose.position.z - CENTER_Z) > 0.5) {
        pose_msg.header.stamp = ros::Time::now();
        pose_msg.pose.position.z = CENTER_Z;
        pose_msg.pose.orientation = fixed_quat;
        
        pos_pub.publish(pose_msg);
        
        ROS_INFO_THROTTLE(2.0, "Ascendiendo... Z actual: %.1fm", current_local_pose.pose.position.z);
        ros::spinOnce();
        rate.sleep();
    }
    ROS_INFO("Altura y posicion alcanzadas. Iniciando oscilacion (+/- %.1fm).", AMPLITUDE);

    // 3. Bucle de Oscilación (Seno)
    double t = 0.0;
    double dt = 1.0 / 20.0;
    double omega = (2.0 * M_PI) / PERIOD; // Frecuencia angular para el periodo definido

    while (ros::ok()) {
        t += dt;
        
        // Calcular nueva Z: Centro + Amplitud * sen(wt)
        double z_target = CENTER_Z + AMPLITUDE * sin(omega * t);

        // Seguridad: Evitar que Z sea negativo (bajo tierra)
        if (z_target < 0.5) z_target = 0.5;

        pose_msg.header.stamp = ros::Time::now();
        pose_msg.pose.position.z = z_target;
        // Ojo: no tocamos pose_msg.pose.position.y, así que se queda en 3.0 para siempre
        pose_msg.pose.orientation = fixed_quat; // Mantener orientación

        pos_pub.publish(pose_msg);

        ROS_INFO_THROTTLE(1.0, "Drone 2 Target Z: %.2fm", z_target);

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}