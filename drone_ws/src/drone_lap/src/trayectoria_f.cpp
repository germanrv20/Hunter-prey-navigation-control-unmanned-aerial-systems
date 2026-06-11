#include "ros/ros.h"
#include "geometry_msgs/PoseStamped.h" 
#include "mavros_msgs/State.h"
#include <cmath>
#include <vector>

// -----------------------------------------
// Variables Globales
// -----------------------------------------
mavros_msgs::State current_state;
geometry_msgs::PoseStamped current_local_pose_msg; 
bool local_pose_received = false;

// Estructura simple para definir los puntos de la trayectoria
struct Waypoint {
    double x;
    double y;
    double z;
};

void state_cb(const mavros_msgs::State::ConstPtr& msg) {
    current_state = *msg;
}

void local_pose_cb(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    current_local_pose_msg = *msg;
    local_pose_received = true;
}

// -----------------------------------------
// Main
// -----------------------------------------

int main(int argc, char** argv)
{
    ros::init(argc, argv, "drone2_waypoints_node");
    ros::NodeHandle nh;

    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>("/drone2/mavros/state", 10, state_cb);
    ros::Subscriber local_pose_sub = nh.subscribe<geometry_msgs::PoseStamped>("/drone2/mavros/local_position/pose", 10, local_pose_cb);
    ros::Publisher pos_pub = nh.advertise<geometry_msgs::PoseStamped>("/drone2/mavros/setpoint_position/local", 10);
    
    ros::Rate rate(20.0); // 20 Hz

    // 1. Esperar conexión con el simulador
    ROS_INFO("Esperando conexión y pose local del Dron 2...");
    while (ros::ok() && (!current_state.connected || !local_pose_received)) {
        ros::spinOnce();
        rate.sleep();
    }
    ROS_INFO("Sistema listo. Cargando Plan de Vuelo (Waypoints)...");

    geometry_msgs::PoseStamped pose_msg;
    geometry_msgs::Quaternion fixed_yaw_q = current_local_pose_msg.pose.orientation;

    // -------------------------------------------------------------
    // 2. DEFINICIÓN DE LA TRAYECTORIA (TEST DE ESTRÉS PARA DRON 1)
    // -------------------------------------------------------------
    std::vector<Waypoint> flight_plan = {
        {5.0,  -30.0, 8.0},  // WP 0: 
        {15.0, -20.0, 5.0},  // WP 1: 
        {35.0,  -25.0,  12},  // WP 2: 
        {40.0, 20.0,   9.0},  // WP 3: 
        {25.0,  25.0,  15.0},  // WP 4: 
        {35.0,  80.0,  15.0}   // WP 5: 
    };

    // Distancia de tolerancia para dar un waypoint por alcanzado
    const double TOLERANCE = 1.2; // metros

    // 3. EJECUCIÓN DE LA TRAYECTORIA
    for (size_t i = 0; i < flight_plan.size(); ++i) {
        
        Waypoint target = flight_plan[i];
        ROS_INFO("==> Viajando al Waypoint %zu: [X: %.1f, Y: %.1f, Z: %.1f]", i, target.x, target.y, target.z);

        while (ros::ok()) {
            // Calcular distancia actual al waypoint objetivo
            double dx = current_local_pose_msg.pose.position.x - target.x;
            double dy = current_local_pose_msg.pose.position.y - target.y;
            double dz = current_local_pose_msg.pose.position.z - target.z;
            double dist = std::sqrt(dx*dx + dy*dy + dz*dz);

            // Si estamos lo suficientemente cerca, rompemos el bucle para pasar al siguiente WP
            if (dist < TOLERANCE) {
                ROS_INFO("Waypoint %zu alcanzado.", i);
                break; 
            }

            // Publicar el setpoint constantemente
            pose_msg.header.stamp = ros::Time::now();
            pose_msg.pose.position.x = target.x;
            pose_msg.pose.position.y = target.y;
            pose_msg.pose.position.z = target.z;
            pose_msg.pose.orientation = fixed_yaw_q; // Mantener siempre la misma orientación frontal
            
            pos_pub.publish(pose_msg);
            
            ros::spinOnce();
            rate.sleep();
        }
    }

    // 4. MODO HOVER AL FINALIZAR
    ROS_INFO("¡TRAYECTORIA COMPLETADA! Manteniendo posición final.");
    Waypoint final_wp = flight_plan.back();
    
    while (ros::ok()) {
        pose_msg.header.stamp = ros::Time::now();
        pose_msg.pose.position.x = final_wp.x;
        pose_msg.pose.position.y = final_wp.y;
        pose_msg.pose.position.z = final_wp.z;
        pose_msg.pose.orientation = fixed_yaw_q;
        
        pos_pub.publish(pose_msg);
        
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
