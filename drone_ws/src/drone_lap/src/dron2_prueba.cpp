#include "ros/ros.h"
#include "geometry_msgs/PoseStamped.h" 
#include "mavros_msgs/State.h"
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -----------------------------------------
// Variables Globales
// -----------------------------------------
mavros_msgs::State current_state;
geometry_msgs::PoseStamped current_local_pose_msg; 
bool local_pose_received = false;

// Configuración del Círculo
const double Z_TARGET = 7.0; 
const double CENTER_X_FIXED = -0.0; 
const double CENTER_Y_FIXED = -10.0; 
const double RADIUS = 4.0;
const double SPEED = 0.5; // m/s

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
    ros::init(argc, argv, "drone2_orbit_approach_node");
    ros::NodeHandle nh;

    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>("/drone2/mavros/state", 10, state_cb);
    ros::Subscriber local_pose_sub = nh.subscribe<geometry_msgs::PoseStamped>("/drone2/mavros/local_position/pose", 10, local_pose_cb);
    ros::Publisher pos_pub = nh.advertise<geometry_msgs::PoseStamped>("/drone2/mavros/setpoint_position/local", 10);
    
    ros::Rate rate(20.0); // 20 Hz

    // 1. Esperar conexión
    ROS_INFO("Esperando conexión y pose local...");
    while (ros::ok() && (!current_state.connected || !local_pose_received)) {
        ros::spinOnce();
        rate.sleep();
    }
    ROS_INFO("Sistema listo. Centro: (%.1f, %.1f)", CENTER_X_FIXED, CENTER_Y_FIXED);

    geometry_msgs::PoseStamped pose_msg;
    // Mantener orientación inicial fija (o 0.0 si prefieres mirar al norte)
    geometry_msgs::Quaternion fixed_yaw_q = current_local_pose_msg.pose.orientation;

    // ======================================================
    // FASE 1: DESPEGUE (Aproximación a Altura)
    // ======================================================
    ROS_INFO("FASE 1: Despegue a Z=%.1fm...", Z_TARGET);
    
    // Objetivo: Misma X/Y, solo subir Z
    double start_x = current_local_pose_msg.pose.position.x;
    double start_y = current_local_pose_msg.pose.position.y;

    while (ros::ok() && std::abs(current_local_pose_msg.pose.position.z - Z_TARGET) > 0.5) 
    {
        pose_msg.header.stamp = ros::Time::now();
        pose_msg.pose.position.x = start_x;
        pose_msg.pose.position.y = start_y;
        pose_msg.pose.position.z = Z_TARGET;
        pose_msg.pose.orientation = fixed_yaw_q;

        pos_pub.publish(pose_msg);
        
        ros::spinOnce();
        rate.sleep();
    }
    
    // ======================================================
    // FASE 2: APROXIMACIÓN AL INICIO DEL CÍRCULO
    // ======================================================
    // Punto de inicio del círculo (ángulo 0)
    double circle_start_x = CENTER_X_FIXED + RADIUS; 
    double circle_start_y = CENTER_Y_FIXED;          
    
    ROS_INFO("FASE 2: Moviendo al inicio del círculo (%.1f, %.1f)...", circle_start_x, circle_start_y);

    bool target_reached = false;
    while (ros::ok() && !target_reached) 
    {
        // Calcular distancia al objetivo
        double dx = current_local_pose_msg.pose.position.x - circle_start_x;
        double dy = current_local_pose_msg.pose.position.y - circle_start_y;
        double dist = sqrt(dx*dx + dy*dy);

        if (dist < 0.3) { // Tolerancia de 0.5m
            target_reached = true;
        }

        pose_msg.header.stamp = ros::Time::now();
        pose_msg.pose.position.x = circle_start_x;
        pose_msg.pose.position.y = circle_start_y;
        pose_msg.pose.position.z = Z_TARGET;
        pose_msg.pose.orientation = fixed_yaw_q;

        pos_pub.publish(pose_msg);
        
        ROS_INFO_THROTTLE(1.0, "Aproximando... Distancia: %.2fm", dist);
        ros::spinOnce();
        rate.sleep();
    }

    ROS_INFO("Posición alcanzada. FASE 3: Iniciando órbita.");


    // ======================================================
    // FASE 3: MOVIMIENTO CIRCULAR CONTINUO
    // ======================================================
    double omega = SPEED / RADIUS;
    double t = 0.0;
    double dt = 1.0 / 20.0;

    while (ros::ok()) {
        
        t += dt;
        double current_angle = omega * t;
        
        // Cálculo del punto en el círculo
        pose_msg.pose.position.x = CENTER_X_FIXED + RADIUS * cos(current_angle);
        pose_msg.pose.position.y = CENTER_Y_FIXED + RADIUS * sin(current_angle);
        pose_msg.pose.position.z = Z_TARGET;

        pose_msg.pose.orientation = fixed_yaw_q; 

        // Publicar
        pose_msg.header.stamp = ros::Time::now(); 
        pos_pub.publish(pose_msg); 

        ROS_INFO_THROTTLE(1.0, "Orbitando... [x: %.2f, y: %.2f]", pose_msg.pose.position.x, pose_msg.pose.position.y);

        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}