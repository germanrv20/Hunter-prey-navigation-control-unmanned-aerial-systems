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

// --- CONFIGURACIÓN DEL CENTRO ---
// Ajusta esto para que el círculo rodee al Drone 1
const double CENTER_X = 0.0; 
const double CENTER_Y = -10.0; 

// --- PARÁMETROS DE LA ONDA Z (Tu fórmula) ---
const double Z0 = 7.0;           // Altura base (z0)
const double AMP = 1;          // Amplitud (amp) +/- 2m
const double WAVE_FREQ_HZ = 0.1; // Frecuencia de la onda (0.1 Hz = 10s por ciclo)


// --- PARÁMETROS DEL CÍRCULO XY ---
const double RADIUS = 4.0;
const double SPEED_XY = 0.3; // m/s

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
    ros::init(argc, argv, "drone2_wavy_circle_node");
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
    ROS_INFO("Sistema listo. Centro: (%.1f, %.1f)", CENTER_X, CENTER_Y);

    geometry_msgs::PoseStamped pose_msg;
    geometry_msgs::Quaternion fixed_yaw_q = current_local_pose_msg.pose.orientation;

    // 2. FASE DE DESPEGUE (Ir al punto inicial de la trayectoria)
    // Calculamos dónde empieza la trayectoria en t=0
    double start_x = CENTER_X + RADIUS; // cos(0)=1
    double start_y = CENTER_Y;          // sin(0)=0
    double start_z = Z0;                // sin(0)=0

    ROS_INFO("FASE 1: Yendo al punto de inicio (%.1f, %.1f, %.1f)...", start_x, start_y, start_z);
    
    while (ros::ok()) 
    {
        double dx = current_local_pose_msg.pose.position.x - start_x;
        double dy = current_local_pose_msg.pose.position.y - start_y;
        double dz = current_local_pose_msg.pose.position.z - start_z;
        double dist = sqrt(dx*dx + dy*dy + dz*dz);

        if (dist < 0.3) break; // Llegamos

        pose_msg.header.stamp = ros::Time::now();
        pose_msg.pose.position.x = start_x;
        pose_msg.pose.position.y = start_y;
        pose_msg.pose.position.z = start_z;
        pose_msg.pose.orientation = fixed_yaw_q;
        
        pos_pub.publish(pose_msg);
        ros::spinOnce();
        rate.sleep();
    }
    
    ROS_INFO("Posición alcanzada. FASE 2: Iniciando Círculo Ondulado (Seno).");

    // 3. FASE 2: MOVIMIENTO CIRCULAR ONDULADO
    double omega_circle = SPEED_XY / RADIUS; 
    double t = 0.0;
    double dt = 1.0 / 20.0;

    while (ros::ok()) {
        t += dt;
        
        // 1. Calcular posición XY (Círculo)
        double angle_circle = omega_circle * t;
        pose_msg.pose.position.x = CENTER_X + RADIUS * cos(angle_circle);
        pose_msg.pose.position.y = CENTER_Y + RADIUS * sin(angle_circle);
        
        // 2. Calcular posición Z ( fórmula de Onda)
        // z = z0 + amp * sin(2 * PI * freq * t)
        pose_msg.pose.position.z = Z0 + AMP * sin(2 * M_PI * WAVE_FREQ_HZ * t);

        // Orientación
        pose_msg.pose.orientation = fixed_yaw_q; 

        // Publicar
        pose_msg.header.stamp = ros::Time::now(); 
        pos_pub.publish(pose_msg); 

        ROS_INFO_THROTTLE(1.0, 
            "Onda: [x: %.1f, y: %.1f, z: %.1f]",
            pose_msg.pose.position.x,
            pose_msg.pose.position.y,
            pose_msg.pose.position.z);

        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}