#include "ros/ros.h"
#include "geometry_msgs/PoseStamped.h" 
#include "mavros_msgs/State.h"
#include "gazebo_msgs/ModelStates.h" 
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ==========================================
//  PARÁMETROS CONFIGURABLES
// ==========================================
int MODO_TRAYECTORIA = 1;      
const double VEL_APROX = 1.5; 
const double Z_DESEADA = 7.0;  

// --- PARÁMETROS MODO 1: SLALOM (Zig-Zag) ---
double VEL_X_SLALOM = 1.5;    
double AMP_Y_SLALOM = 2.0;     
double FREQ_SLALOM  = 0.09;    

// --- PARÁMETROS MODO 2: EVASIÓN ---
double VEL_X_EVASION = 1.8;   
double TIEMPO_FRENADO = 8.0;   
double VEL_Y_ESCAPE  = 1.2;    

// --- PARÁMETROS MODO 3: ESPIRAL ---
double RADIO_ESPIRAL = 4.5;    
double VEL_ANG_ESPIRAL = 0.1;  
double AMP_Z_ESPIRAL = 0.8;    
// ==========================================

struct Pose {
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
};

// Variables Globales
mavros_msgs::State current_state;
geometry_msgs::PoseStamped current_local_pose_msg;
Pose current_global_pose; 

bool local_pose_received = false;
bool global_pose_received = false;
bool centered = false;                       
bool approach_initialized = false;           

Eigen::Vector3d start_local_pos; // Dónde empieza el vuelo de aproximación
Pose local_target_origin;        // Dónde está el (0,0,7) en coordenadas locales

double travel_time = 0.0; // Tiempo total calculado para el viaje
double t_aprox = 0.0;     // Tiempo transcurrido de aproximación

// --- Funciones de Transformación ---
Eigen::Matrix4d pose_to_transform(const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = orientation.toRotationMatrix();
    T.block<3,1>(0,3) = position;
    return T;
}

Pose global_to_local_pose(const Pose& global_ref, const Pose& local_ref, const Pose& global_input) {
    Eigen::Matrix4d T_global_ref = pose_to_transform(global_ref.position, global_ref.orientation);
    Eigen::Matrix4d T_local_ref = pose_to_transform(local_ref.position, local_ref.orientation);
    Eigen::Matrix4d T_global_input = pose_to_transform(global_input.position, global_input.orientation);
    Eigen::Matrix4d T_L_from_G = T_local_ref * T_global_ref.inverse();
    Eigen::Matrix4d T_res = T_L_from_G * T_global_input;
    return { T_res.block<3,1>(0,3), Eigen::Quaterniond(T_res.block<3,3>(0,0)) };
}

void state_cb(const mavros_msgs::State::ConstPtr& msg) { current_state = *msg; }
void local_pose_cb(const geometry_msgs::PoseStamped::ConstPtr& msg) { current_local_pose_msg = *msg; local_pose_received = true; }
void gazebo_cb(const gazebo_msgs::ModelStates::ConstPtr& msg) {
    for(size_t i = 0; i < msg->name.size(); i++) {
        if(msg->name[i] == "drone2") {
            current_global_pose.position << msg->pose[i].position.x, msg->pose[i].position.y, msg->pose[i].position.z;
            current_global_pose.orientation = Eigen::Quaterniond(msg->pose[i].orientation.w, msg->pose[i].orientation.x, msg->pose[i].orientation.y, msg->pose[i].orientation.z);
            global_pose_received = true; break;
        }
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "drone2_test_bench_node");
    ros::NodeHandle nh;
    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>("/drone2/mavros/state", 10, state_cb);
    ros::Subscriber pose_sub = nh.subscribe<geometry_msgs::PoseStamped>("/drone2/mavros/local_position/pose", 10, local_pose_cb);
    ros::Subscriber gazebo_sub = nh.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states", 10, gazebo_cb);
    ros::Publisher pos_pub = nh.advertise<geometry_msgs::PoseStamped>("/drone2/mavros/setpoint_position/local", 10);
    ros::Rate rate(20.0);

    while (ros::ok() && (!current_state.connected || !local_pose_received || !global_pose_received)) {
        ros::spinOnce(); rate.sleep();
    }
    
    geometry_msgs::PoseStamped target_msg;
    double t_adj = 0.0; 
    double dt = 1.0 / 20.0;

    while (ros::ok()) {
        
        // --- PRECAUCIÓN: ESPERAR DESPEGUE MANUAL ---
        if (current_local_pose_msg.pose.position.z < 1.0 && !approach_initialized) {
            ROS_INFO_THROTTLE(2.0, "Esperando despegue manual. Z actual: %.2f", current_local_pose_msg.pose.position.z);
            target_msg.pose.position = current_local_pose_msg.pose.position; 
            target_msg.header.stamp = ros::Time::now();
            target_msg.header.frame_id = "map";
            pos_pub.publish(target_msg);
            ros::spinOnce();
            rate.sleep();
            continue;
        }

        // --- FASE 1: IR AL CENTRO (0,0,7) GLOBAL ---
        if (!centered) {
            if (!approach_initialized) {
                start_local_pos << current_local_pose_msg.pose.position.x, 
                                   current_local_pose_msg.pose.position.y, 
                                   current_local_pose_msg.pose.position.z;
                
                Pose global_ref = current_global_pose;
                Pose local_ref = { start_local_pos, Eigen::Quaterniond(current_local_pose_msg.pose.orientation.w, current_local_pose_msg.pose.orientation.x, current_local_pose_msg.pose.orientation.y, current_local_pose_msg.pose.orientation.z) };
                Pose global_input = { {2.0, 2.0, Z_DESEADA}, global_ref.orientation };
                
                local_target_origin = global_to_local_pose(global_ref, local_ref, global_input);
                
                double dist_total = (local_target_origin.position - start_local_pos).norm();
                travel_time = dist_total / VEL_APROX;
                
                approach_initialized = true;
                ROS_INFO("Iniciando aproximacion: Distancia %.1f m. Tiempo estimado: %.1f s", dist_total, travel_time);
            }

            t_aprox += dt; 

            // 1. Mover el Setpoint (La zanahoria)
            if (t_aprox < travel_time) {
                double alpha = t_aprox / travel_time; 
                Eigen::Vector3d current_setpoint = start_local_pos + alpha * (local_target_origin.position - start_local_pos);
                
                target_msg.pose.position.x = current_setpoint.x();
                target_msg.pose.position.y = current_setpoint.y();
                target_msg.pose.position.z = current_setpoint.z();
                
                ROS_INFO_THROTTLE(1.0, "Aproximando (Setpoint)... %.0f%%", alpha * 100.0);
            } 
            // 2. Esperar a que el dron físico alcance al Setpoint
            else {
                // Mantenemos el setpoint fijo en el centro final
                target_msg.pose.position.x = local_target_origin.position.x();
                target_msg.pose.position.y = local_target_origin.position.y();
                target_msg.pose.position.z = local_target_origin.position.z();

                // Comprobamos la distancia FÍSICA REAL del dron al punto final
                Eigen::Vector3d real_pos(current_local_pose_msg.pose.position.x, 
                                         current_local_pose_msg.pose.position.y, 
                                         current_local_pose_msg.pose.position.z);
                
                double dist_to_target = (local_target_origin.position - real_pos).norm();

                // Si el dron está a menos de 30 centímetros del centro, damos luz verde
                if (dist_to_target < 1.0) { 
                    centered = true;
                    ROS_INFO("¡Centro FÍSICO alcanzado con éxito! Iniciando trayectoria modo %d...", MODO_TRAYECTORIA);
                    ros::Duration(2.0).sleep(); // Pausa de 2 segundos para que se quede completamente quieto antes de empezar
                } else {
                    ROS_INFO_THROTTLE(1.0, "Setpoint finalizado. Esperando que el dron llegue físicamente... Distancia restante: %.2f m", dist_to_target);
                }
            }
        }
        // --- FASE 2: EJECUTAR TRAYECTORIA ---
        else {
            t_adj += dt; 
            switch(MODO_TRAYECTORIA) {
                case 1: // SLALOM
                    target_msg.pose.position.x = local_target_origin.position.x() + VEL_X_SLALOM * t_adj; 
                    target_msg.pose.position.y = local_target_origin.position.y() + AMP_Y_SLALOM * sin(FREQ_SLALOM * 2 * M_PI * t_adj);
                    target_msg.pose.position.z = local_target_origin.position.z();
                    break;
                case 2: // EVASIÓN
                    if (t_adj < TIEMPO_FRENADO) {
                        target_msg.pose.position.x = local_target_origin.position.x() + VEL_X_EVASION * t_adj;
                        target_msg.pose.position.y = local_target_origin.position.y();
                    } else if (t_adj < TIEMPO_FRENADO + 2.0) { 
                        target_msg.pose.position.x = local_target_origin.position.x() + VEL_X_EVASION * TIEMPO_FRENADO; 
                        target_msg.pose.position.y = local_target_origin.position.y();
                    } else {
                        target_msg.pose.position.x = local_target_origin.position.x() + VEL_X_EVASION * TIEMPO_FRENADO;
                        target_msg.pose.position.y = local_target_origin.position.y() + VEL_Y_ESCAPE * (t_adj - (TIEMPO_FRENADO + 2.0));
                    }
                    target_msg.pose.position.z = local_target_origin.position.z();
                    break;
                case 3: // ESPIRAL
                    target_msg.pose.position.x = local_target_origin.position.x() + RADIO_ESPIRAL * cos(VEL_ANG_ESPIRAL * t_adj);
                    target_msg.pose.position.y = local_target_origin.position.y() + RADIO_ESPIRAL * sin(VEL_ANG_ESPIRAL * t_adj);
                    target_msg.pose.position.z = local_target_origin.position.z() + AMP_Z_ESPIRAL * sin(VEL_ANG_ESPIRAL * 0.5 * t_adj);
                    break;
            }
        }

        target_msg.header.stamp = ros::Time::now();
        target_msg.header.frame_id = "map";
        target_msg.pose.orientation = current_local_pose_msg.pose.orientation;
        pos_pub.publish(target_msg);
        ros::spinOnce(); 
        rate.sleep();
    }
    return 0;
}