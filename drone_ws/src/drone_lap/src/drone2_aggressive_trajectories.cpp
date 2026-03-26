#include "ros/ros.h"
#include "geometry_msgs/PoseStamped.h" 
#include "mavros_msgs/State.h"
#include <cmath>

// -----------------------------------------
// Variables Globales y Configuración
// -----------------------------------------
mavros_msgs::State current_state;
geometry_msgs::PoseStamped current_local_pose; 
bool local_pose_received = false;

/* * SELECCIONA EL MODO:
 * 1: Slalom (Zig-Zag) -> Prueba Roll/Pitch
 * 2: Evasión (Frenada) -> Prueba Kalman
 * 3: Espiral 3D (Helix) -> Prueba Control de Distancia (Z)
 * 4: Cuadrado Agresivo -> Prueba Cambios de Rumbo 90°
 * 5: Figura en 8 (Lemniscata) -> Prueba Seguimiento Curvo Continuo
 */
int MODO_TRAYECTORIA = 3; 

const double Z_BASE = 5.0;
const double START_X = 0.0;
const double START_Y = 0.0;

void state_cb(const mavros_msgs::State::ConstPtr& msg) { current_state = *msg; }
void pose_cb(const geometry_msgs::PoseStamped::ConstPtr& msg) { 
    current_local_pose = *msg; 
    local_pose_received = true; 
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "drone2_test_bench_node");
    ros::NodeHandle nh;

    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>("/drone2/mavros/state", 10, state_cb);
    ros::Subscriber pose_sub = nh.subscribe<geometry_msgs::PoseStamped>("/drone2/mavros/local_position/pose", 10, pose_cb);
    ros::Publisher pos_pub = nh.advertise<geometry_msgs::PoseStamped>("/drone2/mavros/setpoint_position/local", 10);
    
    ros::Rate rate(20.0);

    while (ros::ok() && (!current_state.connected || !local_pose_received)) {
        ros::spinOnce();
        rate.sleep();
    }
    ROS_INFO("Dron Presa listo. Modo Trayectoria: %d", MODO_TRAYECTORIA);

    geometry_msgs::PoseStamped target_pose;
    target_pose.pose.orientation = current_local_pose.pose.orientation;
    
    double t = 0.0;
    double dt = 1.0 / 20.0;

    while (ros::ok()) {
        t += dt;
        target_pose.header.stamp = ros::Time::now();

        switch(MODO_TRAYECTORIA) {
            case 1: // SLALOM
                target_pose.pose.position.x = START_X + 1.5 * t;
                target_pose.pose.position.y = START_Y + 3.0 * sin(0.5 * 2 * M_PI * t);
                target_pose.pose.position.z = Z_BASE;
                break;

            case 2: // EVASIÓN
                if (t < 5.0) target_pose.pose.position.x = START_X + 2.0 * t;
                else if (t < 7.0) { /* Frenada */ }
                else target_pose.pose.position.y += 4.0 * dt;
                target_pose.pose.position.z = Z_BASE;
                break;

            case 3: // ESPIRAL 3D (Helix)
                // El dron gira mientras sube y baja. Prueba el PID de distancia.
                target_pose.pose.position.x = START_X + 4.0 * cos(0.4 * t);
                target_pose.pose.position.y = START_Y + 4.0 * sin(0.4 * t);
                target_pose.pose.position.z = Z_BASE + 2.0 * sin(0.2 * t); 
                ROS_INFO_THROTTLE(1, "Modo Espiral: Probando Z variable");
                break;

            case 4: // CUADRADO AGRESIVO
                // Cambios de 90 grados a alta velocidad para desafiar al Filtro de Kalman
                {
                    double side = 6.0; // metros
                    double speed = 2.0; // m/s
                    double period = (side * 4) / speed;
                    double cycle_t = fmod(t * speed, side * 4);

                    if (cycle_t < side) { // Lado 1
                        target_pose.pose.position.x = START_X + cycle_t;
                        target_pose.pose.position.y = START_Y;
                    } else if (cycle_t < side * 2) { // Lado 2
                        target_pose.pose.position.x = START_X + side;
                        target_pose.pose.position.y = START_Y + (cycle_t - side);
                    } else if (cycle_t < side * 3) { // Lado 3
                        target_pose.pose.position.x = START_X + side - (cycle_t - side * 2);
                        target_pose.pose.position.y = START_Y + side;
                    } else { // Lado 4
                        target_pose.pose.position.x = START_X;
                        target_pose.pose.position.y = START_Y + side - (cycle_t - side * 3);
                    }
                    target_pose.pose.position.z = Z_BASE;
                }
                break;

            case 5: // FIGURA EN 8 (Lemniscata de Bernoulli)
                // Curvas con aceleración variable en ambos ejes
                {
                    double a = 5.0; 
                    double scale = 1 + sin(0.3 * t) * sin(0.3 * t);
                    target_pose.pose.position.x = START_X + (a * cos(0.3 * t)) / scale;
                    target_pose.pose.position.y = START_Y + (a * sin(0.3 * t) * cos(0.3 * t)) / scale;
                    target_pose.pose.position.z = Z_BASE;
                }
                break;
        }

        pos_pub.publish(target_pose);
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}