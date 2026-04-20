#include "ros/ros.h"
#include "geometry_msgs/TwistStamped.h" 
#include "geometry_msgs/PointStamped.h" 
#include "mavros_msgs/State.h"
#include <cmath>
#include <algorithm> 

// -----------------------------------------
// Variables Globales
// -----------------------------------------
mavros_msgs::State current_state;
ros::Publisher vel_pub; 

// Constante para detectar pérdida de objetivo
const double PIXEL_SENTINEL = -999.0;

// --- TUNING DEL CONTROLADOR ---

// 1. PID Yaw (Horizontal - Controla giro)
const double Kp_yaw = 0.008;  
const double Ki_yaw = 0.0000;   
const double Kd_yaw = 0.001;
const double MAX_YAW_RATE = 1.0; // rad/s

// 2. PID Altura (Vertical - Controla subida/bajada)
const double Kp_vert = 0.009;   
const double Ki_vert = 0.000;  
const double Kd_vert = 0.0008;   
const double MAX_VERT_VEL = 1.0;  // rad/s


// 3. PID Avance (Distancia - Eje X Frontal)
const double Kp_dist = 0.27; //0.01; // valores sin modelo regresion
const double Ki_dist = 0.000;
const double Kd_dist = 0.014; //0.0014; 
const double MAX_LINEAR_X = 1.8; // m/s



// -----------------------------------------
// Clase PID Genérica
// -----------------------------------------
class PIDController {
public:
    double Kp, Ki, Kd;
    double integral_error;
    double prev_error;
    ros::Time last_time;

    PIDController(double kp, double ki, double kd) : Kp(kp), Ki(ki), Kd(kd), integral_error(0.0), prev_error(0.0) {}

    double compute(double current_error) {
        ros::Time current_time = ros::Time::now();
        
        if (last_time.isZero()) {
            last_time = current_time;
            prev_error = current_error;
            return 0.0; 
        }
        
        double dt = (current_time - last_time).toSec();
        if (dt < 0.001) return 0.0; 

        double proportional = Kp * current_error;
        integral_error += current_error * dt;
        double integral = Ki * integral_error;
        double derivative = Kd * (current_error - prev_error) / dt;

        // Anti-windup simple para evitar acumulación excesiva
        if (std::abs(current_error) < 2.0) integral_error = 0.0;

        prev_error = current_error;
        last_time = current_time;

        return proportional + integral + derivative;
    }
    
    // Reinicia el PID si se pierde el objetivo
    void reset() {
        integral_error = 0.0;
        prev_error = 0.0;
        last_time = ros::Time(0);
    }
};

// Instancias de los controladores (Globales para acceso en callback)
PIDController yaw_pid(Kp_yaw, Ki_yaw, Kd_yaw);
PIDController vert_pid(Kp_vert, Ki_vert, Kd_vert);
PIDController dist_pid(Kp_dist, Ki_dist, Kd_dist);

// -----------------------------------------
// Callbacks
// -----------------------------------------

void state_cb(const mavros_msgs::State::ConstPtr& msg) {
    current_state = *msg;
}

// Callback ÚNICO que recibe ambos errores (X e Y) y genera el comando
void error_cb(const geometry_msgs::PointStamped::ConstPtr& msg) {
    // Extraemos errores
    double err_x = msg->point.x; // Error Horizontal
    double err_y = msg->point.y; // Error Vertical
    double err_dist = msg->point.z; //Error Distancia



    // Failsafe: Si el dron no está listo, no hacemos nada
    if (!current_state.connected || !current_state.armed || (current_state.mode != "GUIDED" && current_state.mode != "POSITION")) {
        return;
    }

    double cmd_yaw = 0.0;
    double cmd_z = 0.0;
    double cmd_x = 0.0;

    // Comprobamos si el objetivo es visible
    // (Usamos err_x > -999.0 como indicador de que el dron es visible)
    if (err_x > PIXEL_SENTINEL + 1.0) {


        //  SISTEMA DE PRIORIDAD DINÁMICA (Distancia vs Altura)
        //double dist_ratio = std::min(std::abs(err_dist) / 100.0, 1.0); 
        //double alt_multiplier = 1.0 - (dist_ratio * 0.8);
        //double current_max_z = MAX_VERT_VEL * alt_multiplier;

        cmd_yaw = yaw_pid.compute(err_x);
        cmd_z   = vert_pid.compute(err_y) ; //* alt_multiplier;
        cmd_x   = dist_pid.compute(err_dist);

  
        // ZONAS MUERTAS APLICADAS A LOS MOTORES 
        if (std::abs(err_x) < 5.0)  cmd_yaw = 0.0;
        if (std::abs(err_y) < 5.0) cmd_z   = 0.0;
        //if (std::abs(err_dist) < 10.0) cmd_x = 0.0;


        //   LÍMITES DE SEGURIDAD
        cmd_yaw = std::min(std::max(cmd_yaw, -MAX_YAW_RATE), MAX_YAW_RATE);
        cmd_z   = std::min(std::max(cmd_z, -MAX_VERT_VEL), MAX_VERT_VEL);
        cmd_x   = std::min(std::max(cmd_x, -MAX_LINEAR_X), MAX_LINEAR_X);

    } else {
        // Objetivo Perdido: Parada de emergencia 
        cmd_yaw = 0.0;
        cmd_z = 0.0; 
        cmd_x = 0.0;
        
        //Resetear PIDs para evitar saltos cuando se recupere
        yaw_pid.reset(); 
        vert_pid.reset();
        dist_pid.reset();
    }

    // --- 3. PUBLICAR COMANDO COMBINADO ---
    geometry_msgs::TwistStamped twist_msg; 
    twist_msg.header.stamp = ros::Time::now(); 
    twist_msg.header.frame_id = "base_link"; 
    
    twist_msg.twist.linear.x = cmd_x; // Control velocidad (PID 3)
    twist_msg.twist.linear.y = 0.0;
    twist_msg.twist.linear.z = cmd_z;   // Control Vertical (PID 2)
    
    twist_msg.twist.angular.x = 0.0;
    twist_msg.twist.angular.y = 0.0;
    twist_msg.twist.angular.z = cmd_yaw; // Control de Giro (PID 1)

    vel_pub.publish(twist_msg); 

    // Log de depuración
    if (err_x > PIXEL_SENTINEL + 1.0) {
        ROS_INFO_THROTTLE(0.5, "Err(X,Y,Dist): (%.1f, %.1f, %.1f) -> Cmd(Yaw,Z,X): (%.3f rad/s, %.2f m/s, %.2f m/s)", 
                          err_x, err_y, err_dist, cmd_yaw, cmd_z, cmd_x);
    } else {
        ROS_WARN_THROTTLE(2.0, "Objetivo Perdido. Esperando...");
    }
}

// -----------------------------------------
// Main
// -----------------------------------------

int main(int argc, char** argv)
{
    ros::init(argc, argv, "drone1_follow_xy_pid");
    ros::NodeHandle nh;

    // Inicializamos publicador
    vel_pub = nh.advertise<geometry_msgs::TwistStamped>("/drone1/mavros/setpoint_velocity/cmd_vel", 10);

    // Suscripciones
    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>("/drone1/mavros/state", 10, state_cb);
    
    ros::Subscriber error_sub = nh.subscribe("/drone1/vision_error", 1, error_cb, ros::TransportHints().tcpNoDelay()); 

    ROS_INFO("Nodo Seguidor Dual (Yaw + Altura) Iniciado.");
    
    ros::spin();

    return 0;
}