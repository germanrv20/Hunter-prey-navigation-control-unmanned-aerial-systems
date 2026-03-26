#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "gazebo_msgs/ModelStates.h"
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PointStamped.h> 
#include <geometry_msgs/Quaternion.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cv_bridge/cv_bridge.h>
#include "ros/package.h"
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>    
#include <opencv2/core/eigen.hpp> 
#include <image_transport/image_transport.h>

#include <iostream>
#include <vector>
#include <std_msgs/Int32.h>

// -----------------------------------------
// Variables globales
// -----------------------------------------
// --- CONFIGURACIÓN PRINCIPAL ---
bool use_yolo = true; // true: PID sigue a YOLO, false: PID sigue a Gazebo
bool use_perfect_lidar = false; // true usar yolo perfecto, false: usar bounding box area
double TARGET_DIST_METERS = 1.0;

// Variables Gazebo (Ground Truth)
geometry_msgs::Pose pose_d1_gz;
geometry_msgs::Pose pose_d2_gz;
bool d1_gz_received = false;
bool d2_gz_received = false;

//variable stado yolo
int yolo_state = 0;

cv::Scalar color;
std::string label;

// Variables para almacenar los datos
struct CameraParams {
    double width;
    double height;
    double hfov;
    bool success;
};
cv::Mat dist_coeffs;
cv::Mat camera_matrix;

// --------------------
// FILTRO EMA PARA CENTRO YOLO
// --------------------
double ema_x = 0.0;
double ema_y = 0.0;
bool ema_initialized = false;

const double EMA_ALPHA = 0.55;


// Variables YOLO (4 Puntos)
double yolo_x1 = 0, yolo_y1 = 0, yolo_x2 = 0, yolo_y2 = 0;
bool yolo_detected = false;

sensor_msgs::Image latest_image;
bool image_received = false;

// -----------------------------------------
// Callbacks
// -----------------------------------------
void imageCallback(const sensor_msgs::Image::ConstPtr& msg) {
    latest_image = *msg;
    image_received = true;
}

void modelStatesCallback(const gazebo_msgs::ModelStates::ConstPtr& msg) {
    for (size_t i = 0; i < msg->name.size(); i++) {
        if (msg->name[i] == "drone1") {
            pose_d1_gz = msg->pose[i];
            d1_gz_received = true;
        }
        if (msg->name[i] == "drone2") {
            pose_d2_gz = msg->pose[i];
            d2_gz_received = true;
        }
    }
}

// <---   CALLBACK PARA YOLO ---
void yoloCallback(const geometry_msgs::Quaternion::ConstPtr& msg) {
    // Si x2 es mayor que x1, asumimos que hay detección válida
    if (msg->z > msg->x) {
        yolo_x1 = msg->x; // x1
        yolo_y1 = msg->y; // y1
        yolo_x2 = msg->z; // x2
        yolo_y2 = msg->w; // y2
        yolo_detected = true;
    } else {
        yolo_detected = false;
    }
}

void yoloStateCallback(const std_msgs::Int32::ConstPtr& msg) {
    yolo_state = msg->data;
}

// --- Funciones de transformación ---
Eigen::Matrix4d poseGazeboToTransform(const geometry_msgs::Pose& pose_msg) {
    Eigen::Quaterniond q(pose_msg.orientation.w, pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z);
    Eigen::Matrix3d R = q.toRotationMatrix();
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) << pose_msg.position.x, pose_msg.position.y, pose_msg.position.z;
    return T;
}

Eigen::Matrix4d invertTransform(const Eigen::Matrix4d &T) {
    Eigen::Matrix3d R = T.block<3,3>(0,0);
    Eigen::Vector3d t = T.block<3,1>(0,3);
    Eigen::Matrix4d T_inv = Eigen::Matrix4d::Identity();
    T_inv.block<3,3>(0,0) = R.transpose();
    T_inv.block<3,1>(0,3) = -R.transpose() * t;
    return T_inv;
}

Eigen::Matrix4d getCameraTransform() {
    Eigen::Matrix4d T_c_d1 = Eigen::Matrix4d::Identity();
    T_c_d1(0,3) = 0.0;
    T_c_d1(1,3) = 0.0;
    T_c_d1(2,3) = 0.13;
    return T_c_d1; 
}


// Función que DEVUELVE los datos nominales
CameraParams leer_datos_camera(const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    CameraParams params = {640.0, 480.0, 1.2, false}; // Valores por defecto

    if (fs.isOpened()) {
        fs["width"] >> params.width;
        fs["height"] >> params.height;
        fs["hfov"] >> params.hfov;
        params.success = true;
        fs.release();
    } else {
        ROS_ERROR("No se pudo abrir el archivo de configuracion en: %s", path.c_str());
    }
    return params;
}

// Función que DEVUELVE la matriz de distorsión
cv::Mat leer_distorsion_camera(const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    cv::Mat dist;

    if (fs.isOpened()) {
        fs["distortion_coefficients"] >> dist;
        fs.release();
    } else {
        ROS_WARN("Usando distorsion cero por defecto.");
        dist = cv::Mat::zeros(5, 1, CV_64F);
    }
    return dist;
}
// -----------------------------------------
// Main
// -----------------------------------------
int main(int argc, char** argv)
{
    ros::init(argc, argv, "drone_projection_opencv_node");
    ros::NodeHandle nh;

    ros::Publisher error_pub = nh.advertise<geometry_msgs::PointStamped>("/drone1/vision_error", 1);

    ros::Subscriber model_states_sub = nh.subscribe("/gazebo/model_states", 10, modelStatesCallback);
    ros::Subscriber image_sub = nh.subscribe("webcam/image_raw", 10, imageCallback); ///  /capture_node/camera/image    
    
    // SUSCRIPCIÓN AL NODO DE YOLO
    ros::Subscriber yolo_sub = nh.subscribe("/drone1/yolo_pixel_coords", 10, yoloCallback);
    ros::Subscriber state_sub = nh.subscribe("/drone1/yolo_state", 10, yoloStateCallback);

    ros::Rate rate(20); 

    // --- CARGAR CONFIGURACIÓN DESDE ARCHIVO ---
    std::string config_path = ros::package::getPath("drone_lap") + "/config/camera_params.yaml";
    
    // Llamamos a las funciones que devuelven los datos
    CameraParams datos = leer_datos_camera(config_path);
    cv::Mat dist_coeffs = leer_distorsion_camera(config_path);

    // --- CONFIGURACIÓN CÁMARA ---

    //double width = 640.0;
    //double height = 480.0;
    //double h_fov = 1.2;
    double fx = datos.width / (2.0 * tan(datos.hfov / 2.0));
    double fy = fx;
    double cx = datos.width / 2.0;
    double cy = datos.height / 2.0;

    cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    geometry_msgs::PointStamped error_msg;
    double error_x = 0.0;
    double error_y = 0.0;
    double error_z = 0.0;

    Eigen::Matrix3d R_flu2cv;
    R_flu2cv << 0, -1, 0, 0, 0, -1, 1, 0, 0;

    ROS_INFO("Nodo proyeccion V3 iniciado. Modo YOLO: %s", use_yolo ? "ACTIVADO" : "DESACTIVADO");


    // Utilizado para hacer video
    //int frame_count = 0;
    //std::string folder_path = "/home/germanrv/drone_ws/frames/";

    while (ros::ok())
    {

        cv::Scalar color_draw = cv::Scalar(0, 0, 255); // Rojo por defecto
        std::string label_draw = "BUSCANDO...";

        switch (yolo_state) {
            case 1: color_draw = cv::Scalar(0, 255, 0); label_draw = "YOLO+KALMAN"; break;
            case 2: color_draw = cv::Scalar(0, 255, 255); label_draw = "PREDICCION"; break;
            default: color_draw = cv::Scalar(0, 0, 255); label_draw = "BUSCANDO..."; break;
        }

        ros::spinOnce();

        Eigen::Matrix4d Tm_d1 = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d Tm_d2 = Eigen::Matrix4d::Identity();

        if (d1_gz_received) Tm_d1 = poseGazeboToTransform(pose_d1_gz);
        if (d2_gz_received) Tm_d2 = poseGazeboToTransform(pose_d2_gz);
        
        Eigen::Matrix4d Td1_m = invertTransform(Tm_d1); 
        Eigen::Matrix4d Tc_d1 = getCameraTransform();  
        Eigen::Matrix4d Tc_d2_flu = Tc_d1.inverse() * Td1_m * Tm_d2;

        Eigen::Vector4d P_math_4d(0, 0, 0, 1); 
        Eigen::Vector4d P_math_flu = Tc_d2_flu * P_math_4d;
        Eigen::Vector3d P_math_cv = R_flu2cv * P_math_flu.head<3>();

        error_x = -999.0;
        error_y = -999.0;
        error_z = -999.0;
        error_msg.header.stamp = ros::Time::now();
        error_msg.header.frame_id = "drone1_camera"; 

        if (image_received) {
            cv_bridge::CvImagePtr cv_ptr;
            try {
                cv_ptr = cv_bridge::toCvCopy(latest_image, "bgr8");
            } catch (cv_bridge::Exception& e) {
                ROS_ERROR("cv_bridge exception: %s", e.what());
                continue;
            }


            // Clonar para video
            //cv::Mat img_original = cv_ptr->image.clone();

            cv::Point2d center_pt(cx, cy);
            cv::drawMarker(cv_ptr->image, center_pt, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 20, 2);

            // - DECLARAMOS VARIABLES AQUÍ FUERA ---
            std::vector<cv::Point2f> pts_distorted;     // Para guardar punto GT distorsionado
            std::vector<cv::Point2f> pts_undistorted;   // Para guardar punto GT ideal
            cv::Point2d pt_math_undist(0,0);            // Punto ideal
            bool gt_visible = false;                    // Bandera de visibilidad

            double Z_depth = P_math_cv.z(); 

            // ------------------------------------------
            // A. DIBUJAR GROUND TRUTH (GAZEBO)
            // ------------------------------------------
            if (Z_depth > 0) { 
                std::vector<cv::Point3f> pts_3d;
                pts_3d.push_back(cv::Point3f(P_math_cv.x(), P_math_cv.y(), P_math_cv.z()));
                
                cv::projectPoints(pts_3d, cv::Mat::zeros(3,1,CV_64F), cv::Mat::zeros(3,1,CV_64F), camera_matrix, dist_coeffs, pts_distorted);
                cv::undistortPoints(pts_distorted, pts_undistorted, camera_matrix, dist_coeffs, cv::noArray(), camera_matrix);
                
                if (pts_distorted.size() > 0 && pts_undistorted.size() > 0) {
                    pt_math_undist = pts_undistorted[0];
                    cv::Point2d pt_math_dist = pts_distorted[0];

                    if (pt_math_dist.x >= 0 && pt_math_dist.x < datos.width && pt_math_dist.y >= 0 && pt_math_dist.y < datos.height) {
                        // Pintar círculo Rojo
                        cv::circle(cv_ptr->image, pt_math_dist, 3, cv::Scalar(0, 0, 255), 2);
                        gt_visible = true;
                    }
                }
            }

            // ------------------------------------------
            // B. DIBUJAR YOLO (IA) - USANDO X1, Y1, X2, Y2
            // ------------------------------------------
            cv::Point2d yolo_center(0,0); // Calcularemos el centro para el error
            if (yolo_detected) {
                 // Dibujar caja usando las coordenadas exactas recibidas
                cv::Point p1(yolo_x1, yolo_y1);
                cv::Point p2(yolo_x2, yolo_y2);

                cv::rectangle(cv_ptr->image,p1, p2, color_draw, 2);
                cv::putText(cv_ptr->image, label_draw, cv::Point(yolo_x1, yolo_y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color_draw, 2);

               

                // -----------------------------
                // 1. CENTRO RAW (SIN FILTRAR)
                // -----------------------------
                double raw_center_x = (yolo_x1 + yolo_x2) / 2.0;
                double raw_center_y = (yolo_y1 + yolo_y2) / 2.0;

                // -----------------------------
                // 2. FILTRO EMA
                // -----------------------------
                if (!ema_initialized) {
                    ema_x = raw_center_x;
                    ema_y = raw_center_y;
                    ema_initialized = true;
                } else {
                    ema_x = EMA_ALPHA * raw_center_x + (1.0 - EMA_ALPHA) * ema_x;
                    ema_y = EMA_ALPHA * raw_center_y + (1.0 - EMA_ALPHA) * ema_y;
                }

                // Centro filtrado final
                yolo_center.x = ema_x;
                yolo_center.y = ema_y;

                // Dibujar centro filtrado (VERDE)
                cv::drawMarker(cv_ptr->image, yolo_center,
                               cv::Scalar(0, 255, 0),
                               cv::MARKER_TILTED_CROSS,
                               10, 2);

                // Línea comparativa (Amarilla) entre GT y YOLO
                if (gt_visible && pts_distorted.size() > 0) {
                    cv::line(cv_ptr->image,
                             yolo_center,
                             cv::Point((int)pts_distorted[0].x,
                                       (int)pts_distorted[0].y),
                             cv::Scalar(0, 255, 255),
                             1);
                }
            }
            else {
                // Si no detecta, mantenemos último valor filtrado
                // (esto evita saltos a 0,0)
                yolo_center.x = ema_x;
                yolo_center.y = ema_y;
            }

            // ------------------------------------------
            // C. CALCULAR EL ERROR FINAL PARA EL DRON
            // ------------------------------------------
            if (use_yolo) {
                if (yolo_detected) {

                    // 1. Calcular la matriz de la Cámara respecto al Mundo (c_T_m)
                    Eigen::Matrix4d Tm_d1 = poseGazeboToTransform(pose_d1_gz); // Dron en Mapa
                    Eigen::Matrix4d Tc_d1 = getCameraTransform();             // Cámara en Dron
                    Eigen::Matrix4d c_T_m = Tm_d1 * Tc_d1;                    // Cámara en Mapa (FLU)

                    // 2. Extraer la rotación y obtener los ángulos RPY
                    Eigen::Matrix3d R_cam_world_flu = c_T_m.block<3,3>(0,0);
                    Eigen::Quaterniond q_cam_w(R_cam_world_flu);

                    // Convertimos a tf2 para extraer Roll, Pitch y Yaw fácilmente
                    tf2::Quaternion q_tf(q_cam_w.x(), q_cam_w.y(), q_cam_w.z(), q_cam_w.w());
                    tf2::Matrix3x3 m_tf(q_tf);
                    double roll, pitch, yaw;
                    m_tf.getRPY(roll, pitch, yaw); 

                    // 3. Crear el Cuaternión de Compensación en Ejes de Cámara (OpenCV)
                    // - Pitch sobre el eje X (UnitX): Pitch positivo (nariz arriba) baja el rayo.
                    // - Roll sobre el eje Z (UnitZ): Ladeo de la cámara.
                    Eigen::Quaterniond q_p(Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitX()));
                    Eigen::Quaterniond q_r(Eigen::AngleAxisd(roll,  Eigen::Vector3d::UnitZ()));

                    // Combinamos: la rotación total de compensación
                    Eigen::Quaterniond q_comp = q_p * q_r;

                    // 4. Convertir Cuaternión a Matriz y rotar el Rayo Ideal (0,0,1)
                    Eigen::Matrix3d R_comp_mat = q_comp.toRotationMatrix();
                    Eigen::Vector3d rayo_ideal(0.0, 0.0, 1.0);
                    Eigen::Vector3d rayo_compensado = R_comp_mat * rayo_ideal;

                    // 5. Proyectar el rayo 3D a Píxeles 2D (Teniendo en cuenta distorsión)
                    std::vector<cv::Point3f> pts_3d = { cv::Point3f(rayo_compensado.x(), rayo_compensado.y(), rayo_compensado.z()) };
                    std::vector<cv::Point2f> pts_distorted, pts_undistorted;

                    // Proyección mediante el modelo de la cámara (fx, fy, cx, cy)
                    cv::projectPoints(pts_3d, cv::Mat::zeros(3,1,CV_64F), cv::Mat::zeros(3,1,CV_64F), 
                                      camera_matrix, dist_coeffs, pts_distorted);

                    // Paso a coordenadas ideales (sin distorsión) para el PID
                    cv::undistortPoints(pts_distorted, pts_undistorted, camera_matrix, dist_coeffs, cv::noArray(), camera_matrix);

                    cv::Point2d comp_center_undist = pts_undistorted[0]; // Para el PID
                    cv::Point2d comp_center_viz = pts_distorted[0];     // Para dibujo

                    // 6. Dibujar la Cruceta Naranja Compensada
                    cv::drawMarker(cv_ptr->image, comp_center_viz, cv::Scalar(0, 165, 255), cv::MARKER_CROSS, 20, 2);

                    error_x = comp_center_undist.x - yolo_center.x;
                    error_y = comp_center_undist.y - yolo_center.y;
                    
                    // Línea de control naranja
                    cv::line(cv_ptr->image, comp_center_viz, yolo_center, cv::Scalar(0, 165, 255), 2);

                    if(use_perfect_lidar){
                        error_z = Z_depth - TARGET_DIST_METERS;
                        cv::putText(cv_ptr->image, "DIST: LIDAR (Perfecto)", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
                    }else{
                        double current_area = (yolo_x2 - yolo_x1) * (yolo_y2 - yolo_y1);
                        double TARGET_AREA = 30000.0; // ¡Calibrar este número a 1 metro


                        double current_size = std::sqrt(current_area);
                        double target_size = std::sqrt(TARGET_AREA);


                        error_z = target_size - current_size;
                        
                        // TEXTO MEJORADO
                        std::string texto_area = cv::format("Size: %.0f px | Z_real: %.2f m", current_size, Z_depth);
                        cv::putText(cv_ptr->image, texto_area, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
                    
                    }

                    
                    // Línea verde de control (Centro Imagen -> Centro YOLO)
                    cv::line(cv_ptr->image, center_pt, yolo_center, cv::Scalar(0, 255, 0), 2);
                } else {
                    error_x = -999.0;
                    error_y = -999.0;
                    error_z = -999.0;
                    cv::putText(cv_ptr->image, "YOLO LOST", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
                }



            } else {
                // Modo Gazebo puro
                if (gt_visible && pts_distorted.size() > 0) {
                    error_x = center_pt.x - pt_math_undist.x;
                    error_y = center_pt.y - pt_math_undist.y;
                    error_z = Z_depth - TARGET_DIST_METERS;
                    cv::line(cv_ptr->image, center_pt, pts_distorted[0], cv::Scalar(0, 0, 255), 2);
                }
            }

            cv::imshow("Drone1 Vision (Red=GT, Green=YOLO)", cv_ptr->image);
            /*
            // 2. CONCATENAR HORIZONTALMENTE
            cv::Mat combined;
            cv::hconcat(img_original, cv_ptr->image, combined);

            
            // 3. MOSTRAR Y GUARDAR
            cv::imshow("TFG Demo: Original vs Procesada", combined);
            
            // Guardamos el frame con un nombre secuencial (ej: frame_0001.jpg)
            char filename[100];
            sprintf(filename, "frame_%04d.jpg", frame_count);
            cv::imwrite(folder_path + filename, combined);
            
            frame_count++;
            */
            cv::waitKey(1);
        }

        error_msg.point.x = error_x; 
        error_msg.point.y = error_y;
        error_msg.point.z = error_z;
        error_pub.publish(error_msg);
        
        rate.sleep();
    }

    return 0;
}