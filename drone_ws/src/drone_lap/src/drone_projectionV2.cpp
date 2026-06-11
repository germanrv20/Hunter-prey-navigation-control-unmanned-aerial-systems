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
#include <std_msgs/Float64.h>

#include <iostream>
#include <vector>
#include <std_msgs/Int32.h>

// -----------------------------------------
// Variables globales
// -----------------------------------------
// --- CONFIGURACIÓN PRINCIPAL ---
bool use_yolo = true; 
bool use_perfect_lidar = false; 
bool area_mode = false; 
double TARGET_DIST_METERS = 1.5;

// Variables Gazebo (Ground Truth)
geometry_msgs::Pose pose_d1_gz;
geometry_msgs::Pose pose_d2_gz;
bool d1_gz_received = false;
bool d2_gz_received = false;

// Variable estado yolo
int yolo_state = 0;

cv::Scalar color;
std::string label;

struct CameraParams {
    double width;
    double height;
    double hfov;
    bool success;
};
cv::Mat dist_coeffs;
cv::Mat camera_matrix;

// --------------------
// FILTRO EMA GLOBAL ÚNICO (ESQUINAS YOLO)
// --------------------
double ema_x1 = 0.0, ema_y1 = 0.0, ema_x2 = 0.0, ema_y2 = 0.0;
bool ema_box_initialized = false;
const double EMA_ALPHA_GLOBAL = 0.55;

// Variables YOLO (4 Puntos)
double yolo_x1 = 0, yolo_y1 = 0, yolo_x2 = 0, yolo_y2 = 0;
bool yolo_detected = false;

sensor_msgs::Image latest_image;
bool image_received = false;

double ia_distance = -999.0;

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

void yoloCallback(const geometry_msgs::Quaternion::ConstPtr& msg) {
    if (msg->z > msg->x) {
        yolo_x1 = msg->x;
        yolo_y1 = msg->y;
        yolo_x2 = msg->z;
        yolo_y2 = msg->w;
        yolo_detected = true;
    } else {
        yolo_detected = false;
    }
}

void yoloStateCallback(const std_msgs::Int32::ConstPtr& msg) {
    yolo_state = msg->data;
}

void distanceCallback(const std_msgs::Float64::ConstPtr& msg) {
    ia_distance = msg->data;
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

CameraParams leer_datos_camera(const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    CameraParams params = {640.0, 480.0, 1.2, false};
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
    ros::Subscriber image_sub = nh.subscribe("webcam/image_raw", 10, imageCallback); 
    ros::Subscriber yolo_sub = nh.subscribe("/drone1/yolo_pixel_coords", 10, yoloCallback);
    ros::Subscriber state_sub = nh.subscribe("/drone1/yolo_state", 10, yoloStateCallback);
    ros::Subscriber dist_sub = nh.subscribe("/drone1/estimated_distance", 10, distanceCallback);

    ros::Rate rate(20); 

    std::string config_path = ros::package::getPath("drone_lap") + "/config/camera_params.yaml";
    CameraParams datos = leer_datos_camera(config_path);
    cv::Mat dist_coeffs = leer_distorsion_camera(config_path);

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

    while (ros::ok())
    {
        cv::Scalar color_draw = cv::Scalar(0, 0, 255);
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

            cv::Point2d center_pt(cx, cy);
            cv::drawMarker(cv_ptr->image, center_pt, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 20, 2);

            std::vector<cv::Point2f> pts_distorted;
            std::vector<cv::Point2f> pts_undistorted;
            cv::Point2d pt_math_undist(0,0);
            bool gt_visible = false;

            double Z_depth = P_math_cv.z(); 

            // A. DIBUJAR GROUND TRUTH (GAZEBO)
            if (Z_depth > 0) { 
                std::vector<cv::Point3f> pts_3d;
                pts_3d.push_back(cv::Point3f(P_math_cv.x(), P_math_cv.y(), P_math_cv.z()));
                
                cv::projectPoints(pts_3d, cv::Mat::zeros(3,1,CV_64F), cv::Mat::zeros(3,1,CV_64F), camera_matrix, dist_coeffs, pts_distorted);
                cv::undistortPoints(pts_distorted, pts_undistorted, camera_matrix, dist_coeffs, cv::noArray(), camera_matrix);
                
                if (pts_distorted.size() > 0 && pts_undistorted.size() > 0) {
                    pt_math_undist = pts_undistorted[0];
                    cv::Point2d pt_math_dist = pts_distorted[0];

                    if (pt_math_dist.x >= 0 && pt_math_dist.x < datos.width && pt_math_dist.y >= 0 && pt_math_dist.y < datos.height) {
                        cv::circle(cv_ptr->image, pt_math_dist, 3, cv::Scalar(0, 0, 255), 2);
                        gt_visible = true;
                    }
                }
            }

            // B. DIBUJAR YOLO E IA
            cv::Point2d yolo_center(0,0);
            // IMPORTANTE: Declaramos ancho y alto AQUÍ FUERA del if para que todo el programa las vea
            double bbox_width = 0.0;
            double bbox_height = 0.0;

            if (yolo_detected) {
                // 1. FILTRO ÚNICO A LAS 4 ESQUINAS
                if (!ema_box_initialized) {
                    ema_x1 = yolo_x1; ema_y1 = yolo_y1;
                    ema_x2 = yolo_x2; ema_y2 = yolo_y2;
                    ema_box_initialized = true;
                } else {
                    ema_x1 = EMA_ALPHA_GLOBAL * yolo_x1 + (1.0 - EMA_ALPHA_GLOBAL) * ema_x1;
                    ema_y1 = EMA_ALPHA_GLOBAL * yolo_y1 + (1.0 - EMA_ALPHA_GLOBAL) * ema_y1;
                    ema_x2 = EMA_ALPHA_GLOBAL * yolo_x2 + (1.0 - EMA_ALPHA_GLOBAL) * ema_x2;
                    ema_y2 = EMA_ALPHA_GLOBAL * yolo_y2 + (1.0 - EMA_ALPHA_GLOBAL) * ema_y2;
                }

                // 2. ACTUALIZAR VARIABLES FILTRADAS
                yolo_center.x = (ema_x1 + ema_x2) / 2.0;
                yolo_center.y = (ema_y1 + ema_y2) / 2.0;
                bbox_width = ema_x2 - ema_x1;
                bbox_height = ema_y2 - ema_y1;

                // Dibujar
                cv::rectangle(cv_ptr->image, cv::Point(ema_x1, ema_y1), cv::Point(ema_x2, ema_y2), color_draw, 2);
                cv::putText(cv_ptr->image, label_draw, cv::Point(ema_x1, ema_y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color_draw, 2);
                cv::drawMarker(cv_ptr->image, yolo_center, cv::Scalar(0, 255, 0), cv::MARKER_TILTED_CROSS, 10, 2);

                if (gt_visible && pts_distorted.size() > 0) {
                    cv::line(cv_ptr->image, yolo_center, cv::Point((int)pts_distorted[0].x, (int)pts_distorted[0].y), cv::Scalar(0, 255, 255), 1);
                }
            } else {
                // Si YOLO se pierde, mantenemos el último valor de las esquinas filtradas
                yolo_center.x = (ema_x1 + ema_x2) / 2.0;
                yolo_center.y = (ema_y1 + ema_y2) / 2.0;
                bbox_width = ema_x2 - ema_x1;
                bbox_height = ema_y2 - ema_y1;
            }

            // C. CALCULAR EL ERROR FINAL PARA EL DRON
            if (use_yolo) {
                if (yolo_detected) {

                    Eigen::Matrix4d Tm_d1 = poseGazeboToTransform(pose_d1_gz); 
                    Eigen::Matrix4d Tc_d1 = getCameraTransform();             
                    Eigen::Matrix4d c_T_m = Tm_d1 * Tc_d1;                    

                    Eigen::Matrix3d R_cam_world_flu = c_T_m.block<3,3>(0,0);
                    Eigen::Quaterniond q_cam_w(R_cam_world_flu);

                    tf2::Quaternion q_tf(q_cam_w.x(), q_cam_w.y(), q_cam_w.z(), q_cam_w.w());
                    tf2::Matrix3x3 m_tf(q_tf);
                    double roll, pitch, yaw;
                    m_tf.getRPY(roll, pitch, yaw); 

                    Eigen::Quaterniond q_p(Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitX()));
                    Eigen::Quaterniond q_r(Eigen::AngleAxisd(roll,  Eigen::Vector3d::UnitZ()));

                    Eigen::Quaterniond q_comp = q_p * q_r;

                    Eigen::Matrix3d R_comp_mat = q_comp.toRotationMatrix();
                    Eigen::Vector3d rayo_ideal(0.0, 0.0, 1.0);
                    Eigen::Vector3d rayo_compensado = R_comp_mat * rayo_ideal;

                    std::vector<cv::Point3f> pts_3d = { cv::Point3f(rayo_compensado.x(), rayo_compensado.y(), rayo_compensado.z()) };
                    std::vector<cv::Point2f> pts_distorted, pts_undistorted;

                    cv::projectPoints(pts_3d, cv::Mat::zeros(3,1,CV_64F), cv::Mat::zeros(3,1,CV_64F), 
                                      camera_matrix, dist_coeffs, pts_distorted);
                    cv::undistortPoints(pts_distorted, pts_undistorted, camera_matrix, dist_coeffs, cv::noArray(), camera_matrix);

                    cv::Point2d comp_center_undist = pts_undistorted[0]; 
                    cv::Point2d comp_center_viz = pts_distorted[0];     

                    cv::drawMarker(cv_ptr->image, comp_center_viz, cv::Scalar(0, 165, 255), cv::MARKER_CROSS, 20, 2);

                    double half_width = cv_ptr->image.cols / 2.0;
                    double half_height = cv_ptr->image.rows / 2.0;

                    error_x = comp_center_undist.x - yolo_center.x;
                    error_y = comp_center_undist.y - yolo_center.y;

                    error_x = error_x / half_width;
                    error_y = error_y / half_height;
                    
                    cv::line(cv_ptr->image, comp_center_viz, yolo_center, cv::Scalar(0, 165, 255), 2);

                    if(use_perfect_lidar){
                        error_z = Z_depth - TARGET_DIST_METERS;
                        cv::putText(cv_ptr->image, "DIST: LIDAR (Perfecto)", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
                    }else{
                        // ==========================================
                        // MODELO GEOMÉTRICO 
                        // ==========================================
                        if (bbox_width > 0.0 && bbox_height > 0.0) {
                            
                            
                            const double DRONE_REAL_WIDTH = 0.62; 
                            const double DRONE_REAL_HEIGHT = 0.27; 
                            
                            double dist_w = (fx * DRONE_REAL_WIDTH) / bbox_width;
                            double dist_h = (fy * DRONE_REAL_HEIGHT) / bbox_height;
                            
                            double geometric_distance = dist_w;//(dist_w + dist_h) / 2.0;
                            
                            error_z = geometric_distance - TARGET_DIST_METERS;
                            
                            std::string texto_info = cv::format("Geom (W+H): %.2f m | Err Z: %.2f m", geometric_distance, error_z);
                            cv::putText(cv_ptr->image, texto_info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
                            
                            double error_medicion = Z_depth - geometric_distance;
                            ROS_INFO_THROTTLE(0.5, "[Dist W: %.2fm | Dist H: %.2fm] -> Media: %.2fm | Error: %.2fm", 
                                              dist_w, dist_h, geometric_distance, error_medicion);
                            
                        } else {
                            error_z = -999.0;
                        }
                    }
                    cv::line(cv_ptr->image, center_pt, yolo_center, cv::Scalar(0, 255, 0), 2);
                } else {
                    error_x = -999.0;
                    error_y = -999.0;
                    error_z = -999.0;
                    cv::putText(cv_ptr->image, "YOLO LOST", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
                }

            } else {
                if (gt_visible && pts_distorted.size() > 0) {
                    error_x = center_pt.x - pt_math_undist.x;
                    error_y = center_pt.y - pt_math_undist.y;
                    error_z = Z_depth - TARGET_DIST_METERS;
                    cv::line(cv_ptr->image, center_pt, pts_distorted[0], cv::Scalar(0, 0, 255), 2);
                }
            }

            cv::imshow("Drone1 Vision (Red=GT, Green=YOLO)", cv_ptr->image);
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