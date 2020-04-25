#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iostream>
#include <memory>
#include <ros/ros.h>
#include <ros_mesh_rgbd_rendering/ViewGenerator.h>
#include <yaml-cpp/yaml.h>

class ViewGenerator {
public:
  using Ptr = std::shared_ptr<ViewGenerator>;
  ViewGenerator(const ros::NodeHandle &node_handle);
  virtual ~ViewGenerator();
  bool srvViewGeneratorCallBack(
      ros_mesh_rgbd_rendering::ViewGenerator::Request &req,
      ros_mesh_rgbd_rendering::ViewGenerator::Response &res);

protected:
  ros::NodeHandle node_handle_;
  ros::ServiceServer view_generator_service_;
};

ViewGenerator::ViewGenerator(const ros::NodeHandle &node_handle)
    : node_handle_(node_handle) {
  view_generator_service_ = node_handle_.advertiseService(
      "view_generator_service", &ViewGenerator::srvViewGeneratorCallBack, this);
}
ViewGenerator::~ViewGenerator() {}

bool ViewGenerator::srvViewGeneratorCallBack(
    ros_mesh_rgbd_rendering::ViewGenerator::Request &req,
    ros_mesh_rgbd_rendering::ViewGenerator::Response &res) {
  ROS_INFO("Running view generator service");
  std::vector<double> intrinsic{
      req.focal_length, 0, req.c_x, 0, req.focal_length, req.c_y, 0, 0, 1};
  const std::string name_space = "/view_generator";
  node_handle_.setParam(name_space + "/intrinsic", intrinsic);

  const double vertical_angle_step = M_PI / req.vertical_views;
  const double horizontal_angle_step = 2 * M_PI / req.horizontal_views;
  const int num_views = req.vertical_views * req.horizontal_views;
  node_handle_.setParam(name_space + "/views", num_views);
  for (std::size_t i = 0; i < req.vertical_views; i++)
    for (std::size_t j = 0; j < req.horizontal_views; j++) {
      std::size_t index = j + i * req.horizontal_views;

      double vertical_rad = vertical_angle_step * (i);
      double horizontal_rad = horizontal_angle_step * (j);
      double z = req.distance * sin(vertical_rad);
      double oxy = req.distance * cos(vertical_rad);
      double x = oxy * sin(horizontal_rad);
      double y = oxy * cos(horizontal_rad);

      std::vector<double> translation{x, y, z};

      Eigen::Vector3d view_dir = Eigen::Vector3d(x, y, z);
      view_dir.normalized();
      Eigen::Vector3d cam_normal(0, 0, -1);
      Eigen::Vector3d rpy = Eigen::Quaterniond()
                                .setFromTwoVectors(cam_normal, view_dir)
                                .toRotationMatrix()
                                .eulerAngles(0, 1, 2);
      std::vector<double> vec_rpy{rpy[0], rpy[1], rpy[2]};
      std::ostringstream curr_frame_prefix;
      curr_frame_prefix << std::setw(2) << std::setfill('0') << index;
      std::string position =
          name_space + "/" + curr_frame_prefix.str() + "_position";
      std::string orientation =
          name_space + "/" + curr_frame_prefix.str() + "_orientation";
      node_handle_.setParam(position, translation);
      node_handle_.setParam(orientation, vec_rpy);
    }
  ROS_INFO("views are saved into %s", req.data_path.c_str());
  std::string dump_file =
      "rosparam dump " + req.data_path + "/config.yaml /view_generator";
  system(dump_file.c_str());
  res.success = true;
  ROS_INFO("Successfully create views");
  return true;
}
int main(int argc, char **argv) {

  ros::init(argc, argv, "view_generator_node");
  ros::NodeHandle node;
  ViewGenerator::Ptr view_generator = std::make_shared<ViewGenerator>(node);
  ros::spin();
  return 1;
}
