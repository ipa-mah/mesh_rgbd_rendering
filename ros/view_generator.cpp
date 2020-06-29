#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/surface/texture_mapping.h>
#include <pcl/visualization/cloud_viewer.h>
#include <ros/ros.h>
#include <ros_mesh_rgbd_rendering/ViewGenerator.h>
#include <yaml-cpp/yaml.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iostream>
#include <memory>

class ViewGenerator
{
  public:
	using Ptr = std::shared_ptr<ViewGenerator>;
	ViewGenerator(const ros::NodeHandle& node_handle);
	virtual ~ViewGenerator();
	bool srvViewGeneratorCallBack(ros_mesh_rgbd_rendering::ViewGenerator::Request& req, ros_mesh_rgbd_rendering::ViewGenerator::Response& res);
	bool cameraPosesVisualization(std::vector<Eigen::Affine3f>& cam2worlds, const Eigen::Matrix3f& intrins);

  protected:
	ros::NodeHandle node_handle_;
	ros::ServiceServer view_generator_service_;
};

ViewGenerator::ViewGenerator(const ros::NodeHandle& node_handle) : node_handle_(node_handle)
{
	view_generator_service_ = node_handle_.advertiseService("view_generator_service", &ViewGenerator::srvViewGeneratorCallBack, this);
}
ViewGenerator::~ViewGenerator()
{
}

bool ViewGenerator::srvViewGeneratorCallBack(ros_mesh_rgbd_rendering::ViewGenerator::Request& req, ros_mesh_rgbd_rendering::ViewGenerator::Response& res)
{
	ROS_INFO("Running view generator service");
	std::vector<double> intrinsic_vec{req.focal_length, 0, req.c_x, 0, req.focal_length, req.c_y, 0, 0, 1};
	Eigen::Matrix3f intrin;
	intrin.setIdentity();
	intrin(0, 0) = req.focal_length;
	intrin(1, 1) = req.focal_length;
	intrin(0, 2) = req.c_x;
	intrin(1, 2) = req.c_y;
	std::vector<Eigen::Affine3f> cam2worlds;
	const std::string name_space = "/view_generator";
	node_handle_.setParam(name_space + "/intrinsic", intrinsic_vec);

	const double vertical_angle_step = M_PI / req.vertical_views;
	const double horizontal_angle_step = 2 * M_PI / req.horizontal_views;
	const int num_views = req.vertical_views * req.horizontal_views;
	node_handle_.setParam(name_space + "/views", num_views);
	for (std::size_t i = 0; i < req.vertical_views; i++)
		for (std::size_t j = 0; j < req.horizontal_views; j++)
		{
			std::size_t index = j + i * req.horizontal_views;
			Eigen::Matrix4f extrins;
			extrins.setIdentity();
			double vertical_rad = vertical_angle_step * (i);
			double horizontal_rad = horizontal_angle_step * (j);
			double z = req.distance * sin(vertical_rad);
			double oxy = req.distance * cos(vertical_rad);
			double x = oxy * sin(horizontal_rad);
			double y = oxy * cos(horizontal_rad);

			// std::vector<double> translation{x, y, z};
			std::vector<double> translation{0, 0, req.distance};
			// translation world2cam (world position w.r.t camera coord)
			extrins.topRightCorner(3, 1) = Eigen::Vector3f(0, 0, req.distance);
			Eigen::Vector3d view_dir = Eigen::Vector3d(x, y, z);
			view_dir.normalized();
			Eigen::Vector3d cam_normal(0, 0, -1);
			Eigen::Matrix3d rot = Eigen::Quaterniond().setFromTwoVectors(cam_normal, view_dir).toRotationMatrix();
			extrins.topLeftCorner(3, 3) = rot.cast<float>();
			Eigen::Affine3f extrin_aff(extrins.inverse());
			cam2worlds.push_back(extrin_aff);
			Eigen::Vector3d rpy = rot.eulerAngles(0, 1, 2);
			std::vector<double> vec_rpy{rpy[0], rpy[1], rpy[2]};
			std::ostringstream curr_frame_prefix;
			curr_frame_prefix << std::setw(2) << std::setfill('0') << index;
			std::string position = name_space + "/" + curr_frame_prefix.str() + "_position";
			std::string orientation = name_space + "/" + curr_frame_prefix.str() + "_orientation";
			node_handle_.setParam(position, translation);
			node_handle_.setParam(orientation, vec_rpy);
		}
	ROS_INFO("views are saved into %s", req.data_path.c_str());
	std::string dump_file = "rosparam dump " + req.data_path + "/config.yaml /view_generator";
	system(dump_file.c_str());
	res.success = true;
	ROS_INFO("Successfully create views");
	if (req.visualized)
	{
		cameraPosesVisualization(cam2worlds, intrin);
	}
	return true;
}

bool ViewGenerator::cameraPosesVisualization(std::vector<Eigen::Affine3f>& cam2worlds, const Eigen::Matrix3f& intrins)
{
	pcl::visualization::PCLVisualizer visu("cameras");
	// read current camera
	double focal_x = intrins(0, 0);
	double focal_y = intrins(1, 1);
	double height = intrins(1, 2) * 2;
	double width = intrins(0, 2) * 2;
	for (size_t view = 0; view < cam2worlds.size(); view++)
	{
		// create a 5-point visual for each camera
		pcl::PointXYZ p1, p2, p3, p4, p5;
		p1.x = 0;
		p1.y = 0;
		p1.z = 0;  // origin point
		// double angleX = RAD2DEG (2.0 * atan (width / (2.0*focal)));
		// double angleY = RAD2DEG (2.0 * atan (height / (2.0*focal)));
		double dist = 0.05;
		double minX, minY, maxX, maxY;
		maxX = dist * tan(atan(width / (2.0 * focal_x)));
		minX = -maxX;
		maxY = dist * tan(atan(height / (2.0 * focal_y)));
		minY = -maxY;
		p2.x = minX;
		p2.y = minY;
		p2.z = dist;
		p3.x = maxX;
		p3.y = minY;
		p3.z = dist;
		p4.x = maxX;
		p4.y = maxY;
		p4.z = dist;
		p5.x = minX;
		p5.y = maxY;
		p5.z = dist;
		// Transform points from camera coordinate to world coordinate
		p1 = pcl::transformPoint(p1, cam2worlds[view]);
		p2 = pcl::transformPoint(p2, cam2worlds[view]);
		p3 = pcl::transformPoint(p3, cam2worlds[view]);
		p4 = pcl::transformPoint(p4, cam2worlds[view]);
		p5 = pcl::transformPoint(p5, cam2worlds[view]);
		std::stringstream ss;
		// ss << "Cam " << view;
		visu.addText3D(ss.str(), p1, 0.1, 1.0, 1.0, 1.0, ss.str());
		ss.str("");
		ss << "camera_" << view << "line1";
		visu.addLine(p1, p2, ss.str());
		ss.str("");
		ss << "camera_" << view << "line2";
		visu.addLine(p1, p3, ss.str());
		ss.str("");
		ss << "camera_" << view << "line3";
		visu.addLine(p1, p4, ss.str());
		ss.str("");
		ss << "camera_" << view << "line4";
		visu.addLine(p1, p5, ss.str());
		ss.str("");
		ss << "camera_" << view << "line5";
		visu.addLine(p2, p5, ss.str());
		ss.str("");
		ss << "camera_" << view << "line6";
		visu.addLine(p5, p4, ss.str());
		ss.str("");
		ss << "camera_" << view << "line7";
		visu.addLine(p4, p3, ss.str());
		ss.str("");
		ss << "camera_" << view << "line8";
		visu.addLine(p3, p2, ss.str());
	}
	// add a coordinate system
	visu.addCoordinateSystem(1.0, "global");
	// add the mesh's cloud (colored on Z axis)
	// pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> color_handler (cloud.makeShared(), "z");
	// visu.addPointCloud (cloud.makeShared(), color_handler, "cloud");
	// reset camera
	visu.resetCamera();
	// wait for user input
	visu.spin();
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "view_generator_node");
	ros::NodeHandle node;
	ViewGenerator::Ptr view_generator = std::make_shared<ViewGenerator>(node);
	ros::spin();
	return 1;
}
