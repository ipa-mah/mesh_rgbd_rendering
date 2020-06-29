#include <pcl/TextureMesh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/texture_mapping.h>

#include <json/json.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <ros/ros.h>
#include <opencv2/opencv.hpp>
void validate(const std::string& data_path, const std::vector<Eigen::Affine3d>& cam2worlds)
{
	for (int view = 0; view < cam2worlds.size(); view++)
	{
		pcl::PointCloud<pcl::PointXYZ> cam_cloud, world_cloud;
		std::ostringstream curr_frame_prefix;
		curr_frame_prefix << std::setw(6) << std::setfill('0') << view;
		pcl::io::loadPLYFile(data_path + "/frame-" + curr_frame_prefix.str() + "cloud.ply", cam_cloud);
		pcl::transformPointCloud(cam_cloud, world_cloud, cam2worlds[view]);
		pcl::io::savePLYFile(data_path + "/frame-" + curr_frame_prefix.str() + "world_cloud.ply", world_cloud);
	}
}

void readPose(std::string& pose_file, std::vector<Eigen::Affine3d>& cam2worlds)
{
	Eigen::Matrix3d rot;
	Eigen::Vector3d trans;
	std::vector<Eigen::Matrix3d> rot_vec;
	std::vector<Eigen::Vector3d> tran_vec;
	std::fstream myfile(pose_file);
	cam2worlds.resize(18);
	for (size_t i = 0; i < 18; i++)
	{
		Eigen::Matrix4d world2cam;
		world2cam.setIdentity();
		for (unsigned int j = 0; j < 9; j++)
		{
			int r = j / 3;
			int c = j % 3;
			myfile >> world2cam(r, c);
		}
		for (int k = 0; k < 3; k++)
		{
			myfile >> world2cam(k, 3);
			world2cam(k, 3) /= 1000.0;
		}
		cam2worlds[i].matrix() = world2cam.inverse();
		std::cout << cam2worlds[i].matrix() << std::endl;
	}
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "camera_pose_visualizer");
	ros::NodeHandle node;
	std::vector<Eigen::Affine3d> poses;
	std::string pose_file = "/home/hah/catkin_ws/src/ros_mesh_rgbd_rendering/data/test.txt";
	readPose(pose_file, poses);
	std::string data_path = "/home/hah/catkin_ws/src/ros_mesh_rgbd_rendering/data/";
	//  if(argc < 2)
	//  {
	//    std::cout <<"Usage : ./object_rendering data_path"<<std::endl;
	//    return EXIT_FAILURE;
	//  }

	// std::string data_path = argv[1];

	Eigen::Matrix3d intrins;
	intrins.setIdentity();
	intrins(0, 0) = 857.8027725219727;
	intrins(1, 1) = 857.8027725219727;
	intrins(0, 2) = 400.0;
	intrins(1, 2) = 400.0;

	/*

	Json::Value root;
	std::ifstream config_doc(data_path+"/config.json", std::ifstream::binary);
	if(!config_doc.is_open())
	{
	  std::cout<<"No config.json file in the data path"<<std::endl;
	  return 0;
	}
	config_doc >> root;

	intrins(0,0) = root["camera_matrix"].get("focal_x",500).asDouble();
	intrins(1,1) = root["camera_matrix"].get("focal_y",500).asDouble();
	intrins(0,2) = root["camera_matrix"].get("c_x",320).asDouble();
	intrins(1,2) = root["camera_matrix"].get("c_y",240).asDouble();

	std::cout<<"camera matrix: "<<intrins<<std::endl;


	int view=0;
	for(const  Json::Value& node : root["views"])
	{
	  Eigen::Affine3d cam2world;
	  cam2world.setIdentity();
	  Eigen::Matrix3d rot;
	  cam2world.translation() = Eigen::Vector3d(node["translation"][0].asDouble(),node["translation"][1].asDouble(),node["translation"][2].asDouble());
	  for(int i=0;i<node["rotation"].size();i++)
	  {
		int r = i / 3 ;
		int c = i % 3 ;
		rot(r,c) = node["rotation"][i].asDouble();
	  }
	  cam2world.rotate(rot);
	  std::cout<<"pose "<<view<<":"<<std::endl<<cam2world.matrix()<<std::endl;
	  poses.push_back(cam2world);
	  view++;
	}
	*/
	std::cout << "Virtual Intrinsics:" << std::endl << intrins << std::endl;
	int image_width = intrins(0, 2) * 2;
	int image_height = intrins(1, 2) * 2;

	//  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ);
	pcl::TextureMesh mesh;
	pcl::io::loadOBJFile(data_path + "/texture_model.obj", mesh);
	pcl::PointCloud<pcl::PointXYZ> cloud;
	pcl::fromPCLPointCloud2(mesh.cloud, cloud);
	std::cout << cloud.points.size() << std::endl;
	// visualization object
	pcl::visualization::PCLVisualizer visu("cameras");
	// read current camera
	double focal_x = intrins(0, 0);
	double focal_y = intrins(1, 1);
	double height = intrins(1, 2) * 2;
	double width = intrins(0, 2) * 2;
	for (int view = 0; view < poses.size(); view++)
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
		p1 = pcl::transformPoint(p1, poses[view].cast<float>());
		p2 = pcl::transformPoint(p2, poses[view].cast<float>());
		p3 = pcl::transformPoint(p3, poses[view].cast<float>());
		p4 = pcl::transformPoint(p4, poses[view].cast<float>());
		p5 = pcl::transformPoint(p5, poses[view].cast<float>());
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
	pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> color_handler(cloud.makeShared(), "z");
	visu.addPointCloud(cloud.makeShared(), color_handler, "cloud");

	// reset camera
	visu.resetCamera();

	// wait for user input
	visu.spin();

	return 0;
}
