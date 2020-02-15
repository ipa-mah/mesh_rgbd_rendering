#include <iostream>
#include <json/json.h>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
int main(int argc, char** argv)
{
    if(argc < 7)
    {
        std::cout <<"Usage : ./object_rendering data_path focal_length c_x c_y vertical_views horizontal_views"<<std::endl;
        return EXIT_FAILURE;
    }

    std::string data_path = argv[1];
    Json::Value root;
    root["camera_matrix"]["focal_x"] = std::atof(argv[2]);
    root["camera_matrix"]["focal_y"] =  std::atof(argv[2]);
    root["camera_matrix"]["c_x"] =  std::atof(argv[3]);
    root["camera_matrix"]["c_y"] =  std::atof(argv[4]);
    std::vector<Eigen::Affine3d> poses;
    int vertical_views = std::atoi(argv[5]);
    int horizontal_views = std::atoi(argv[6]);
    double vertical_angle_step = M_PI/vertical_views;
    double horizontal_angle_step = 2*M_PI/horizontal_views;
    double distance = 0.4;
    for (int j=0;j<=horizontal_views;j++)
        for(int i=0;i<=vertical_views;i++)
        {
            int index = j * vertical_views+i;
            Json::Value node ;
            node["id"] = index;
            Eigen::Affine3d cam2world;
            cam2world.setIdentity();
            double vertical_rad = vertical_angle_step * (i);
            double horizontal_rad = horizontal_angle_step * (j);
            double z = distance * sin(vertical_rad);
            double oxy = distance * cos(vertical_rad);
            double x = oxy* sin(horizontal_rad);
            double y = oxy* cos(horizontal_rad);
            cam2world.translation() = Eigen::Vector3d(x,y,z);
            Json::Value trans_vec(Json::arrayValue);
            for (int v=0;v<3;v++) {
                if(std::fabs(cam2world.translation()[v])<0.000001) cam2world.translation()[v] = 0;
                trans_vec.append(cam2world.translation()[v]);
            }
            node["translation"] = trans_vec;
            ////////////////////////////
            Eigen::Vector3d view_dir =  cam2world.translation();
            view_dir.normalized();
            Eigen::Vector3d cam_normal(0,0,-1);
            Eigen::Matrix3d world2cam_transform = Eigen::Quaterniond().
                    setFromTwoVectors(cam_normal,view_dir).toRotationMatrix();
            cam2world.rotate(world2cam_transform);
            poses.push_back(cam2world);

            Json::Value rot_vec(Json::arrayValue);
            for(int v=0;v<9;v++)
            {
                int r = v / 3 ;
                int c = v % 3;
                if(std::fabs(world2cam_transform(r,c))<0.000001) world2cam_transform(r,c) = 0;
                rot_vec.append(world2cam_transform(r,c));
            }
            node["rotation"] = rot_vec;
            root["views"].append(node);
        }

    std::ofstream file_id;
    file_id.open(data_path+"/config.json");
    Json::StyledWriter styled_writer;
    file_id << styled_writer.write(root);
}
