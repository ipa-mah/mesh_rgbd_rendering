#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <ros_mesh_rgbd_rendering/triangle_mesh.h>
#include <ros_object_detection_msgs/MeshRGBDRendering.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include <yaml-cpp/yaml.h>
#include <unordered_map>

struct VertFace
{
  VertFace() : vertex_id(), face_id()
  {
  }
  int vertex_id;
  int face_id;
};

class MeshDepthRecover
{
public:
  using Ptr = std::shared_ptr<MeshDepthRecover>;
  MeshDepthRecover(const ros::NodeHandle& node_handle);
  virtual ~MeshDepthRecover();

  bool srvMeshDepthRecoverCallBack(ros_object_detection_msgs::MeshRGBDRendering::Request& req, ros_object_detection_msgs::MeshRGBDRendering::Response& res);

  bool readData(const std::string& data_path);
  bool depthRendering(const std::vector<Eigen::Matrix4d>& extrinsics, std::vector<cv::Mat>& depth_images);
  bool isInside(const Eigen::Vector2d& p, const Eigen::Vector2d& v0, const Eigen::Vector2d& v1, const Eigen::Vector2d& v2, Eigen::Vector3d& bcoords);
  bool checkPointInsideTriangle(const pcl::PointXY& p1, const pcl::PointXY& p2, const pcl::PointXY& p3, const pcl::PointXY& pt);
  bool getPixelCoords4GlobalPt(const int height, const int width, const Eigen::Matrix3d& intrins, Eigen::Vector2d& pixel, const Eigen::Matrix4d& world_to_cam, const Eigen::Vector3d& global_pt);
  bool faceProjected(const int height, const int width, const Eigen::Matrix3d& intrins, const Eigen::Matrix4d& world_to_cam, const Eigen::Vector3d& global_pt0, const Eigen::Vector3d& global_pt1,
                     const Eigen::Vector3d& global_pt2, Eigen::Vector2d& pixel0, Eigen::Vector2d& pixel1, Eigen::Vector2d& pixel2);
  Eigen::Vector3d globalToCameraSpace(const Eigen::Vector3d& pt, const Eigen::Matrix4d& world_to_cam);
  Eigen::Vector2d cameraToImgSpace(const Eigen::Vector3d& pt, const Eigen::Matrix3d& intrins);
  void getTriangleCircumscribedCircleCentroid(const pcl::PointXY& p1, const pcl::PointXY& p2, const pcl::PointXY& p3, pcl::PointXY& circumcenter, double& radius);

public:
protected:
protected:
  ros::NodeHandle node_handle_;
  ros::ServiceServer mesh_depth_recover_service_;
  Eigen::Matrix3d intrins_;
  int image_width_;
  int image_height_;

protected:
  bool readTriangleMesh(const std::string& mesh_file);
  TriangleMesh::Ptr triangle_mesh_;
};

MeshDepthRecover::MeshDepthRecover(const ros::NodeHandle& node_handle) : node_handle_(node_handle)
{
  mesh_depth_recover_service_ = node_handle_.advertiseService("mesh_depth_recover_service", &MeshDepthRecover::srvMeshDepthRecoverCallBack, this);
}
MeshDepthRecover::~MeshDepthRecover()
{
}
bool MeshDepthRecover::readTriangleMesh(const std::string& mesh_file)
{
  triangle_mesh_ = std::make_shared<TriangleMesh>();
  if (!readTextureMeshfromOBJFile(mesh_file, triangle_mesh_))
  {
    std::cout << " MeshDepthRecover::readTriangleMesh is not successful" << std::endl;
    return false;
  };
}
bool MeshDepthRecover::depthRendering(const std::vector<Eigen::Matrix4d>& extrinsics, std::vector<cv::Mat>& depth_images)
{
  depth_images.resize(extrinsics.size());
  pcl::PointCloud<pcl::PointXYZ>::Ptr mesh_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cam_cloud(new pcl::PointCloud<pcl::PointXYZ>);

  for (int idx_vertex = 0; idx_vertex < triangle_mesh_->vertices_.size(); idx_vertex++)
  {
    Eigen::Vector3d vertex = triangle_mesh_->vertices_[idx_vertex];
    pcl::PointXYZ p(vertex(0), vertex(1), vertex(2));
    mesh_cloud->points.push_back(p);
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::size_t view = 0; view < extrinsics.size(); view++)
  {
    pcl::transformPointCloud(*mesh_cloud, *cam_cloud, extrinsics[view].cast<float>());
    std::vector<bool> visible_faces;
    visible_faces.resize(triangle_mesh_->triangles_.size());
    pcl::PointCloud<pcl::PointXY>::Ptr projections(new pcl::PointCloud<pcl::PointXY>);
    std::vector<VertFace> uv_indexes;
    pcl::PointXY nan_point;
    nan_point.x = std::numeric_limits<float>::quiet_NaN();
    nan_point.y = std::numeric_limits<float>::quiet_NaN();
    VertFace u_null;
    u_null.vertex_id = -1;
    u_null.face_id = -1;

    int cpt_invisible = 0;
    for (std::size_t idx_face = 0; idx_face < triangle_mesh_->triangles_.size(); ++idx_face)
    {
      Eigen::Vector2d uv_coord0;
      Eigen::Vector2d uv_coord1;
      Eigen::Vector2d uv_coord2;
      pcl::PointXYZ pt0 = mesh_cloud->points[triangle_mesh_->triangles_[idx_face][0]];
      pcl::PointXYZ pt1 = mesh_cloud->points[triangle_mesh_->triangles_[idx_face][1]];
      pcl::PointXYZ pt2 = mesh_cloud->points[triangle_mesh_->triangles_[idx_face][2]];
      Eigen::Vector3d global_pt0(pt0.x, pt0.y, pt0.z);
      Eigen::Vector3d global_pt1(pt1.x, pt1.y, pt1.z);
      Eigen::Vector3d global_pt2(pt2.x, pt2.y, pt2.z);
      // project each vertice, if one is out of view, stop
      if (faceProjected(image_height_, image_width_, intrins_, extrinsics[view], global_pt0, global_pt1, global_pt2, uv_coord0, uv_coord1, uv_coord2))
      {
#ifdef _OPENMP
#pragma omp critical
#endif
        {
          // add UV coordinates
          pcl::PointXY uv0, uv1, uv2;
          uv0.x = static_cast<float>(uv_coord0(0));
          uv0.y = static_cast<float>(uv_coord0(1));
          uv1.x = static_cast<float>(uv_coord1(0));
          uv1.y = static_cast<float>(uv_coord1(1));
          uv2.x = static_cast<float>(uv_coord2(0));
          uv2.y = static_cast<float>(uv_coord2(1));
          projections->points.push_back(uv0);
          projections->points.push_back(uv1);
          projections->points.push_back(uv2);
        }
        VertFace u1, u2, u3;
        u1.vertex_id = triangle_mesh_->triangles_[idx_face][0];
        u2.vertex_id = triangle_mesh_->triangles_[idx_face][1];
        u3.vertex_id = triangle_mesh_->triangles_[idx_face][2];
        u1.face_id = idx_face;
        u2.face_id = idx_face;
        u3.face_id = idx_face;
#ifdef _OPENMP
#pragma omp critical
#endif
        {
          uv_indexes.push_back(u1);
          uv_indexes.push_back(u2);
          uv_indexes.push_back(u3);
        }

        visible_faces[idx_face] = true;
      }
      else
      {
#ifdef _OPENMP
#pragma omp critical
#endif
        {
          projections->points.push_back(nan_point);
          projections->points.push_back(nan_point);
          projections->points.push_back(nan_point);
          uv_indexes.push_back(u_null);
          uv_indexes.push_back(u_null);
          uv_indexes.push_back(u_null);
        }
        // keep track of visibility
        visible_faces[idx_face] = false;
        cpt_invisible++;
      }
    }

    // TODO handle case were no face could be projected
    if (triangle_mesh_->triangles_.size() - cpt_invisible != 0)
    {
      // create kdtree
      pcl::KdTreeFLANN<pcl::PointXY> kdtree;
      kdtree.setInputCloud(projections);
      std::vector<int> idxNeighbors;
      std::vector<float> neighborsSquaredDistance;

      for (int idx_face = 0; idx_face < static_cast<int>(triangle_mesh_->triangles_.size()); ++idx_face)
      {
        if (!visible_faces[idx_face])
        {
          continue;
        }
        pcl::PointXY uv_coord1;
        pcl::PointXY uv_coord2;
        pcl::PointXY uv_coord3;
        // face is in the camera's FOV
        uv_coord1 = projections->points[idx_face * 3 + 0];
        uv_coord2 = projections->points[idx_face * 3 + 1];
        uv_coord3 = projections->points[idx_face * 3 + 2];

        double radius;
        pcl::PointXY center;
        getTriangleCircumscribedCircleCentroid(uv_coord1, uv_coord2, uv_coord3, center, radius);  // this function yields faster results than getTriangleCircumcenterAndSize
        if (kdtree.radiusSearch(center, radius, idxNeighbors, neighborsSquaredDistance) > 0)
        {
          for (size_t i = 0; i < idxNeighbors.size(); ++i)
          {
            if (std::max(cam_cloud->points[triangle_mesh_->triangles_[idx_face][0]].z,
                         std::max(cam_cloud->points[triangle_mesh_->triangles_[idx_face][1]].z, cam_cloud->points[triangle_mesh_->triangles_[idx_face][2]].z)) <
                cam_cloud->points[uv_indexes[idxNeighbors[i]].vertex_id].z)
            {
              if (checkPointInsideTriangle(uv_coord1, uv_coord2, uv_coord3, projections->points[idxNeighbors[i]]))
              {
                visible_faces[uv_indexes[idxNeighbors[i]].face_id] = false;
              }
            }
          }
        }
      }
    }
    // printf("%f percent faces are visible in camera \n",(float)std::count(visible_faces.begin(),
    //                                                                     visible_faces.end(),true)/visible_faces.size());

    cv::Mat depth(image_height_, image_width_, CV_16UC1);
    cv::Mat mask(image_height_, image_width_, CV_8UC1);
    cv::Mat vertex_map(image_height_, image_width_, CV_32FC3);

    depth = 0;
    // mask = 0;
    vertex_map = 0;
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.width = depth.cols;
    cloud.height = depth.rows;
    cloud.is_dense = true;
    cloud.points.resize(depth.cols * depth.rows, pcl::PointXYZ(0, 0, 0));
    for (std::size_t idx_face = 0; idx_face < triangle_mesh_->triangles_.size(); idx_face++)
    {
      if (visible_faces[idx_face])
      {
        // vertices
        Eigen::Vector2d pixel0, pixel1, pixel2;

        pcl::PointXYZ pt0 = mesh_cloud->points[triangle_mesh_->triangles_[idx_face][0]];
        pcl::PointXYZ pt1 = mesh_cloud->points[triangle_mesh_->triangles_[idx_face][1]];
        pcl::PointXYZ pt2 = mesh_cloud->points[triangle_mesh_->triangles_[idx_face][2]];

        Eigen::Vector3d global_p0(pt0.x, pt0.y, pt0.z);
        Eigen::Vector3d global_p1(pt1.x, pt1.y, pt1.z);
        Eigen::Vector3d global_p2(pt2.x, pt2.y, pt2.z);

        if (faceProjected(image_height_, image_width_, intrins_, extrinsics[view], global_p0, global_p1, global_p2, pixel0, pixel1, pixel2))
        {
          Eigen::AlignedBox2d box;
          box.extend(pixel0);
          box.extend(pixel1);
          box.extend(pixel2);

          Eigen::Vector2d max_corner_pixel = box.max(), min_corner_pixel = box.min();
          int min_y = static_cast<int>(floor(min_corner_pixel[1]));  // Note +y is to the bottom (UV COORD- original is bottom-left)
          int max_y = static_cast<int>(ceil(max_corner_pixel[1]));
          max_y = std::min(max_y, image_height_ - 1);
          int min_x = static_cast<int>(floor(min_corner_pixel[0]));
          int max_x = static_cast<int>(ceil(max_corner_pixel[0]));
          max_x = std::min(max_x, image_width_ - 1);
          for (int x = min_x; x <= max_x; x++)
          {
            for (int y = min_y; y <= max_y; y++)
            {
              Eigen::Vector2d uv(x, y);
              Eigen::Vector3d bcoords;
              if (isInside(uv, pixel0, pixel1, pixel2, bcoords))
              {
                Eigen::Vector3d global_pt_uv = bcoords[0] * global_p0 + bcoords[1] * global_p1 + bcoords[2] * global_p2;
                Eigen::Vector3d cam_pt_uv;
                cam_pt_uv = globalToCameraSpace(global_pt_uv, extrinsics[view]);
                depth.at<unsigned short>(y, x) = static_cast<unsigned short>(cam_pt_uv[2] * 1000);
                vertex_map.at<cv::Vec3f>(y, x) = cv::Vec3f(cam_pt_uv[0], cam_pt_uv[1], cam_pt_uv[2]);

                // mask.at<uchar>(y,x) = 255;
                pcl::PointXYZ pt;
                pt.x = static_cast<float>(cam_pt_uv[0]);
                pt.y = static_cast<float>(cam_pt_uv[1]);
                pt.z = static_cast<float>(cam_pt_uv[2]);
                size_t index = y * cloud.width + x;
                cloud.points[index] = pt;
              }
            }
          }
        }
      }
    }

    depth_images[view] = depth;
    // std::ostringstream curr_frame_prefix;
    // curr_frame_prefix<<view;
    // curr_frame_prefix << std::setw(6) << std::setfill('0') << view;
    // cv::imwrite(object_data_path_+"/frame-"+curr_frame_prefix.str()+".depth.png",depth);
    // cv::imwrite(data_path_+"/frame-"+curr_frame_prefix.str()+".mask.png",mask);
    // pcl::io::savePLYFile(object_data_path_+"/frame-"+curr_frame_prefix.str()+".cloud.ply",cloud);
    // saveVertexMap(vertex_map,object_data_path_+object_name_+"_"+curr_frame_prefix.str()+"_coloredPC_xyz_32F3.bin");
  }
  return true;
}

bool MeshDepthRecover::srvMeshDepthRecoverCallBack(ros_object_detection_msgs::MeshRGBDRendering::Request& req, ros_object_detection_msgs::MeshRGBDRendering::Response& res)
{

  std::vector<Eigen::Matrix4d> extrinsics;
  for (const auto& extrin : req.extrinsics) {
    Eigen::Quaterniond quar(extrin.rotation.x,
                            extrin.rotation.z,
                            extrin.rotation.y,
                            extrin.rotation.w);
    Eigen::Vector3d trans(extrin.translation.x,
                          extrin.translation.y,
                          extrin.translation.z);
    Eigen::Matrix4d pose;
    pose.setIdentity();
    pose.topRightCorner(3,1) = trans;
    Eigen::Matrix3d rot;
    rot = quar.toRotationMatrix();
    pose.topLeftCorner(3,3) = rot;
    extrinsics.push_back(pose);
  }
  if(!readTriangleMesh(req.mesh_file))
  {
    res.success = false;
    return false;
  };
  std::vector<cv::Mat> depth_images;
  depthRendering(extrinsics,depth_images);
  for (size_t i = 0; i < depth_images.size(); i++) {
    std::ostringstream curr_frame_prefix;
    curr_frame_prefix << std::setw(6) << std::setfill('0') << i;
    cv::imwrite(req.data_path+"/frame-"+curr_frame_prefix.str()+".depth.png",depth_images[i]);
  }

}

bool MeshDepthRecover::isInside(const Eigen::Vector2d& p, const Eigen::Vector2d& v0, const Eigen::Vector2d& v1, const Eigen::Vector2d& v2, Eigen::Vector3d& bcoords)
{
  Eigen::Vector2d e1 = v1 - v0, e2 = v2 - v0, e0 = p - v0;
  double e12 = e1[0] * e2[1] - e1[1] * e2[0];  // e1 x e2
  if (fabs(e12) < 1e-8)
  {  // triangle is degenerate: two edges are almost colinear
    bcoords = Eigen::Vector3d(0, 0, 0);
    return false;
  }
  bcoords[1] = (e0[0] * e2[1] - e0[1] * e2[0]) / e12;  // e0 x e2
  bcoords[2] = (e1[0] * e0[1] - e1[1] * e0[0]) / e12;  // e1 x e0
  bcoords[0] = 1 - bcoords[1] - bcoords[2];
  return (bcoords.minCoeff() >= 0.0);
}
bool MeshDepthRecover::checkPointInsideTriangle(const pcl::PointXY& p1, const pcl::PointXY& p2, const pcl::PointXY& p3, const pcl::PointXY& pt)
{
  // Compute vectors
  Eigen::Vector2d v0, v1, v2;
  v0(0) = p3.x - p1.x;
  v0(1) = p3.y - p1.y;  // v0= C - A
  v1(0) = p2.x - p1.x;
  v1(1) = p2.y - p1.y;  // v1= B - A
  v2(0) = pt.x - p1.x;
  v2(1) = pt.y - p1.y;  // v2= P - A

  // Compute dot products
  double dot00 = v0.dot(v0);  // dot00 = dot(v0, v0)
  double dot01 = v0.dot(v1);  // dot01 = dot(v0, v1)
  double dot02 = v0.dot(v2);  // dot02 = dot(v0, v2)
  double dot11 = v1.dot(v1);  // dot11 = dot(v1, v1)
  double dot12 = v1.dot(v2);  // dot12 = dot(v1, v2)

  // Compute barycentric coordinates
  double invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);
  double u = (dot11 * dot02 - dot01 * dot12) * invDenom;
  double v = (dot00 * dot12 - dot01 * dot02) * invDenom;

  // Check if point is in triangle
  return ((u >= 0) && (v >= 0) && (u + v < 1));
}

bool MeshDepthRecover::getPixelCoords4GlobalPt(const int height, const int width, const Eigen::Matrix3d& intrins, Eigen::Vector2d& pixel, const Eigen::Matrix4d& world_to_cam,
                                               const Eigen::Vector3d& global_pt)
{
  Eigen::Affine3d a;
  a.matrix() = world_to_cam;
  Eigen::Vector3d uv = intrins * a * global_pt;
  pixel(0) = uv(0) / uv(2);
  pixel(1) = uv(1) / uv(2);
  double w = pixel(0) / width;
  double h = 1.0 - pixel(1) / height;
  if (w > 0.0 && w < 1.0 && h >= 0.0 && h < 1.0)
    return true;
  else
    return false;
}
bool MeshDepthRecover::faceProjected(const int height, const int width, const Eigen::Matrix3d& intrins, const Eigen::Matrix4d& world_to_cam, const Eigen::Vector3d& global_pt0,
                                     const Eigen::Vector3d& global_pt1, const Eigen::Vector3d& global_pt2, Eigen::Vector2d& pixel0, Eigen::Vector2d& pixel1, Eigen::Vector2d& pixel2)
{
  return (getPixelCoords4GlobalPt(height, width, intrins, pixel0, world_to_cam, global_pt0) && getPixelCoords4GlobalPt(height, width, intrins, pixel1, world_to_cam, global_pt1) &&
          getPixelCoords4GlobalPt(height, width, intrins, pixel2, world_to_cam, global_pt2));
}
Eigen::Vector3d MeshDepthRecover::globalToCameraSpace(const Eigen::Vector3d& pt, const Eigen::Matrix4d& world_to_cam)
{
  return (world_to_cam * Eigen::Vector4d(pt[0], pt[1], pt[2], 1.0)).head<3>();
}
Eigen::Vector2d MeshDepthRecover::cameraToImgSpace(const Eigen::Vector3d& pt, const Eigen::Matrix3d& intrins)
{
  return Eigen::Vector2d(intrins(0, 0) * pt[0] / pt[2] + intrins(0, 2), intrins(1, 1) * pt[1] / pt[2] + intrins(1, 2));
}
void MeshDepthRecover::getTriangleCircumscribedCircleCentroid(const pcl::PointXY& p1, const pcl::PointXY& p2, const pcl::PointXY& p3, pcl::PointXY& circumcenter, double& radius)
{
  circumcenter.x = static_cast<float>(p1.x + p2.x + p3.x) / 3;
  circumcenter.y = static_cast<float>(p1.y + p2.y + p3.y) / 3;

  double r1 = (circumcenter.x - p1.x) * (circumcenter.x - p1.x) + (circumcenter.y - p1.y) * (circumcenter.y - p1.y);
  double r2 = (circumcenter.x - p2.x) * (circumcenter.x - p2.x) + (circumcenter.y - p2.y) * (circumcenter.y - p2.y);
  double r3 = (circumcenter.x - p3.x) * (circumcenter.x - p3.x) + (circumcenter.y - p3.y) * (circumcenter.y - p3.y);
  radius = std::sqrt(std::max(r1, std::max(r2, r3)));
}
int main(int argc, char** argv)
{
  ros::init(argc, argv, "mesh_depth_recover_node");
  ros::NodeHandle node;

  MeshDepthRecover::Ptr mesh_rgbd_rendering = std::make_shared<MeshDepthRecover>(node);
  ros::spin();

  return 0;
}
