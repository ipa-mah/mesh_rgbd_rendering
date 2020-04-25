#include <iostream>
#include <ostream>

// PCL
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>
// dynamic reconfigure
#include <dynamic_reconfigure/server.h>
#include <ipa_rubbish_bin_detection/ipa_rubbish_bin_detectionConfig.h>
// ROS
#include <ros/node_handle.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

// OpenCV
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

// TOPICS
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
// TF
#include <Eigen/Geometry>
#include <eigen_conversions/eigen_msg.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>

// MESSAGES
#include <actionlib/client/simple_action_client.h>
#include <geometry_msgs/Pose.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <std_srvs/Trigger.h>

// HEADERS
#include "ipa_rubbish_bin_detection/CylinderParams.h"
#include "ipa_rubbish_bin_detection/shape_estimation.hpp"
// There are 4 fields and a total of 7 parameters used to define this.
struct CylinderParams {
  /* Radius of the cylinder. */
  double radius;
  /* Direction vector towards the z-axis of the cylinder. */
  Eigen::Vector3d direction_vec;
  /* Center point of the cylinder. */
  Eigen::Vector3d center_pt;
  /* Height of the cylinder. */
  double height;
  CylinderParams()
      : radius(0), direction_vec(0, 0, 0), center_pt(0, 0, 0), height(0) {}
};
inline std::ostream &operator<<(std::ostream &s, const CylinderParams &v) {

  s << "radius: " << v.radius << std::endl;
  s << "direction vector:" << std::endl;
  s << v.direction_vec << std::endl;
  s << "center point:" << std::endl;
  s << v.center_pt << std::endl;
  s << "height: " << v.height << std::endl;

  return (s);
}
class IpaWasteBasteDetection {
public:
  typedef std::shared_ptr<IpaWasteBasteDetection> Ptr;
  IpaWasteBasteDetection(ros::NodeHandle node_handle);
  virtual ~IpaWasteBasteDetection();
  //================ Functions
  //==================================================================================
  //=============================================================================================================
  /**
   * Function is called if point cloud topic is received.
   * @param [in] point_cloud2_rgb_msg	Point cloude message from camera.
   */
  void preprocessingCallback(
      const sensor_msgs::PointCloud2ConstPtr &point_cloud2_rgb_msg);

  /**
   * Converts: "sensor_msgs::PointCloud2" \f$ \rightarrow \f$
   *"pcl::PointCloud<pcl::PointXYZRGB>::Ptr".
   *
   * The function detects a plane and cylinder in the point cloud, draw the 3D
   *bounding box of the cylinder object in the color image
   *
   *
   *	@param [in] 	input_cloud 	original point cloud from depth camera.
   *  @param [out] 	  cylinder_coud 	Point cloud for cylinder object.
   *	@param [out] 	color_image		Shows image with the bounding
   *box of the cylinder object if detected
   *	@param [out]	plane_model coefficients of the plane floor
   *  @param [out]	plane_model coefficients of the cylinder object
   *  @param [out]	cylinder_model coefficients of the cylinder object
   *  @param [out]	cylinder_params parameters of the cylinder object
   *	@return 		True if any cylinder could be found in the
   *original point cloud.
   */
  bool cylinderDetection(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud,
                         pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cylinder_cloud,
                         cv::Mat &color_image, const std_msgs::Header &header,
                         pcl::ModelCoefficients::Ptr &plane_model,
                         pcl::ModelCoefficients::Ptr &cylinder_model,
                         Eigen::Affine3d &transform_cam2plane,
                         CylinderParams &cylinder_params);

  void calculateGraspingPoints(
      std::vector<geometry_msgs::PoseStamped> &grasping_poses,
      const std_msgs::Header &header,
      pcl::PointCloud<pcl::PointXYZRGB> &original_cloud,
      const CylinderParams &cylinder_params,
      const Eigen::Affine3d &transform_cam2plane,
      const Eigen::Affine3d &center_wtf_camera_pose);

protected:
  /// dynamic reconfigure
  dynamic_reconfigure::Server<
      ipa_rubbish_bin_detection::ipa_rubbish_bin_detectionConfig>
      dynamic_reconfigure_server_;
  void dynamicReconfigureCallback(
      ipa_rubbish_bin_detection::ipa_rubbish_bin_detectionConfig &config,
      uint32_t level);

  // ROS node handle
  ros::NodeHandle node_handle_;

  // Used to subscribe and publish images.
  image_transport::ImageTransport *it_;

  // Used to receive point cloud topic from camera.
  ros::Subscriber camera_depth_points_sub_;

  // Publisher for plane segmented cloud.
  ros::Publisher floor_plane_pub_;

  // visualize cylinder
  ros::Publisher marker_pub_;

  // Publisher for cylinder segmented cloud
  ros::Publisher cylinder_pub_;

  // Publisher for cylinder parameters
  ros::Publisher cylinder_params_pub_;

  // For received result from action server: Means image with positions of dirt
  image_transport::Publisher
      cylinder_detection_image_pub_; // topic for publishing the image
                                     // containing the dirt positions

  Eigen::Matrix3d cam_params_;
  double z_min_val_, z_max_val_; // z threshold for pass through filter

  // plane segmentation
  double floor_plane_inlier_distance_;
  int floor_search_iterations_;
  int min_plane_points_;

  // cylinder detection
  double cylinder_distace_threshold_;
  double cylinder_radius_;

  double ground_truth_cylinder_height_;
  bool check_ground_truth_cylinder_height_;

  // Use direction of cylinder as direction of plane
  bool use_direction_plane_;

  // remove outliers by using PCL function
  bool use_remove_outliers_;
  // clustering
  int min_cluster_size_;
  int max_cluster_size_;
  double cluster_tolerance_;

  // camera parameters
  double focal_x_, focal_y_;
  double c_x_, c_y_;

  // number of grasping points
  int num_grasping_points_;
  // debug parametesr to show pointclouds, images
  std::map<std::string, bool> debug_;

  tf::TransformBroadcaster tf_broadcaster_; /** tf brodcaster */

}; // end class
