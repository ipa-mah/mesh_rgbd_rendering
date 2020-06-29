#include <ros/ros.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include <ros/package.h>
#include <ros_object_detection_msgs/MeshRGBDRendering.h>
#include <ros_mesh_rgbd_rendering/shader.h>
#include <ros_mesh_rgbd_rendering/triangle_mesh.h>
// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>

#include <yaml-cpp/yaml.h>
#include <unordered_map>

class MeshRGBDRendering
{
  public:
	using Ptr = std::shared_ptr<MeshRGBDRendering>;
	MeshRGBDRendering(const ros::NodeHandle& node_handle, const std::string& name, const std::string& render_texture_mesh_vertex_shader_file,
					  const std::string& render_texture_mesh_fragment_shader_file);
	virtual ~MeshRGBDRendering();

  bool srvMeshRGBDRenderingCallBack(ros_object_detection_msgs::MeshRGBDRendering::Request& req,
                                    ros_object_detection_msgs::MeshRGBDRendering::Response& res);

	bool readData(const std::string& data_path);

  public:
	bool loadShaders();
	bool compileShaders()
	{
	}
	/// Function to create a window and initialize GLFW
	/// This function MUST be called from the main thread.
	bool createVisualizerWindow(const std::string& window_name = "MeshRGBDRendering", const int width = 640, const int height = 480, const int left = 50, const int top = 50,
								const bool visible = true);

	bool bindTriangleMesh(std::vector<Eigen::Vector3f>& points, std::vector<Eigen::Vector2f>& uvs);
	void release();

  protected:
  protected:
	ros::NodeHandle node_handle_;
	ros::ServiceServer mesh_rgbd_rendering_service_;
	Eigen::Matrix3d intrins_;
	std::vector<GLMatrix4f> extrinsics_;
	int image_width_;
	int image_height_;

  protected:
	std::string render_texture_mesh_vertex_shader_file_;	// vertex shader file
	std::string render_texture_mesh_fragment_shader_file_;  // fragment shader file
	std::string vertex_shader_code_;						// code string in vertex shader file
	std::string fragment_shader_code_;						// code string in fragment shader file

	//  GLuint vertex_position_; // name of vertex position in gsls shader
	//  GLuint vertex_uv_;       // name of uv position in gsls fragment shader
	//  GLuint texture_;
	//  GLuint MVP_;        // extrinsics in gsls vertex shader
	//  int num_materials_; // number of texture images
	GLuint vao_id_;

	//  std::vector<int> array_offsets_; //??
	//  std::vector<GLsizei> draw_array_sizes_;
	//  std::vector<GLuint> vertex_position_buffers_;
	//  std::vector<GLuint> vertex_uv_buffers_;
	//  std::vector<GLuint> texture_buffers_;

	// window
	GLFWwindow* window_ = NULL;
	std::string window_name_ = "MeshRGBDRendering";

	Shader::Ptr shader_;
	TriangleMesh::Ptr triangle_mesh_;
};

MeshRGBDRendering::MeshRGBDRendering(const ros::NodeHandle& node_handle, const std::string& name, const std::string& render_texture_mesh_vertex_shader_file,
									 const std::string& render_texture_mesh_fragment_shader_file)
  : node_handle_(node_handle), render_texture_mesh_vertex_shader_file_(render_texture_mesh_vertex_shader_file), render_texture_mesh_fragment_shader_file_(render_texture_mesh_fragment_shader_file)
{
	mesh_rgbd_rendering_service_ = node_handle_.advertiseService("mesh_rgbd_rendering_service", &MeshRGBDRendering::srvMeshRGBDRenderingCallBack, this);
}

MeshRGBDRendering::~MeshRGBDRendering()
{
}

bool MeshRGBDRendering::createVisualizerWindow(const std::string& window_name, const int width, const int height, const int left, const int top, const bool visible)
{
	window_name_ = window_name;
	if (window_)
	{  // window already created
		glfwSetWindowPos(window_, left, top);
		glfwSetWindowSize(window_, width, height);
		return true;
	}
	if (!glfwInit())
	{
		ROS_INFO("MeshRGBDRendering::createVisualizerWindow::Failed to initialize GLFW");
		glfwTerminate();
		return false;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
#ifndef HEADLESS_RENDERING
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_VISIBLE, visible ? 1 : 0);
	window_ = glfwCreateWindow(width, height, window_name_.c_str(), NULL, NULL);
	if (!window_)
	{
		ROS_INFO("MeshRGBDRendering::createVisualizerWindow::Failed to create window");
		glfwTerminate();
		return false;
	}

	// Initialize GLEW
	glfwMakeContextCurrent(window_);
	glewExperimental = GL_TRUE;  // Needed for core profile
	if (glewInit() != GLEW_OK)
	{
		ROS_INFO("MeshRGBDRendering::createVisualizerWindow::Failed to initialize GLEW");
		glfwTerminate();
		return false;
	}

	// 1. bind Vertex Array Object
	glGenVertexArrays(1, &vao_id_);
	glBindVertexArray(vao_id_);

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window_, GLFW_STICKY_KEYS, GL_TRUE);
	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);
	return true;
}
/*
bool MeshRGBDRendering::bindTriangleMesh(std::vector<Eigen::Vector3f> &points,
										 std::vector<Eigen::Vector2f> &uvs)
{
  if(!triangle_mesh_->hasTriangles())
  {
	ROS_ERROR("Binding failed with empty triangle mesh.");
	return false;
  }
  std::vector<std::vector<Eigen::Vector3f>> tmp_points;
  std::vector<std::vector<Eigen::Vector2f>> tmp_uvs;
  num_materials_ = static_cast<int>(triangle_mesh_->textures_.size());

  array_offsets_.resize(num_materials_);
  draw_array_sizes_.resize(num_materials_);
  vertex_position_buffers_.resize(num_materials_);
  vertex_uv_buffers_.resize(num_materials_);
  texture_buffers_.resize(num_materials_);

  tmp_uvs.resize(num_materials_);
  tmp_points.resize(num_materials_);


  //Bind vertex and uv per material

  for(std::size_t i = 0 ; i < triangle_mesh_->triangles_.size(); i++)
  {
	const Eigen::Vector3i& triangle = triangle_mesh_->triangles_[i];
	int mi = triangle_mesh_->triangle_material_ids_[i]; // material id
	for(std::size_t j=0; j < 3 ; j++)
	{
	  std::size_t idx = 3 * i +j;
	  int vertex_idx = triangle(j);
	  tmp_points[mi].push_back(triangle_mesh_->vertices_[vertex_idx].cast<float>());
	  tmp_uvs[mi].push_back(triangle_mesh_->triangle_uvs_[idx].cast<float>());
	}
  }

  // Bind textures

  for (int mi = 0; mi < num_materials_; mi ++) {
	glGenTextures(1, & texture_buffers_[mi]);
	glBindTexture(GL_TEXTURE_2D,texture_buffers_[mi]);

	glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,triangle_mesh_->textures_[mi].cols,
				 triangle_mesh_->textures_[mi].rows,0,
				 GL_BGR,GL_UNSIGNED_BYTE,triangle_mesh_->textures_[mi].data);

	// Set texture clamping method
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  }

  // Point seperations
  array_offsets_[0] = 0;
  draw_array_sizes_[0] = tmp_points[0].size();
  for (int mi = 1; mi < num_materials_; ++mi) {
	draw_array_sizes_[mi] = tmp_points[mi].size();
	array_offsets_[mi] = array_offsets_[mi - 1] + draw_array_sizes_[mi - 1];
  }

  //prepare chunk of points and uvs
  points.clear();
  uvs.clear();
  for(int mi = 0; mi< num_materials_ ; mi++)
  {
	points.insert(points.end(),tmp_points[mi].begin(),tmp_points[mi].end());
	uvs.insert(uvs.end(),tmp_uvs[mi].begin(),tmp_uvs[mi].end());
  }


  return true;

}
*/
bool MeshRGBDRendering::srvMeshRGBDRenderingCallBack(ros_object_detection_msgs::MeshRGBDRendering::Request& req,
                                                     ros_object_detection_msgs::MeshRGBDRendering::Response& res)
{
	ROS_INFO("Running MeshRGBDRendering service");

	std::string obj_file = req.data_path + req.mesh_file;
	triangle_mesh_ = std::make_shared<TriangleMesh>();
	if (!readTextureMeshfromOBJFile(obj_file, triangle_mesh_))
	{
		std::cout << " MeshRGBDRendering::srvMeshRGBDRenderingCallBack is not successful" << std::endl;
		res.success = false;
		return false;
	};

	ROS_INFO("Reading intrinsics and views from parameter server");
	const std::string name_space = "/view_generator";

	std::vector<double> instrinsic;
	int views;
	if (!node_handle_.getParam(name_space + "/intrinsic", instrinsic))
	{
		std::cout << " MeshRGBDRendering::srvMeshRGBDRenderingCallBack is not "
					 "successful: ros param"
				  << name_space + "/intrinsic is not found";
		res.success = false;
		return false;
	};
	intrins_.setIdentity();
	intrins_(0, 0) = instrinsic[0];
	intrins_(1, 1) = instrinsic[4];
	intrins_(0, 2) = instrinsic[2];
	intrins_(1, 2) = instrinsic[5];
	std::cout << intrins_ << std::endl;
	if (!node_handle_.getParam(name_space + "/views", views))
	{
		std::cout << " MeshRGBDRendering::srvMeshRGBDRenderingCallBack is not "
					 "successful: ros param"
				  << name_space + "/views is not found" << std::endl;
		res.success = false;
		return false;
	};

	for (int i = 0; i < views; i++)
	{
		std::ostringstream curr_frame_prefix;
		curr_frame_prefix << std::setw(2) << std::setfill('0') << i;
		std::string position = name_space + "/" + curr_frame_prefix.str() + "_position";
		std::string orientation = name_space + "/" + curr_frame_prefix.str() + "_orientation";
		std::vector<double> translation, vec_rpy;
		node_handle_.getParam(position, translation);
		node_handle_.getParam(orientation, vec_rpy);
		Eigen::Vector3d trans(translation[0], translation[1], translation[2]);
		Eigen::Vector3d rpy(vec_rpy[0], vec_rpy[1], vec_rpy[2]);
		Eigen::Matrix4d extrinsic;
		extrinsic.setIdentity();
		extrinsic.topRightCorner(3, 1) = trans;
		Eigen::Matrix3d rot;
		rot = Eigen::AngleAxisd(vec_rpy[0], Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(vec_rpy[1], Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(vec_rpy[2], Eigen::Vector3d::UnitZ());
		extrinsic.topLeftCorner(3, 3) = rot;
		extrinsics_.push_back(extrinsic.cast<GLfloat>());
	}

	image_width_ = static_cast<int>((intrins_(0, 2) + 0.5)) * 2;
	image_height_ = static_cast<int>((intrins_(1, 2) + 0.5)) * 2;
	ROS_INFO("Create OpenGL window");
	if (!createVisualizerWindow("MeshRGBDRendering", image_width_, image_height_, 50, 50, true))
	{
		std::cout << " MeshRGBDRendering::srvMeshRGBDRenderingCallBack is not successful" << std::endl;
		res.success = false;
		return false;
	};
	// create shader ptr
	shader_ = std::make_shared<Shader>();
	ROS_INFO("Load and compile shaders");
	shader_->loadShaders(render_texture_mesh_vertex_shader_file_.c_str(), render_texture_mesh_fragment_shader_file_.c_str());
	ROS_INFO("Prepare binding");

	GLuint vertex_position_;  // name of vertex position in gsls shader
	GLuint vertex_uv_;		  // name of uv position in gsls fragment shader
	GLuint texture_;
	GLuint MVP_;		 // extrinsics in gsls vertex shader
	int num_materials_;  // number of texture images

	std::vector<int> array_offsets_;  //??
	std::vector<GLsizei> draw_array_sizes_;
	std::vector<GLuint> vertex_position_buffers_;
	std::vector<GLuint> vertex_uv_buffers_;
	std::vector<GLuint> texture_buffers_;

	// get vertex, uv, transform position
	vertex_position_ = glGetAttribLocation(shader_->program_, "vertex_position");
	vertex_uv_ = glGetAttribLocation(shader_->program_, "vertex_uv");
	MVP_ = glGetUniformLocation(shader_->program_, "MVP");
	texture_ = glGetUniformLocation(shader_->program_, "diffuse_texture");

	// Bind geometry
	ROS_INFO("Bind geometry");
	std::vector<Eigen::Vector3f> points;
	std::vector<Eigen::Vector2f> uvs;

	if (!triangle_mesh_->hasTriangles())
	{
		ROS_ERROR("Binding failed with empty triangle mesh.");
		return false;
	}
	std::vector<std::vector<Eigen::Vector3f>> tmp_points;
	std::vector<std::vector<Eigen::Vector2f>> tmp_uvs;
	num_materials_ = static_cast<int>(triangle_mesh_->textures_.size());

	array_offsets_.resize(num_materials_);
	draw_array_sizes_.resize(num_materials_);
	vertex_position_buffers_.resize(num_materials_);
	vertex_uv_buffers_.resize(num_materials_);
	texture_buffers_.resize(num_materials_);

	tmp_uvs.resize(num_materials_);
	tmp_points.resize(num_materials_);

	// Bind vertex and uv per material

	for (std::size_t i = 0; i < triangle_mesh_->triangles_.size(); i++)
	{
		const Eigen::Vector3i& triangle = triangle_mesh_->triangles_[i];
		int mi = triangle_mesh_->triangle_material_ids_[i];  // material id
		for (std::size_t j = 0; j < 3; j++)
		{
			std::size_t idx = 3 * i + j;
			int vertex_idx = triangle(j);
			tmp_points[mi].push_back(triangle_mesh_->vertices_[vertex_idx].cast<float>());
			tmp_uvs[mi].push_back(triangle_mesh_->triangle_uvs_[idx].cast<float>());
		}
	}

	// Bind textures

	for (int mi = 0; mi < num_materials_; mi++)
	{
		glGenTextures(1, &texture_buffers_[mi]);
		glBindTexture(GL_TEXTURE_2D, texture_buffers_[mi]);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, triangle_mesh_->textures_[mi].cols, triangle_mesh_->textures_[mi].rows, 0, GL_BGR, GL_UNSIGNED_BYTE, triangle_mesh_->textures_[mi].data);

		// Set texture clamping method
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	// Point seperations
	array_offsets_[0] = 0;
	draw_array_sizes_[0] = tmp_points[0].size();
	for (int mi = 1; mi < num_materials_; ++mi)
	{
		draw_array_sizes_[mi] = tmp_points[mi].size();
		array_offsets_[mi] = array_offsets_[mi - 1] + draw_array_sizes_[mi - 1];
	}

	// prepare chunk of points and uvs
	points.clear();
	uvs.clear();
	for (int mi = 0; mi < num_materials_; mi++)
	{
		points.insert(points.end(), tmp_points[mi].begin(), tmp_points[mi].end());
		uvs.insert(uvs.end(), tmp_uvs[mi].begin(), tmp_uvs[mi].end());
	}

	// create buffer and bind geometry
	for (int mi = 0; mi < num_materials_; mi++)
	{
		// vertex
		glGenBuffers(1, &vertex_position_buffers_[mi]);
		glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffers_[mi]);
		glBufferData(GL_ARRAY_BUFFER, draw_array_sizes_[mi] * sizeof(Eigen::Vector3f), points.data() + array_offsets_[mi], GL_STATIC_DRAW);

		// uv
		glGenBuffers(1, &vertex_uv_buffers_[mi]);
		glBindBuffer(GL_ARRAY_BUFFER, vertex_uv_buffers_[mi]);
		glBufferData(GL_ARRAY_BUFFER, draw_array_sizes_[mi] * sizeof(Eigen::Vector2f), uvs.data() + array_offsets_[mi], GL_STATIC_DRAW);
	}

	// Get a handle for our "MVP" uniform
	GLMatrix4f projection_matrix;
	GLMatrix4f look_at_matrix;
	GLMatrix4f view_matrix;

	float z_near = 0.1f;
	float z_far = 10.0f;

	projection_matrix.setZero();
	projection_matrix(0, 0) = static_cast<float>(intrins_(0, 0) / intrins_(0, 2));
	projection_matrix(1, 1) = static_cast<float>(intrins_(1, 1) / intrins_(1, 2));
	projection_matrix(2, 2) = (z_near + z_far) / (z_near - z_far);
	projection_matrix(2, 3) = -2.0f * z_far * z_near / (z_far - z_near);
	projection_matrix(3, 2) = -1;

	look_at_matrix.setIdentity();
	look_at_matrix(1, 1) = -1.0f;
	look_at_matrix(2, 2) = -1.0f;

	GLMatrix4f gl_trans_scale;
	gl_trans_scale.setIdentity();
	Eigen::Vector3d trans(0, 0, 0.5);
	Eigen::Matrix3d rot;
	rot.setZero();
	rot(0, 0) = 1;
	rot(1, 2) = 1;
	rot(2, 1) = -1;
	Eigen::Matrix4d pose;
	pose.setIdentity();
	pose.topRightCorner(3, 1) = trans;
	pose.topLeftCorner(3, 3) = rot;
	// std::cout<<pose<<std::endl;

	ROS_INFO("Run rendering");
	for (std::size_t view = 0; view < extrinsics_.size(); view++)
	{
		// view_matrix = projection_matrix * look_at_matrix * gl_trans_scale * extrinsics_[view].cast<GLfloat>();
		view_matrix = projection_matrix * look_at_matrix * gl_trans_scale * pose.cast<GLfloat>();

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glUseProgram(shader_->program_);
		for (int mi = 0; mi < num_materials_; mi++)
		{
			glUniformMatrix4fv(MVP_, 1, GL_FALSE, view_matrix.data());

			glUniform1i(texture_, 0);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, texture_buffers_[mi]);

			glEnableVertexAttribArray(vertex_position_);
			glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffers_[mi]);
			glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);

			glEnableVertexAttribArray(vertex_uv_);
			glBindBuffer(GL_ARRAY_BUFFER, vertex_uv_buffers_[mi]);
			glVertexAttribPointer(vertex_uv_, 2, GL_FLOAT, GL_FALSE, 0, NULL);

			// Draw
			glViewport(0, 0, image_width_, image_height_);
			glDrawArrays(GL_TRIANGLES, 0, draw_array_sizes_[mi]);

			glDisableVertexAttribArray(vertex_position_);
			glDisableVertexAttribArray(vertex_uv_);
			glBindTexture(GL_TEXTURE_2D, 0);
		}
		// snippet of code to save color image
		// glFinish();
		cv::Mat mat(image_height_, image_width_, CV_8UC3);
		glPixelStorei(GL_PACK_ALIGNMENT, (mat.step & 3) ? 1 : 4);
		glPixelStorei(GL_PACK_ROW_LENGTH, mat.step / mat.elemSize());
		glReadPixels(0, 0, mat.cols, mat.rows, GL_BGR, GL_UNSIGNED_BYTE, mat.data);

		cv::flip(mat, mat, 0);
		std::ostringstream curr_frame_prefix;
		curr_frame_prefix << std::setw(6) << std::setfill('0') << view;
		cv::imwrite(req.data_path + "/frame-" + curr_frame_prefix.str() + ".rtexture.png", mat);
	}

	ROS_INFO("Rendering is sucessful");
	res.success = true;

	release();
	return true;
}
void MeshRGBDRendering::release()
{
	triangle_mesh_->clear();
	//  for (auto buf : vertex_position_buffers_) {
	//    glDeleteBuffers(1, &buf);
	//  }
	//  for (auto buf : vertex_uv_buffers_) {
	//    glDeleteBuffers(1, &buf);
	//  }
	//  for (auto buf : texture_buffers_) {
	//    glDeleteTextures(1, &buf);
	//  }

	//  glDeleteVertexArrays(1, &vao_id_);

	//
	//  vertex_position_buffers_.clear();
	//  vertex_uv_buffers_.clear();
	//  texture_buffers_.clear();
	//  draw_array_sizes_.clear();
	//  array_offsets_.clear();
	//  num_materials_ = 0;
	extrinsics_.clear();
	shader_->deleteProgram();
	glfwTerminate();
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "mesh_rgbd_rendering");
	ros::NodeHandle node;

	std::string shader_path = ros::package::getPath("ros_mesh_rgbd_rendering") + "/shader/";
	std::string vert_shader_file = shader_path + "TextureSimpleVertexShader.glsl";
	std::string frag_shader_file = shader_path + "TextureSimpleFragmentShader.glsl";

	MeshRGBDRendering::Ptr mesh_rgbd_rendering = std::make_shared<MeshRGBDRendering>(node, "MeshRGBDRenderingWindows", vert_shader_file, frag_shader_file);
	ros::spin();

	return 0;
}
