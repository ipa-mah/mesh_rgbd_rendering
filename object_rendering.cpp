#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <pcl/PolygonMesh.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/TextureMesh.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Core>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/ext.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "sources/shader.h"
#include "sources/gbuffer.h"
#include <json/json.h>



bool isDirExist(const std::string& filename)
{
    bool result = true;
    std::ifstream _file( filename.c_str(), std::ios::in );
    if(!_file) // it exists
        result = false;
    _file.close();
    return result;
}
bool createDir(const std::string& path)
{
    //If the directory is existed
    if( isDirExist(path) ){
        return true;
    }
    system( ("mkdir " + path).c_str() );
    return true;
}
void saveVertexMap(cv::Mat& mat, std::string filename)
{
    std::ofstream f(filename.c_str(), std::ios_base::binary);
    if(!f.is_open())
    {
        std::cerr << "ERROR - SaveVertexMap:" << std::endl;
        std::cerr << "\t ... Could not open " << filename << " \n";
        return ;
    }

    int channels = mat.channels();

    int header[3];
    header[0] = mat.rows;
    header[1] = mat.cols;
    header[2] = channels;
    f.write((char*)header, 3 * sizeof(int));

    float* ptr = 0;
    for(unsigned int row=0; row<(unsigned int)mat.rows; row++)
    {
        ptr = mat.ptr<float>(row);
        f.write((char*)ptr, channels * mat.cols * sizeof(float));
    }
    f.close();
}

GLFWwindow* window;
int image_width_,image_height_;
// glfw: whenever the window size changed (by OS or user resize) this callback function executes
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}
bool initGLFW()
{
    if (!glfwInit())
    {
        std::cout << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // We don't want the old OpenGL
    glfwWindowHint(GLFW_VISIBLE, false);  // hide window after creation
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);  // uncomment this statement to fix compilation on OS X
#endif

    // Open a window and create its OpenGL context
    window = glfwCreateWindow(image_width_, image_height_, "RenderingWindow", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version "
                     "of the tutorials."
                  << std::endl; // this is from some online tutorial
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Initialize GLEW
    glewExperimental = true;  // Needed for core profile
    if (glewInit() != GLEW_OK)
    {
        fprintf(stderr, "Failed to initialize GLEW\n");
        getchar();
        glfwTerminate();
        return false;
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    // Dark blue background
    glClearColor(0.0f, 0.0f, 0.f, 0.0f);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    return true;
}


struct VertFace
{
    VertFace() : vertex_id(),face_id(){}
    int vertex_id;
    int face_id;
};
struct Vertex
{
    glm::vec3 pos;// position
    glm::vec3 color; // color
    glm::vec2 uv;
    glm::vec3 normal;
    Vertex(){pos = color = glm::vec3(0);}
};


struct TextureMesh
{
    std::unordered_map<std::string, int> material_names;
    std::vector<cv::Mat> texture_images;
    cv::Mat texture_atlas;
    std::vector<unsigned int> faces;
    std::vector<Vertex> vertices;
    //pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::PolygonMesh polygon_mesh;
    int vertex_num_;
    int face_num_;
    std::string mesh_suffix_;
    bool vtx_normal;
    bool vtx_texture;
    TextureMesh()
    {
        vtx_normal = vtx_texture = false;
        vertex_num_ = face_num_ = 0;
    }
};


class ObjectRendering
{
public:
    typedef std::shared_ptr<ObjectRendering> Ptr;
    ObjectRendering();
    virtual ~ObjectRendering();
    bool readData(const std::string& data_path,const std::string& object_name);
    bool saveTextureImage();
    bool saveDepthImage();
    void draw();
    void saveOdometryIPA();
private:
    std::string data_path_;
    std::string object_data_path_,object_name_;
    TextureMesh texture_mesh_;
    std::vector<int> image_y_bases_;
    GLFWwindow* window_;
    Eigen::Matrix3d intrins_;
    std::vector<Eigen::Affine3d> cam2world_;
    const float near_, far_;
    unsigned int VAO_, VBO_, EBO_;

    bool readMTLandTextureImages(const std::string obj_folder, const std::string mtl_fname, std::unordered_map<std::string, int>& material_names,
                                 std::vector<cv::Mat>& texture_images,cv::Mat& texture_atlas);
    void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    bool interpolate(const Eigen::Vector2d& p, const Eigen::Vector2d& v0, const Eigen::Vector2d& v1, const Eigen::Vector2d& v2,Eigen::Vector3d& bcoords);
    bool getPixelCoords4GlobalPt(const int height, const int width,
                                 const Eigen::Matrix3d& intrins,Eigen::Vector2d& pixel, const Eigen::Matrix4d& world_to_cam,const Eigen::Vector3d& global_pt);
    bool faceProjected(const int height, const int width, const Eigen::Matrix3d& intrins,const Eigen::Matrix4d& world_to_cam,
                       const Eigen::Vector3d &global_pt0, const Eigen::Vector3d &global_pt1,
                       const Eigen::Vector3d &global_pt2, Eigen::Vector2d &pixel0,
                       Eigen::Vector2d &pixel1, Eigen::Vector2d &pixel2);
    Eigen::Vector3d globalToCameraSpace(const Eigen::Vector3d& pt, const Eigen::Matrix4d& world_to_cam);
    Eigen::Vector2d cameraToImgSpace(const Eigen::Vector3d& pt, const Eigen::Matrix3d& intrins);
    void getTriangleCircumscribedCircleCentroid ( const pcl::PointXY &p1, const pcl::PointXY &p2, const pcl::PointXY &p3, pcl::PointXY &circumcenter, double &radius);
    bool checkPointInsideTriangle(const pcl::PointXY &p1, const pcl::PointXY &p2, const pcl::PointXY &p3, const pcl::PointXY &pt);
    bool isInside(const Eigen::Vector2d& p, const Eigen::Vector2d& v0, const Eigen::Vector2d& v1, const Eigen::Vector2d& v2,Eigen::Vector3d& bcoords);
    glm::mat4 computeTransformationForFrame(int frame_idx);

};

ObjectRendering::ObjectRendering(): near_(0.00001f),far_(100000.0)
{
    intrins_.setIdentity();



}
ObjectRendering::~ObjectRendering()
{}
void ObjectRendering::framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);

}
bool ObjectRendering::readMTLandTextureImages(const std::string obj_folder, const std::string mtl_fname, std::unordered_map<std::string, int>& material_names,
                                              std::vector<cv::Mat>& texture_images,cv::Mat& texture_atlas)
{
    std::ifstream readin(mtl_fname, std::ios::in);
    if (readin.fail() || readin.eof())
    {
        std::cout << "Cannot read MTL file " << mtl_fname << std::endl;
        return false;
    }
    std::string str_line, str_first, str_img, str_mtl;
    while (!readin.eof() && !readin.fail())
    {
        getline(readin, str_line);
        if (!readin.good() || readin.eof())
            break;
        std::istringstream iss(str_line);
        if (iss >> str_first)
        {
            if (str_first == "newmtl")
            {
                iss >> str_mtl;
                material_names[str_mtl] = int(texture_images.size());
            }
            if (str_first == "map_Kd")
            {
                iss >> str_img;
                str_img = obj_folder + str_img;
                cv::Mat img = cv::imread(str_img, CV_LOAD_IMAGE_COLOR);
                if (img.empty() || img.depth() != CV_8U)
                {
                    std::cout << "ERROR: cannot read color image " << str_img << std::endl;
                    return false;
                }
                texture_images.push_back(std::move(img));
            }
        }
    }
    readin.close();
    std::cout<<"ObjectRendering::readMTLandTextureImages: number of texture images: "<<texture_images.size()<<std::endl;
    int width = 0, height = 0;
    int y_base = 0;

    for (int i = 0; i < texture_images.size(); ++i)
    {
        width = std::max(width, texture_images[i].cols);
        image_y_bases_.push_back(y_base);
        y_base += texture_images[i].rows;
    }
    height = y_base;

    int n = 2;
    while (n < width)
        n *= 2;
    int texture_atlas_width = n;
    n = 2;
    while (n < height)
        n *= 2;
    int texture_atlas_height_ = n;
    texture_atlas = cv::Mat(texture_atlas_height_, texture_atlas_width, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 0; i < texture_images.size(); ++i)
    {
        texture_images[i].copyTo(texture_atlas(cv::Rect(0, texture_atlas_height_ - image_y_bases_[i] - texture_images[i].rows,
                                                        texture_images[i].cols, texture_images[i].rows)));

    }
    return true;
}
bool ObjectRendering::readData(const std::string &data_path,const std::string& object_name)
{

    data_path_ = data_path;

   // std::string training_data_path = "/home/ipa-mah/object_perception_intern_ws/src/cob_object_perception_intern/cob_object_detection/common/files/models/";
    std::string training_data_path = data_path;
    object_data_path_ = training_data_path+"pc_" + object_name+"/";
    std::cout<<object_data_path_<<std::endl;
    createDir(object_data_path_);
    object_name_ = object_name;
    Json::Value root;
    std::ifstream config_doc(data_path+"/config.json", std::ifstream::binary);
    if(!config_doc.is_open())
    {
        std::cout<<"No config.json file in the data path"<<std::endl;
        return false;
    }
    config_doc >> root;

    intrins_(0,0) = root["camera_matrix"].get("focal_x",500).asDouble();
    intrins_(1,1) = root["camera_matrix"].get("focal_y",500).asDouble();
    intrins_(0,2) = root["camera_matrix"].get("c_x",320).asDouble();
    intrins_(1,2) = root["camera_matrix"].get("c_y",240).asDouble();
    for(const auto& node : root["views"])
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
        cam2world_.push_back(cam2world);
    }

    std::cout<<"Virtual Intrinsics:"<<std::endl<<intrins_<<std::endl;
    image_width_ = intrins_(0,2) * 2;
    image_height_ = intrins_(1,2) * 2;
    std::cout<<"image width: "<<image_width_<<std::endl;
    std::cout<<"image height: "<<image_height_<<std::endl;



    std::string obj_file = data_path +"/texture_model.obj";
    std::ifstream readin(obj_file, std::ios::in);
    if (readin.fail() || readin.eof())
    {
        std::cout << "Cannot read OBJ file " << obj_file << std::endl;
        return false;
    }
    std::string str_line, str_first, str_mtl_name, mtl_fname;
    std::vector<glm::vec3> vertices, normals;
    std::vector<glm::vec2> uvs;
    float x, y, z;
    unsigned int f, vt, vn, cur_tex_idx;
    std::string obj_folder;
    int face_vidx = 0;
    pcl::PointCloud<pcl::PointXYZ> cloud;
    while (!readin.eof() && !readin.fail())
    {
        getline(readin, str_line);
        if (!readin.good() || readin.eof())
            break;
        if (str_line.size() <= 1)
            continue;
        std::istringstream iss(str_line);
        iss >> str_first;
        if (str_first[0] == '#')
            continue;
        else if (str_first == "mtllib")
        {  // mtl file
            iss >> mtl_fname;
            size_t pos = obj_file.find_last_of("\\/");
            if (pos != std::string::npos)
                obj_folder = obj_file.substr(0, pos + 1);
            mtl_fname = obj_folder + mtl_fname;
            if (!readMTLandTextureImages(obj_folder,mtl_fname,texture_mesh_.material_names,
                                         texture_mesh_.texture_images,texture_mesh_.texture_atlas))
                return false;
        }
        else if (str_first == "v")
        {
            iss >> x >> y >> z;
            vertices.push_back(glm::vec3(x, y, z));
            cloud.points.push_back(pcl::PointXYZ(x,y,z));
        }
        else if (str_first == "vt")
        {
            iss >> x >> y;
            uvs.push_back(glm::vec2(x, y));
            if (!texture_mesh_.vtx_texture)
                texture_mesh_.vtx_texture = true;
        }
        else if (str_first == "vn")
        {
            iss >> x >> y >> z;
            normals.push_back(glm::vec3(x, y, z));
            if (!texture_mesh_.vtx_normal)
                texture_mesh_.vtx_normal = true;
        }
        else if (str_first == "usemtl")
        {
            iss >> str_mtl_name;
            if (texture_mesh_.material_names.find(str_mtl_name) == texture_mesh_.material_names.end())
            {
                std::cout << "ERROR: cannot find this material " << str_mtl_name << " in the mtl file " << mtl_fname << std::endl;
                return false;
            }
            cur_tex_idx = texture_mesh_.material_names[str_mtl_name];
        }
        else if (str_first == "f")
        {
            int loop = 3;
            pcl::Vertices vert_indices;

            while (loop-- > 0)
            {
                iss >> f;
                Vertex vtx;
                f--;
                vtx.pos = vertices[f];
                if (texture_mesh_.vtx_texture)
                {
                    if (texture_mesh_.vtx_normal)
                    {  // 'f/vt/vn'
                        iss.get();
                        iss >> vt;
                        vt--;
                        iss.get();
                        iss >> vn;
                        vn--;
                    }
                    else
                    {  // 'f/vt'
                        iss.get();
                        iss >> vt;
                        vt--;
                    }

                    vtx.uv[0] = uvs[vt][0] * texture_mesh_.texture_images[cur_tex_idx].cols / texture_mesh_.texture_atlas.cols;
                    vtx.uv[1] = (uvs[vt][1] * texture_mesh_.texture_images[cur_tex_idx].rows + image_y_bases_[cur_tex_idx]) /
                            ( texture_mesh_.texture_atlas.rows);
                    if (texture_mesh_.vtx_normal)
                        vtx.normal = normals[vn];
                }
                else if (texture_mesh_.vtx_normal)
                {  // 'f//vn'
                    iss.get();
                    iss.get();
                    iss >> vn;
                    vn--;
                    vtx.normal = normals[vn];
                }

                texture_mesh_.vertices.push_back(vtx);
                texture_mesh_.faces.push_back(face_vidx++);
                vert_indices.vertices.push_back(f);
            }
            texture_mesh_.polygon_mesh.polygons.push_back(vert_indices);
        }
    }
    pcl::toPCLPointCloud2(cloud,texture_mesh_.polygon_mesh.cloud);
    readin.close();
    texture_mesh_.vertex_num_ = int(vertices.size());
    texture_mesh_.face_num_ = face_vidx / 3;
    texture_mesh_.mesh_suffix_ = "obj";
    std::cout << "#Vertex: " << texture_mesh_.vertex_num_ << ", #Faces: " << texture_mesh_.face_num_ << std::endl;

    return true;
}

void ObjectRendering::draw()
{
    // draw mesh
    glViewport(0, 0, image_width_, image_height_);
    glBindVertexArray(VAO_);
    glDrawElements(GL_TRIANGLES, int(texture_mesh_.faces.size()), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}


bool ObjectRendering::saveTextureImage()
{
    std::cout<<"save rgb images"<<std::endl;
    GBuffer image_buffer;
    image_buffer.initNew(image_width_,image_height_);

    glGenVertexArrays(1, &VAO_);
    glGenBuffers(1, &VBO_);
    glGenBuffers(1, &EBO_);

    glBindVertexArray(VAO_);
    // load data into vertex buffers
    glBindBuffer(GL_ARRAY_BUFFER, VBO_);
    glBufferData(GL_ARRAY_BUFFER, texture_mesh_.vertices.size() * sizeof(Vertex), &texture_mesh_.vertices[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, texture_mesh_.faces.size() * sizeof(unsigned int), &texture_mesh_.faces[0], GL_STATIC_DRAW);

    // Set attributes for vertices
    // vertex positions
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *)0);
    // vertex colors
    glEnableVertexAttribArray(1);
    // The first parameter is the offset, while the last one is the offset variable name and struct name which
    // must be exactly the same as that used in struct Vertex.
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          (void *)offsetof(Vertex, color));  // name of 'color' variable in struct Vertex
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          (void *)offsetof(Vertex, uv));  // name of 'uv' variable in struct Vertex

    cv::flip(texture_mesh_.texture_atlas, texture_mesh_.texture_atlas, 0);
    unsigned int texture0_;
    glGenTextures(1, &texture0_);
    glBindTexture(GL_TEXTURE_2D, texture0_);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Set texture clamping method
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

    // NOTE: this will generate texture on GPU. It may crash on a GPU card with insufficient memory if
    // using a super large texture image.
    glTexImage2D(GL_TEXTURE_2D,  // Type of texture
                 0,                       // Pyramid level (for mip-mapping) - 0 is the top level
                 GL_RGB,                  // Internal colour format to convert to
                 texture_mesh_.texture_atlas.cols,    // Image width  i.e. 640 for Kinect in standard mode
                 texture_mesh_.texture_atlas.rows,   // Image height i.e. 480 for Kinect in standard mode
                 0,                       // Border width in pixels (can either be 1 or 0)
                 GL_BGR,                  // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
                 GL_UNSIGNED_BYTE,        // Image data type
                 texture_mesh_.texture_atlas.ptr());   // The actual image data itself

    glGenerateMipmap(GL_TEXTURE_2D);
    glBindVertexArray(0);
    Shader shader;
    shader.LoadShaders("../sources/savemode.vert","../sources/savemode.frag");
    shader.setInt("texture_sampler",0);
    glm::mat4 transform_perspective; // perspective transformation
    transform_perspective = glm::mat4(0);
    transform_perspective[0][0] = intrins_(0,0) / intrins_(0,2);
    transform_perspective[1][1] = intrins_(1,1) / intrins_(1,2);
    transform_perspective[2][2] = (near_ + far_) / (near_ - far_);
    transform_perspective[2][3] = -1;  // glm matrix is in column-major
    transform_perspective[3][2] = 2 * far_ * near_ / (near_ - far_);
    for(int frame_idx = 0; frame_idx<cam2world_.size();frame_idx++)
    {
        glClear(GL_COLOR_BUFFER_BIT||GL_DEPTH_BUFFER_BIT);
        shader.useProgram();
        shader.setFloat("near",near_);
        shader.setFloat("far",far_);
        shader.setBool("flag_show_color", false);
        shader.setBool("flag_show_texture", true);
        image_buffer.bindForWriting();
        // NOTE: depth test and clear function MUST be put after binding frame buffer and also need to
        // run for each frame, otherwise the extracted depth and color data will not have depth test.
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 glm_cam2world;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
            {
                glm_cam2world[j][i] = cam2world_[frame_idx](i,j);
            }
        glm::mat4 trans_model = glm::inverse(glm_cam2world); //extrinsics
        glm::mat4 trans_scale = glm::mat4(1.0);
        // In 3D model (world) coordinate space, +x is to the right, +z is to the inside of the screen, so +y is to the bottom.
        glm::mat4 trans_camera = glm::lookAt(glm::vec3(0, 0, 0),  // In OpenGL camera position is fixed at (0,0,0)
                                             glm::vec3(0, 0, 1),                                   // +z, position where the camera is looking at
                                             glm::vec3(0, -1, 0)                                   // +y direction
                                             );
        glm::mat4 gl_transform = transform_perspective * trans_camera * trans_scale * trans_model;
        shader.setMat4("transform",gl_transform);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture0_);
        shader.setInt("texture_sampler", 0);
        draw();
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        image_buffer.bindForReading();

        image_buffer.setReadBuffer(GBuffer::GBUFFER_TEXTURE_TYPE_COLOR);
        cv::Mat mat(image_height_, image_width_, CV_8UC3);
        //use fast 4-byte alignment (default anyway) if possible
        glPixelStorei(GL_PACK_ALIGNMENT, (mat.step & 3) ? 1 : 4);
        //set length of one complete row in destination data (doesn't need to equal img.cols)
        glPixelStorei(GL_PACK_ROW_LENGTH, mat.step/mat.elemSize());
        glReadPixels(0, 0, mat.cols, mat.rows, GL_BGR, GL_UNSIGNED_BYTE, mat.data);
        cv::flip(mat, mat, 0);
        std::ostringstream curr_frame_prefix;
        //        curr_frame_prefix << std::setw(6) << std::setfill('0') << frame_idx;
        //        cv::imwrite(data_path_+"/frame-"+curr_frame_prefix.str()+".rtexture.png",mat);
        curr_frame_prefix<<frame_idx;
        cv::imwrite(object_data_path_+object_name_+"_"+curr_frame_prefix.str()+"_coloredPC_color_8U3.png",mat);
    }

}

bool ObjectRendering::saveDepthImage()
{

    std::cout<<"save depth images"<<std::endl;
    pcl::PointCloud<pcl::PointXYZ> mesh_cloud,cam_cloud;
    pcl::fromPCLPointCloud2(texture_mesh_.polygon_mesh.cloud,mesh_cloud);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int view = 0; view <cam2world_.size(); view++)
    {
        pcl::transformPointCloud(mesh_cloud,cam_cloud,cam2world_[view].cast<float>().inverse());
        std::vector<bool> visible_faces;
        visible_faces.resize(texture_mesh_.polygon_mesh.polygons.size());
        pcl::PointCloud<pcl::PointXY>::Ptr projections (new pcl::PointCloud<pcl::PointXY>);
        std::vector<VertFace> uv_indexes;
        pcl::PointXY nan_point;
        nan_point.x = std::numeric_limits<float>::quiet_NaN ();
        nan_point.y = std::numeric_limits<float>::quiet_NaN ();
        VertFace u_null;
        u_null.vertex_id = -1;
        u_null.face_id = -1;

        int cpt_invisible=0;
        for (int idx_face = 0; idx_face <  static_cast<int> (texture_mesh_.polygon_mesh.polygons.size ()); ++idx_face)
        {
            Eigen::Vector2d uv_coord0;
            Eigen::Vector2d uv_coord1;
            Eigen::Vector2d uv_coord2;
            pcl::PointXYZ pt0 = mesh_cloud.points[texture_mesh_.polygon_mesh.polygons[idx_face].vertices[0]];
            pcl::PointXYZ pt1 = mesh_cloud.points[texture_mesh_.polygon_mesh.polygons[idx_face].vertices[1]];
            pcl::PointXYZ pt2 = mesh_cloud.points[texture_mesh_.polygon_mesh.polygons[idx_face].vertices[2]];
            Eigen::Vector3d global_pt0(pt0.x,pt0.y,pt0.z);
            Eigen::Vector3d global_pt1(pt1.x,pt1.y,pt1.z);
            Eigen::Vector3d global_pt2(pt2.x,pt2.y,pt2.z);
            //project each vertice, if one is out of view, stop
            if (faceProjected(image_height_,image_width_,intrins_,cam2world_[view].inverse().matrix(),global_pt0,global_pt1,global_pt2,uv_coord0,uv_coord1,uv_coord2))
            {

#ifdef _OPENMP
#pragma omp critical
#endif
                {
                    // add UV coordinates
                    pcl::PointXY uv0,uv1,uv2;
                    uv0.x =static_cast<float>(uv_coord0(0)); uv0.y =static_cast<float>(uv_coord0(1));
                    uv1.x =static_cast<float>(uv_coord1(0)); uv1.y = static_cast<float>(uv_coord1(1));
                    uv2.x =static_cast<float>(uv_coord2(0)); uv2.y = static_cast<float>(uv_coord2(1));
                    projections->points.push_back (uv0);
                    projections->points.push_back (uv1);
                    projections->points.push_back (uv2);
                }
                VertFace u1, u2, u3;
                u1.vertex_id = texture_mesh_.polygon_mesh.polygons[idx_face].vertices[0];
                u2.vertex_id = texture_mesh_.polygon_mesh.polygons[idx_face].vertices[1];
                u3.vertex_id = texture_mesh_.polygon_mesh.polygons[idx_face].vertices[2];
                u1.face_id = idx_face; u2.face_id = idx_face; u3.face_id = idx_face;
#ifdef _OPENMP
#pragma omp critical
#endif
                {
                    uv_indexes.push_back (u1);
                    uv_indexes.push_back (u2);
                    uv_indexes.push_back (u3);
                }

                visible_faces[idx_face] = true;

            }
            else
            {
#ifdef _OPENMP
#pragma omp critical
#endif
                {
                    projections->points.push_back (nan_point);
                    projections->points.push_back (nan_point);
                    projections->points.push_back (nan_point);
                    uv_indexes.push_back (u_null);
                    uv_indexes.push_back (u_null);
                    uv_indexes.push_back (u_null);
                }
                //keep track of visibility
                visible_faces[idx_face] = false;
                cpt_invisible++;
            }
        }


        // TODO handle case were no face could be projected
        if (texture_mesh_.polygon_mesh.polygons.size() - cpt_invisible !=0)
        {
            //create kdtree
            pcl::KdTreeFLANN<pcl::PointXY> kdtree;
            kdtree.setInputCloud (projections);
            std::vector<int> idxNeighbors;
            std::vector<float> neighborsSquaredDistance;

            for (int idx_face = 0; idx_face <  static_cast<int> (texture_mesh_.polygon_mesh.polygons.size ()); ++idx_face)
            {
                if (!visible_faces[idx_face])
                {

                    continue;
                }
                pcl::PointXY uv_coord1;
                pcl::PointXY uv_coord2;
                pcl::PointXY uv_coord3;
                // face is in the camera's FOV
                uv_coord1=projections->points[idx_face*3 + 0];
                uv_coord2=projections->points[idx_face*3 + 1];
                uv_coord3=projections->points[idx_face*3 + 2];

                double radius;
                pcl::PointXY center;
                getTriangleCircumscribedCircleCentroid(uv_coord1, uv_coord2, uv_coord3, center, radius); // this function yields faster results than getTriangleCircumcenterAndSize
                if (kdtree.radiusSearch (center, radius, idxNeighbors, neighborsSquaredDistance) > 0 )
                {
                    for (size_t i = 0; i < idxNeighbors.size (); ++i)
                    {
                        if (std::max (cam_cloud.points[texture_mesh_.polygon_mesh.polygons[idx_face].vertices[0]].z,
                                      std::max (cam_cloud.points[texture_mesh_.polygon_mesh.polygons[idx_face].vertices[1]].z,
                                                cam_cloud.points[texture_mesh_.polygon_mesh.polygons[idx_face].vertices[2]].z))
                                < cam_cloud.points[uv_indexes[idxNeighbors[i]].vertex_id].z)
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

        cv::Mat depth(image_height_,image_width_,CV_16UC1);
        cv::Mat mask(image_height_,image_width_,CV_8UC1);
        cv::Mat vertex_map(image_height_,image_width_,CV_32FC3);

        depth = 0;
        mask = 0;
        vertex_map = 0;
        pcl::PointCloud<pcl::PointXYZ> cloud;
        cloud.width = depth.cols;
        cloud.height = depth.rows;
        cloud.is_dense = true;
        cloud.points.resize(depth.cols* depth.rows,pcl::PointXYZ(0,0,0));
        for(std::size_t idx_face =0;idx_face<texture_mesh_.polygon_mesh.polygons.size();idx_face++)
        {
            if(visible_faces[idx_face])
            {
                //vertices
                Eigen::Vector2d pixel0,pixel1,pixel2;


                pcl::PointXYZ pt0 = mesh_cloud.points[texture_mesh_.polygon_mesh.polygons[idx_face].vertices[0]];
                pcl::PointXYZ pt1 = mesh_cloud.points[texture_mesh_.polygon_mesh.polygons[idx_face].vertices[1]];
                pcl::PointXYZ pt2 = mesh_cloud.points[texture_mesh_.polygon_mesh.polygons[idx_face].vertices[2]];

                Eigen::Vector3d global_p0(pt0.x,pt0.y,pt0.z);
                Eigen::Vector3d global_p1(pt1.x,pt1.y,pt1.z);
                Eigen::Vector3d global_p2(pt2.x,pt2.y,pt2.z);


                if(faceProjected(image_height_,image_width_,intrins_,cam2world_[view].inverse().matrix(),
                                 global_p0,global_p1,global_p2
                                 ,pixel0,pixel1,pixel2))
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
                    for(int x=min_x;x<=max_x;x++)
                    {
                        for(int y=min_y;y<=max_y;y++)
                        {
                            Eigen::Vector2d uv(x,y);
                            Eigen::Vector3d bcoords;
                            if(isInside(uv,pixel0,pixel1,pixel2,bcoords))
                            {
                                Eigen::Vector3d global_pt_uv = bcoords[0] * global_p0 + bcoords[1] * global_p1 +bcoords[2] * global_p2 ;
                                Eigen::Vector3d cam_pt_uv;
                                cam_pt_uv = globalToCameraSpace(global_pt_uv,cam2world_[view].inverse().matrix());
                                depth.at<unsigned short>(y,x) = static_cast<unsigned short>(cam_pt_uv[2]*1000);
                                vertex_map.at<cv::Vec3f>(y,x) = cv::Vec3f(cam_pt_uv[0],cam_pt_uv[1],cam_pt_uv[2]);

                                mask.at<uchar>(y,x) = 255;
                                pcl::PointXYZ pt;
                                pt.x = cam_pt_uv[0];
                                pt.y = cam_pt_uv[1];
                                pt.z = cam_pt_uv[2];
                                int index = y * cloud.width + x;
                                cloud.points[index] = pt;
                            }
                        }
                    }

                }
            }
        }
        std::ostringstream curr_frame_prefix;
        curr_frame_prefix<<view;
        //curr_frame_prefix << std::setw(6) << std::setfill('0') << view;
        cv::imwrite(object_data_path_+"/frame-"+curr_frame_prefix.str()+".depth.png",depth);
        // cv::imwrite(data_path_+"/frame-"+curr_frame_prefix.str()+".mask.png",mask);
        // pcl::io::savePLYFile(data_path_+"/frame-"+curr_frame_prefix.str()+"cloud.ply",cloud);
        saveVertexMap(vertex_map,object_data_path_+object_name_+"_"+curr_frame_prefix.str()+"_coloredPC_xyz_32F3.bin");
    }
}
bool ObjectRendering::interpolate(const Eigen::Vector2d& p, const Eigen::Vector2d& v0, const Eigen::Vector2d& v1, const Eigen::Vector2d& v2,Eigen::Vector3d& bcoords)
{
    Eigen::Vector2d e1 = v1 - v0, e2 = v2 - v0, e0 = p - v0;
    double e12 = e1[0] * e2[1] - e1[1] * e2[0];  // e1 x e2
    if (fabs(e12) < 1e-8)
    {  // triangle is degenerate: two edges are almost colinear
        bcoords = Eigen::Vector3d(0,0,0);
        return false;
    }
    bcoords[1] = (e0[0] * e2[1] - e0[1] * e2[0]) / e12;  // e0 x e2
    bcoords[2] = (e1[0] * e0[1] - e1[1] * e0[0]) / e12;  // e1 x e0
    bcoords[0] = 1 - bcoords[1] - bcoords[2];
    return  (bcoords.minCoeff()>=0.0);
}


bool ObjectRendering::getPixelCoords4GlobalPt(const int height, const int width,
                                              const Eigen::Matrix3d& intrins,Eigen::Vector2d& pixel, const Eigen::Matrix4d& world_to_cam,const Eigen::Vector3d& global_pt)
{
    Eigen::Affine3d a;
    a.matrix() = world_to_cam;
    Eigen::Vector3d uv= intrins *a * global_pt;
    pixel(0) = uv(0)/uv(2) ;
    pixel(1) = uv(1)/uv(2) ;
    double w = pixel(0)/width;
    double h = 1.0 - pixel(1)/height;
    if(w >= 0.001 && w< 0.999 && h >= 0.001 && h < 0.999)
        return true;
    else
        return false;


}

bool ObjectRendering::faceProjected(const int height, const int width, const Eigen::Matrix3d& intrins,const Eigen::Matrix4d& world_to_cam,
                                    const Eigen::Vector3d &global_pt0, const Eigen::Vector3d &global_pt1,
                                    const Eigen::Vector3d &global_pt2, Eigen::Vector2d &pixel0,
                                    Eigen::Vector2d &pixel1, Eigen::Vector2d &pixel2)
{
    return (getPixelCoords4GlobalPt(height,width,intrins,pixel0,world_to_cam,global_pt0)
            &&
            getPixelCoords4GlobalPt(height,width,intrins,pixel1,world_to_cam,global_pt1)
            &&
            getPixelCoords4GlobalPt(height,width,intrins,pixel2,world_to_cam,global_pt2)
            );
}
Eigen::Vector3d ObjectRendering::globalToCameraSpace(const Eigen::Vector3d& pt, const Eigen::Matrix4d& world_to_cam)
{
    return (world_to_cam * Eigen::Vector4d(pt[0], pt[1], pt[2], 1.0)).head<3>();
}
Eigen::Vector2d ObjectRendering::cameraToImgSpace(const Eigen::Vector3d& pt, const Eigen::Matrix3d& intrins)
{
    return Eigen::Vector2d(intrins(0,0) * pt[0] / pt[2] + intrins(0,2), intrins(1,1) * pt[1] / pt[2] + intrins(1,2));
}

void ObjectRendering::getTriangleCircumscribedCircleCentroid ( const pcl::PointXY &p1, const pcl::PointXY &p2, const pcl::PointXY &p3, pcl::PointXY &circumcenter, double &radius)
{
    circumcenter.x = static_cast<float> (p1.x + p2.x + p3.x ) / 3;
    circumcenter.y = static_cast<float> (p1.y + p2.y + p3.y ) / 3;

    double r1 = (circumcenter.x - p1.x) * (circumcenter.x - p1.x) + (circumcenter.y - p1.y) * (circumcenter.y - p1.y)  ;
    double r2 = (circumcenter.x - p2.x) * (circumcenter.x - p2.x) + (circumcenter.y - p2.y) * (circumcenter.y - p2.y)  ;
    double r3 = (circumcenter.x - p3.x) * (circumcenter.x - p3.x) + (circumcenter.y - p3.y) * (circumcenter.y - p3.y)  ;
    radius = std::sqrt( std::max( r1, std::max( r2, r3) )) ;
}

bool ObjectRendering::checkPointInsideTriangle(const pcl::PointXY &p1, const pcl::PointXY &p2, const pcl::PointXY &p3, const pcl::PointXY &pt)
{
    // Compute vectors
    Eigen::Vector2d v0, v1, v2;
    v0(0) = p3.x - p1.x; v0(1) = p3.y - p1.y; // v0= C - A
    v1(0) = p2.x - p1.x; v1(1) = p2.y - p1.y; // v1= B - A
    v2(0) = pt.x - p1.x; v2(1) = pt.y - p1.y; // v2= P - A

    // Compute dot products
    double dot00 = v0.dot(v0); // dot00 = dot(v0, v0)
    double dot01 = v0.dot(v1); // dot01 = dot(v0, v1)
    double dot02 = v0.dot(v2); // dot02 = dot(v0, v2)
    double dot11 = v1.dot(v1); // dot11 = dot(v1, v1)
    double dot12 = v1.dot(v2); // dot12 = dot(v1, v2)

    // Compute barycentric coordinates
    double invDenom = 1.0 / (dot00*dot11 - dot01*dot01);
    double u = (dot11*dot02 - dot01*dot12) * invDenom;
    double v = (dot00*dot12 - dot01*dot02) * invDenom;

    // Check if point is in triangle
    return ((u >= 0) && (v >= 0) && (u + v < 1));
}

bool ObjectRendering::isInside(const Eigen::Vector2d& p, const Eigen::Vector2d& v0, const Eigen::Vector2d& v1, const Eigen::Vector2d& v2,Eigen::Vector3d& bcoords)
{
    Eigen::Vector2d e1 = v1 - v0, e2 = v2 - v0, e0 = p - v0;
    double e12 = e1[0] * e2[1] - e1[1] * e2[0];  // e1 x e2
    if (fabs(e12) < 1e-8)
    {  // triangle is degenerate: two edges are almost colinear
        bcoords = Eigen::Vector3d(0,0,0);
        return false;
    }
    bcoords[1] = (e0[0] * e2[1] - e0[1] * e2[0]) / e12;  // e0 x e2
    bcoords[2] = (e1[0] * e0[1] - e1[1] * e0[0]) / e12;  // e1 x e0
    bcoords[0] = 1 - bcoords[1] - bcoords[2];
    return  (bcoords.minCoeff()>=0.0);
}


void ObjectRendering::saveOdometryIPA()
{


    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromPCLPointCloud2(texture_mesh_.polygon_mesh.cloud,cloud);
    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(cloud, min_pt, max_pt);
    //get 3d bounding box
    Eigen::Vector3f bb(max_pt.x-min_pt.x,max_pt.y-min_pt.y,max_pt.z-min_pt.z);
    //save odometry to file
    std::string odo_file = object_data_path_+object_name_+"_pt_odo.txt";
    std::string num_cap_file = object_data_path_ +object_name_+ "_info.txt";
    std::string bb_file = object_data_path_ + "BB.txt";
    std::ofstream bb_f;
    bb_f.open(bb_file.c_str());
    bb_f<<1;
    bb_f<<"\n";
    bb_f<<0;
    bb_f<<"\t";
    bb_f<<object_name_;
    bb_f<<"\t";
    bb_f<<bb(0)/2;
    bb_f<<"\t";
    bb_f<<bb(1)/2;
    bb_f<<"\t";
    bb_f<<bb(2);
    bb_f.close();


    std::ofstream info_f;
    info_f.open(num_cap_file.c_str());
    info_f<<cam2world_.size();
    info_f.close();
    std::ofstream f;
    f.open(odo_file.c_str());
    f<<cam2world_.size()<<"\n";
    for(std::size_t i=0;i<cam2world_.size();i++)
    {
        Eigen::Matrix4d cam2bottom;
        Eigen::Matrix3d rot = cam2world_[i].matrix().block<3,3>(0,0);
        Eigen::Quaterniond q(rot);
        Eigen::Vector3d trans = cam2world_[i].matrix().block<3,1>(0,3);
        f << trans(0) << " ";
        f << trans(1) << " ";
        f << trans(2) << " ";
        f << q.w() << " ";
        f << q.x() << " ";
        f << q.y() << " ";
        f << q.z() << "\n";
    }
    f.close();

    std::string cam_info_file = object_data_path_+object_name_+"_intrinsic.txt";
    FILE *fp = fopen(cam_info_file.c_str(), "w");

    fprintf(fp, "%15.8e\t %15.8e\t %15.8e\t\n", intrins_(0,0), 0.0f, intrins_(0,2));
    fprintf(fp, "%15.8e\t %15.8e\t %15.8e\t\n", 0.0f,intrins_(1,1), intrins_(1,2));
    fprintf(fp, "%15.8e\t %15.8e\t %15.8e\t\n", 0.0f, 0.0f, 1.0f);
    fclose(fp);

}


int main(int argc, char** argv)
{
    if(argc < 3)
    {
        std::cout <<"Usage : ./object_rendering data_path object_name"<<std::endl;
        return EXIT_FAILURE;
    }

    std::string data_path = argv[1];
    std::string object_name = argv[2];
    ObjectRendering::Ptr rendering = ObjectRendering::Ptr (new ObjectRendering);

    if(!rendering->readData(data_path,object_name))
    {
        std::cout<<"exit program..."<<std::endl;
    }
    initGLFW();
    rendering->saveTextureImage();
    rendering->saveDepthImage();
    rendering->saveOdometryIPA();
    return 0;
}
