cmake_minimum_required(VERSION 2.8.3)
project(ros_mesh_rgbd_rendering)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
# We need a CMAKE_DIR with some code to find external dependencies
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_BUILD_TYPE Release)
# set catkin packages that are needed when running the code
set(catkin_RUN_PACKAGES
    geometry_msgs
    message_generation
    message_runtime
    pcl_ros
    roscpp
    roslib
    ros_object_detection_msgs
    sensor_msgs
    std_msgs
    )

set(catkin_BUILD_PACKAGES
    cmake_modules
    ${catkin_RUN_PACKAGES}

    )
find_package(catkin REQUIRED COMPONENTS ${catkin_BUILD_PACKAGES})
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
find_package(Boost REQUIRED COMPONENTS system chrono thread)
find_package(Eigen3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLFW3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
link_directories(${PCL_LIBRARY_DIRS})
find_package(yaml-cpp)


# tinyobjectloader
include(ExternalProject)
add_subdirectory(elibs)
include_directories(${CMAKE_SOURCE_DIR}/elibs)

add_service_files(
    FILES
    MeshRGBDRendering.srv
    ViewGenerator.srv
    )
## Generate added messages and services
generate_messages(DEPENDENCIES
    geometry_msgs
    std_msgs)

catkin_package(
    INCLUDE_DIRS common/include
    CATKIN_DEPENDS ${catkin_RUN_PACKAGES}
    DEPENDS
    LIBRARIES
    )

include_directories(
    common/include
    ${Boost_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
    )


add_executable(ros_mesh_rgbd_rendering_service_node ros/ros_mesh_rgbd_rendering_service_node.cpp)
add_dependencies(ros_mesh_rgbd_rendering_service_node
                 ${catkin_EXPORTED_TARGETS}
                )
target_link_libraries(ros_mesh_rgbd_rendering_service_node
    ${catkin_LIBRARIES}
    ${EIGEN3_INCLUDE_DIRS}
    ${GLFW_LIBRARIES}
    ${GLUT_LIBRARY}
    ${GLEW_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${PCL_LIBRARIES}
    tinyobjloader
    ${YAML_CPP_LIBRARIES}
    )

add_executable(mesh_depth_recover_service_node ros/mesh_depth_recover_service_node.cpp)
add_dependencies(mesh_depth_recover_service_node
                 ${catkin_EXPORTED_TARGETS}
                )
target_link_libraries(mesh_depth_recover_service_node
    ${catkin_LIBRARIES}
    ${EIGEN3_INCLUDE_DIRS}
    ${GLFW_LIBRARIES}
    ${GLUT_LIBRARY}
    ${GLEW_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${PCL_LIBRARIES}
    tinyobjloader
    ${YAML_CPP_LIBRARIES}
    )


add_executable(view_generator ros/view_generator.cpp)
add_dependencies(view_generator
                 ${catkin_EXPORTED_TARGETS}
                )
target_link_libraries(view_generator
    ${catkin_LIBRARIES}
    ${EIGEN3_INCLUDE_DIRS}
    ${PCL_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
    )

add_executable(camera_pose_visualizer ros/camera_pose_visualizer.cpp)
target_link_libraries(camera_pose_visualizer
    ${catkin_LIBRARIES}
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_LIBRARIES}
    ${PCL_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
    )







