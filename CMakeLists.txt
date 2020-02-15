cmake_minimum_required (VERSION 3.1)

project (mesh_rgbd_rendering)

# build with c++14
add_compile_options(-std=c++14)
set(CMAKE_BUILD_TYPE Release)


find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(PCL REQUIRED)
#find_package(yaml-cpp REQUIRED)
find_package(jsoncpp REQUIRED)
find_package(glfw3 REQUIRED)
link_directories(${PCL_LIBRARY_DIRS})
include_directories(
    common/include
    ${OpenCV_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${jsoncpp_INCLUDE_DIR}
    glm
    )

file(GLOB sources "${PROJECT_SOURCE_DIR}/common/src/*.cpp")
add_library(${PROJECT_NAME} ${sources})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARY} ${GLEW_LIBRARIES} ${jsoncpp_LIBRARIES} glfw pthread)

add_executable(mesh_rgbd_rendering_main src/mesh_rgbd_rendering.cpp )
target_link_libraries(mesh_rgbd_rendering_main ${PROJECT_NAME})

add_executable(view_visualizer src/view_visualizer.cpp)
target_link_libraries(view_visualizer ${PCL_LIBRARIES} ${OpenCV_LIBS} ${jsoncpp_LIBRARIES})

add_executable(view_generator src/view_generator.cpp)
target_link_libraries(view_generator ${PCL_LIBRARIES} ${OpenCV_LIBS} ${jsoncpp_LIBRARIES} )
