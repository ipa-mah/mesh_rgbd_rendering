cmake_minimum_required (VERSION 3.1)

project (object_rendering)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Release")
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(PCL REQUIRED)
find_package(jsoncpp REQUIRED)
find_package(glfw3 REQUIRED)
link_directories(${PCL_LIBRARY_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${jsoncpp_INCLUDE_DIR}
    )

include_directories(glm)
file(GLOB sources "${PROJECT_SOURCE_DIR}/sources/*.cpp")
add_executable(${PROJECT_NAME} object_rendering.cpp ${sources})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARY} ${GLEW_LIBRARIES} ${jsoncpp_LIBRARIES} glfw pthread)

add_executable(object_visualizer object_visualizer.cpp)
target_link_libraries(object_visualizer ${PCL_LIBRARIES} ${OpenCV_LIBS} ${jsoncpp_LIBRARIES})

add_executable(view_generator view_generator.cpp)
target_link_libraries(view_generator ${PCL_LIBRARIES} ${OpenCV_LIBS} ${jsoncpp_LIBRARIES} )
