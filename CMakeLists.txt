cmake_minimum_required(VERSION 3.15)
project(AKS_proekt)
set(CMAKE_CXX_STANDARD 17)

#set(CMAKE_CXX_COMPILER "mpic++")                           # for mpi
#set(Boost_USE_MULTITHREADED TRUE)                          # for mpi
#find_package(Boost COMPONENTS mpi serialization REQUIRED)  # for mpi

#include_directories(/usr/local/include/opencv4)
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLM REQUIRED)

include_directories(${OpenCV_LIBS})
include_directories(${GLM_INCLUDE_DIRS})
#include_directories(/usr/local/include/GLFW)
#include_directories(/usr/local/include/GL)
find_package(Threads REQUIRED)


include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})

add_executable(reconstruct main.cpp helping_functions/threadsafe_queue.h helping_functions/config_parser.h)
add_executable(reconstruct_t treads.cpp helping_functions/threadsafe_queue.h helping_functions/config_parser.h)

add_executable(visualize visualize.cpp)

target_link_libraries(reconstruct ${OpenCV_LIBS})
target_link_libraries(reconstruct_t ${OpenCV_LIBS} Threads::Threads)

target_link_libraries(visualize ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLFW_LIBRARIES} ${GLM_LIBRARIES})

#add_executable(mpi mpi.cpp helping_functions/config_parser.h)     # for mpi
#target_link_libraries(mpi ${OpenCV_LIBS} ${Boost_LIBRARIES})      # for mpi
