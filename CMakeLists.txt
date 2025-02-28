cmake_minimum_required(VERSION 3.10)
project(USB_Camera_Capture)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O0 -march=native -ggdb -g")

# 查找OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 查找OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

# Python
find_package(Python COMPONENTS Development NumPy REQUIRED)
include_directories(${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})

# 可执行文件
add_executable(usb_cam src/usb_cam.cpp)
add_executable(usb_cam_class src/usb_cam_class.cc)
add_executable(latency_test src/latency_test.cc)
add_executable(test_queue src/test_queue.cc)
add_executable(opencl_pipeline src/opencl_pipeline.cc)
add_executable(opencl_sync src/opencl_sync.cc)

# 链接库
target_link_libraries(usb_cam
    ${OpenCV_LIBS}
    ${Python_LIBRARIES}
    pthread
    rt
)
target_link_libraries(usb_cam_class
    ${OpenCV_LIBS}
    ${OpenCL_LIBRARIES}
    ${Python_LIBRARIES}
    pthread
    rt
)
target_link_libraries(latency_test
    ${OpenCV_LIBS}
    ${Python_LIBRARIES}
    pthread
    rt
)
target_link_libraries(test_queue
    ${OpenCV_LIBS}
    ${OpenCL_LIBRARIES}
    ${Python_LIBRARIES}
    pthread
    rt
)
target_link_libraries(opencl_pipeline
    ${OpenCV_LIBS}
    ${OpenCL_LIBRARIES}
    ${Python_LIBRARIES}
    pthread
    rt
)
target_link_libraries(opencl_sync
    ${OpenCV_LIBS}
    ${OpenCL_LIBRARIES}
    ${Python_LIBRARIES}
    pthread
    rt
)
