cmake_minimum_required(VERSION 2.8.12)

PROJECT(RectangularTargetDetection）

include_directories(/usr/local/include)

link_directories(/usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)

set(NEED_SO libopencv_core.so libopencv_highgui.so libopencv_imgcodecs.so libopencv_imgproc.so)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

ADD_EXECUTABLE(main main.cpp)

TARGET_LINK_LIBRARIES(main ${NEED_SO})
