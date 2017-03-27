cmake_minimum_required(VERSION 2.8)
project( sDrive )
find_package( OpenCV REQUIRED )
add_executable( sDrive main.cpp )
target_link_libraries(sDrive ${OpenCV_LIBS} )
//sample
