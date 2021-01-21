cmake_minimum_required(VERSION 3.17)
project(fast_fourier)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES sources/fast_fourier.cpp headers/fast_fourier.h)

add_library(fast_fourier_lib SHARED ${SOURCE_FILES})
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(fast_fourier_lib OpenMP::OpenMP_CXX)
endif ()

add_executable(fast_fourier main.cpp )

TARGET_LINK_LIBRARIES(fast_fourier fast_fourier_lib)

