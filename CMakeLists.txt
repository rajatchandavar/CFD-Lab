# Definition of the minimum required cmake Version
cmake_minimum_required(VERSION 3.0)
# Definition of the Project
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
# Later you can access the project variable like ${CFDLAB_SOURCE_DIR}
project(CFDLAB VERSION 1.0)

# Define all configuration options
option(gpp9 "compile with gpp9 filesystem" ON)
add_compile_options(-g)

# Definition of the C++ Standard 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CUDA_COMPILE_FLAGS "-arch=sm_86")
enable_language(CUDA)

# Creating the executable of our project and the required dependencies
# the executable is called fluG
file(GLOB files src/*.cpp)
file(GLOB cuda_files src/*.cu)
add_executable(fluG ${files} ${cuda_files})

set_property(TARGET fluG PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_compile_definitions(fluG PUBLIC -Dsolution_liddriven)
target_compile_definitions(fluG PUBLIC -Dsolution_energy)
target_compile_definitions(fluG PUBLIC -Dsolution_parallelization)


# You can find package likes
# find_package(MPI)
# Require a package
find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)


# Find a package with different components e.g. BOOST
# find_package(Boost COMPONENTS filesystem REQUIRED)

# VTK Library
find_package(VTK REQUIRED)
message (STATUS "VTK_VERSION: ${VTK_VERSION}")
include(${VTK_USE_FILE})

# Filesystem library is only available since GCC 9
# (if you are using a different compiler, comment-out these lines -- contributions welcome!)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9")
    message("g++ Version is lower than Version 9")
    if (NOT APPLE)  
      target_link_libraries(fluG PUBLIC stdc++fs)
    endif()
    else()
    message("g++ Version is 9 or higher")
    target_compile_definitions(fluG PUBLIC gpp9)
    target_compile_definitions(fluG PUBLIC -DGCC_VERSION_9_OR_HIGHER)
  endif()
endif()

# Add include directory
target_include_directories(fluG PUBLIC include)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX /usr/local)
endif()

# if you use external libraries you have to link them like
target_link_libraries(fluG PRIVATE MPI::MPI_CXX)
target_link_libraries(fluG PRIVATE ${VTK_LIBRARIES})
target_link_libraries(fluG ${CUDA_RUNTIME_LIBRARY})

install(TARGETS fluG DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
# If you write tests, you can include your subdirectory (in this case tests) as done here
# Testing